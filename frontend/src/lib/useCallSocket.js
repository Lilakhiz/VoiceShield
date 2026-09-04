import { useCallback, useRef, useState, useEffect } from 'react'
import { wsUrl } from './api'

/**
 * Manages one WebSocket connection to either the live call endpoint
 * (/ws/call) or a demo scenario endpoint (/ws/demo/{key}), collecting
 * the stream of RiskSnapshot frames the backend pushes after each
 * processed audio chunk / scripted demo step.
 * 
 * Includes backpressure handling for outgoing audio messages.
 */
export function useCallSocket() {
  const [connected, setConnected] = useState(false)
  const [callId, setCallId] = useState(null)
  const [latest, setLatest] = useState(null)
  const [history, setHistory] = useState([])
  const [error, setError] = useState(null)
  const [readyForNewCall, setReadyForNewCall] = useState(true)
  const socketRef = useRef(null)
  const sendQueueRef = useRef([])
  const isSendingRef = useRef(false)
  const isClosingRef = useRef(false)
  const reconnectTimeoutRef = useRef(null)
  const connectPathRef = useRef(null)
  const connectOnStartRef = useRef(null)
  const maxQueueSize = 50 // Maximum queued audio messages
  const maxReconnectAttempts = 3
  const reconnectAttemptsRef = useRef(0)

  const reset = useCallback(() => {
    setLatest(null)
    setHistory([])
    setError(null)
    setCallId(null)
    setReadyForNewCall(false)
    sendQueueRef.current = []
    isSendingRef.current = false
    reconnectAttemptsRef.current = 0
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }
  }, [])

  const cleanupSocket = useCallback(() => {
    const ws = socketRef.current
    if (ws) {
      ws.onopen = null
      ws.onclose = null
      ws.onerror = null
      ws.onmessage = null
      ws.onbufferedamountlow = null
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        ws.close()
      }
      socketRef.current = null
    }
    isClosingRef.current = false
  }, [])

  const processQueue = useCallback(() => {
    const ws = socketRef.current
    const queue = sendQueueRef.current
    
    if (!ws || ws.readyState !== WebSocket.OPEN || queue.length === 0) {
      isSendingRef.current = false
      return
    }

    isSendingRef.current = true
    
    // Check backpressure - bufferedAmount indicates bytes queued in browser
    // If too much data is buffered, pause sending audio chunks
    if (ws.bufferedAmount > 64 * 1024) { // 64KB threshold
      // Schedule retry when buffer drains
      setTimeout(processQueue, 50)
      return
    }

    // Send next message
    const msg = queue.shift()
    try {
      ws.send(JSON.stringify(msg))
    } catch (e) {
      console.error('WebSocket send error:', e)
    }

    // Continue processing queue
    if (queue.length > 0) {
      // Yield to event loop to allow bufferedAmount to update
      setTimeout(processQueue, 0)
    } else {
      isSendingRef.current = false
    }
  }, [])

  const doConnect = useCallback((path, onStart) => {
    // Prevent duplicate connections
    if (socketRef.current && 
        (socketRef.current.readyState === WebSocket.OPEN || 
         socketRef.current.readyState === WebSocket.CONNECTING)) {
      console.warn('WebSocket connection already exists, closing previous')
      cleanupSocket()
    }

    isClosingRef.current = false
    connectPathRef.current = path
    connectOnStartRef.current = onStart
    reset()
    setReadyForNewCall(false)
    
    const ws = new WebSocket(wsUrl(path))
    socketRef.current = ws

    ws.onopen = () => {
      reconnectAttemptsRef.current = 0
      setConnected(true)
    }
    
    ws.onclose = (event) => {
      setConnected(false)
      isSendingRef.current = false
      setReadyForNewCall(true)
      
      // Only attempt reconnect if not intentionally closed and not a clean closure
      if (!isClosingRef.current && event.code !== 1000 && reconnectAttemptsRef.current < maxReconnectAttempts) {
        reconnectAttemptsRef.current++
        const delay = Math.min(1000 * Math.pow(2, reconnectAttemptsRef.current - 1), 10000)
        reconnectTimeoutRef.current = setTimeout(() => {
          if (!isClosingRef.current) {
            doConnect(connectPathRef.current, connectOnStartRef.current)
          }
        }, delay)
      }
    }
    
    ws.onerror = (event) => {
      console.error('WebSocket error:', event)
      // Don't set error here - onclose will handle it with proper code
    }

    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data)
        if (msg.type === 'started') {
          setCallId(msg.call_id)
          connectOnStartRef.current?.(msg.call_id)
        } else if (msg.type === 'snapshot') {
          setLatest(msg.data)
          setHistory((h) => [...h, msg.data])
        } else if (msg.type === 'error') {
          setError(msg.message)
        } else if (msg.type === 'ended') {
          setConnected(false)
          setReadyForNewCall(true)
        }
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e)
      }
    }

    // Handle drain event for backpressure
    ws.onbufferedamountlow = () => {
      if (!isSendingRef.current && sendQueueRef.current.length > 0) {
        processQueue()
      }
    }

    return ws
  }, [reset, processQueue, cleanupSocket])

  const connect = useCallback((path, onStart) => {
    return doConnect(path, onStart)
  }, [doConnect])

  const send = useCallback((obj) => {
    const ws = socketRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      return false
    }

    // Control messages (start, end) go directly, bypass queue
    const isControlMessage = obj.type === 'start' || obj.type === 'end'
    
    if (isControlMessage) {
      try {
        ws.send(JSON.stringify(obj))
        return true
      } catch (e) {
        console.error('WebSocket send error:', e)
        return false
      }
    }

    // Audio messages go through queue with backpressure
    if (sendQueueRef.current.length >= maxQueueSize) {
      // Drop oldest audio chunk to make room (coalesce)
      sendQueueRef.current.shift()
    }
    
    sendQueueRef.current.push(obj)
    
    if (!isSendingRef.current) {
      processQueue()
    }
    
    return true
  }, [processQueue])

  const close = useCallback(() => {
    isClosingRef.current = true
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }
    cleanupSocket()
    setConnected(false)
    setReadyForNewCall(true)
  }, [cleanupSocket])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      isClosingRef.current = true
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
      cleanupSocket()
    }
  }, [cleanupSocket])

  return { connect, send, close, connected, callId, latest, history, error, reset, readyForNewCall }
}
