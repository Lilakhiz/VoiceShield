import { useEffect, useRef, useState, useCallback } from 'react'
import { api } from '../lib/api'
import { useCallSocket } from '../lib/useCallSocket'
import Meter from '../components/Meter'
import RiskBadge from '../components/RiskBadge'
import TrustContinuity from '../components/TrustContinuity'

const SAFE_VERIFICATION_QUESTIONS = [
  'What is the name of the street we grew up on?',
  'What did we eat at your last birthday dinner?',
  "What's the name of my first pet?",
]

const AUDIO_WORKLET_URL = '/audio-processor.js'
const CHUNK_SIZE = 1600 // 100ms at 16kHz
const TARGET_SAMPLE_RATE = 16000

function useMicCapture(onChunk, active, onError) {
  const ctxRef = useRef(null)
  const streamRef = useRef(null)
  const workletNodeRef = useRef(null)
  const sourceRef = useRef(null)

  useEffect(() => {
    if (!active) return
    let stopped = false

    async function start() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
          audio: {
            sampleRate: TARGET_SAMPLE_RATE,
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true
          }
        })
        streamRef.current = stream
        
        const AudioContext = window.AudioContext || window.webkitAudioContext
        const ctx = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE })
        ctxRef.current = ctx

        // Load and register the AudioWorklet
        await ctx.audioWorklet.addModule(AUDIO_WORKLET_URL)
        
        const source = ctx.createMediaStreamSource(stream)
        sourceRef.current = source
        
        const workletNode = new AudioWorkletNode(ctx, 'audio-processor', {
          processorOptions: {
            chunkSize: CHUNK_SIZE,
            sampleRate: TARGET_SAMPLE_RATE
          }
        })
        workletNodeRef.current = workletNode

        // Handle messages from the worklet
        workletNode.port.onmessage = (event) => {
          if (stopped) return
          const msg = event.data
          if (msg.type === 'audio_chunk') {
            onChunk(msg.pcm_f32, msg.sample_rate)
          }
        }

        // Connect: source -> worklet -> destination (for monitoring)
        source.connect(workletNode)
        workletNode.connect(ctx.destination)
      } catch (err) {
        console.error('AudioWorklet initialization failed:', err)
        // Fallback to ScriptProcessorNode if AudioWorklet fails
        await startFallback()
      }
    }

    async function startFallback() {
      // LEGACY COMPATIBILITY PATH - ScriptProcessorNode is deprecated
      // This fallback exists only for older browsers that don't support AudioWorklet
      // (Chrome < 66, Firefox < 76, Safari < 14.1, Edge < 79)
      // Primary path is AudioWorklet above; this fallback will be removed when
      // legacy browser support is no longer required.
      console.warn('Falling back to deprecated ScriptProcessorNode (legacy browser)')
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
        streamRef.current = stream
        const AudioContext = window.AudioContext || window.webkitAudioContext
        const ctx = new AudioContext()
        ctxRef.current = ctx
        const source = ctx.createMediaStreamSource(stream)
        const processor = ctx.createScriptProcessor(4096, 1, 1)
        source.connect(processor)
        processor.connect(ctx.destination)
        processor.onaudioprocess = (e) => {
          if (stopped) return
          const input = e.inputBuffer.getChannelData(0)
          // Resample to 16kHz if needed
          let samples = input
          if (ctx.sampleRate !== TARGET_SAMPLE_RATE) {
            // Simple decimation for fallback (not ideal but functional)
            const ratio = ctx.sampleRate / TARGET_SAMPLE_RATE
            const newLength = Math.floor(input.length / ratio)
            samples = new Float32Array(newLength)
            for (let i = 0; i < newLength; i++) {
              samples[i] = input[Math.floor(i * ratio)]
            }
          }
          onChunk(Array.from(samples), TARGET_SAMPLE_RATE)
        }
      } catch (err) {
        console.error('Fallback mic capture also failed:', err)
        onError?.(err)
      }
    }

    start().catch((err) => {
      onError?.(err)
    })

    return () => {
      stopped = true
      
      // Flush any remaining audio in the worklet
      if (workletNodeRef.current) {
        workletNodeRef.current.port.postMessage({ type: 'flush' })
        workletNodeRef.current.disconnect()
        workletNodeRef.current = null
      }
      
      if (sourceRef.current) {
        sourceRef.current.disconnect()
        sourceRef.current = null
      }
      
      streamRef.current?.getTracks().forEach((t) => t.stop())
      streamRef.current = null
      
      ctxRef.current?.close()
      ctxRef.current = null
    }
  }, [active, onChunk, onError])
}

export default function LiveCall() {
  const [mode, setMode] = useState(null) // 'live' | 'demo'
  const [speakers, setSpeakers] = useState([])
  const [claimedSpeaker, setClaimedSpeaker] = useState('')
  const [scenarios, setScenarios] = useState([])
  const [micError, setMicError] = useState(null)
  const { connect, send, close, connected, callId, latest, history, error: wsError, readyForNewCall, reset } = useCallSocket()

  useEffect(() => {
    api.listSpeakers().then(setSpeakers).catch(() => {})
    api.listDemoScenarios().then(setScenarios).catch(() => {})
  }, [])

  const chunkHandler = useCallback((samples, sampleRate) => {
    send({ type: 'audio_chunk', sample_rate: sampleRate, pcm_f32: samples })
  }, [send])
  useMicCapture(chunkHandler, mode === 'live' && connected, handleError)

  const startLive = () => {
    // Clear all errors when starting a new call
    setMicError(null)
    reset()
    setMode('live')
    connect('/ws/call', () => {
      send({ type: 'start', claimed_speaker_id: claimedSpeaker || null })
    })
  }

  const startDemo = (key) => {
    // Clear all errors when starting a new demo
    setMicError(null)
    reset()
    setMode('demo')
    connect(`/ws/demo/${key}`)
  }

  const stop = () => {
    if (mode === 'live') send({ type: 'end' })
    close()
    setMode(null)
    setMicError(null)
  }

  const handleError = useCallback((err) => {
    console.error('Live call error:', err)
    if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
      setMicError('Microphone permission denied. Please allow microphone access and try again.')
    } else if (err.name === 'NotFoundError') {
      setMicError('No microphone found. Please connect a microphone and try again.')
    } else if (err.name === 'NotReadableError') {
      setMicError('Microphone is in use by another application.')
    } else {
      setMicError(err.message || 'Failed to start microphone')
    }
    setMode(null)
    reset()
  }, [reset])

  const risk = latest ?? { risk_score: 0, risk_level: 'LOW', trust_score: 100, reasons: [] }
  
  // Separate error display for mic vs WebSocket
  const hasMicError = micError !== null
  const hasWsError = wsError !== null
  const displayError = micError || wsError

  return (
    <div className="grid grid-cols-3 gap-6">
      {/* Left: controls + transcript */}
      <div className="col-span-2 space-y-6">
        <div className="bg-panel border border-line p-5">
          <h2 className="font-sans text-sm text-dim tracking-wide mb-3">SESSION CONTROL</h2>
          {!mode ? (
            <div className="space-y-4">
              {displayError && (
                <div className="bg-rust/10 border border-rust p-3 text-rust text-sm font-mono">
                  {displayError}
                  <button 
                    onClick={() => { 
                      setMicError(null); 
                      // wsError is managed by useCallSocket, user can retry by starting new call
                    }}
                    className="ml-3 bg-amber text-void text-xs font-semibold px-3 py-1"
                  >
                    Dismiss
                  </button>
                </div>
              )}
              <div className="flex flex-wrap items-center gap-3">
                <select
                  value={claimedSpeaker}
                  onChange={(e) => setClaimedSpeaker(e.target.value)}
                  className="bg-void border border-line text-ink text-sm px-3 py-2 font-mono"
                  disabled={hasMicError || hasWsError}
                >
                  <option value="">Claimed identity: unknown</option>
                  {speakers.map((s) => (
                    <option key={s.id} value={s.id}>Claimed identity: {s.name}</option>
                  ))}
                </select>
                <button onClick={startLive}
                  className="bg-amber text-void font-sans font-semibold text-sm px-4 py-2 hover:opacity-90"
                  disabled={!readyForNewCall || hasMicError || hasWsError}
                >
                  Start live call (microphone)
                </button>
              </div>
              <div className="border-t border-line pt-4">
                <div className="text-dim text-xs mb-2 font-sans">Or run a demo scenario</div>
                <div className="flex flex-wrap gap-2">
                  {scenarios.map((s) => (
                    <button key={s.key} onClick={() => startDemo(s.key)}
                      className="border border-line text-ink text-xs font-mono px-3 py-2 hover:border-amber hover:text-amber"
                      disabled={!readyForNewCall || hasMicError || hasWsError}>
                      {s.label}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-between">
              <div className="font-mono text-sm text-dim">
                <span className="live-dot text-amber mr-2">●</span>
                {mode === 'live' ? 'Live microphone call' : 'Demo scenario'} — call {callId?.slice(0, 8)}
              </div>
              <button onClick={stop}
                className="border border-rust text-rust text-sm font-sans px-4 py-2 hover:bg-rust hover:text-void">
                End call
              </button>
            </div>
          )}
        </div>

        <div className="bg-panel border border-line p-5">
          <h2 className="font-sans text-sm text-dim tracking-wide mb-3">LIVE TRANSCRIPT</h2>
          <div className="font-mono text-sm space-y-2 min-h-[120px]">
            {history.length === 0 && <div className="text-dim">Waiting for speech…</div>}
            {history.map((s, i) => (
              s.language?.transcript && (
                <div key={i} className="text-ink">
                  <span className="text-dim mr-2">[{s.language.language}]</span>
                  {s.language.transcript}
                </div>
              )
            ))}
          </div>
        </div>

        <div className="bg-panel border border-line p-5">
          <h2 className="font-sans text-sm text-dim tracking-wide mb-3">TRUST CONTINUITY</h2>
          <TrustContinuity history={history} />
        </div>
      </div>

      {/* Right: instrument panels */}
      <div className="space-y-6">
        <div className="bg-panel border border-line p-5 flex justify-center">
          <RiskBadge level={risk.risk_level} score={risk.risk_score} />
        </div>

        <div className="bg-panel border border-line p-5 space-y-4">
          <h2 className="font-sans text-sm text-dim tracking-wide">SIGNAL PANEL</h2>
          <Meter label="Speaker match" value={latest?.speaker?.similarity ?? 0} invert />
          <Meter label="AI-generated voice probability" value={latest?.deepfake?.ai_generated_probability ?? 0} />
          <Meter label="Replay / recorded-audio probability" value={latest?.replay?.replay_probability ?? 0} />
          <Meter label="Urgency language" value={latest?.threat?.urgency_score ?? 0} />
        </div>

        <div className="bg-panel border border-line p-5">
          <h2 className="font-sans text-sm text-dim tracking-wide mb-3">SENSITIVE-REQUEST ALERTS</h2>
          {(!latest?.threat?.detected_types || latest.threat.detected_types.every(t => t === 'NONE')) ? (
            <div className="text-dim text-sm font-mono">No sensitive requests detected</div>
          ) : (
            <ul className="space-y-1 font-mono text-sm">
              {latest.threat.detected_types.filter(t => t !== 'NONE').map((t) => (
                <li key={t} className="text-rust">⚠ {t.replaceAll('_', ' ')}</li>
              ))}
            </ul>
          )}
        </div>

        {risk.suggest_safe_verification && (
          <div className="border border-rust bg-panel p-5">
            <h2 className="font-sans text-sm text-rust tracking-wide mb-2">SAFE VERIFICATION SUGGESTED</h2>
            <p className="text-dim text-sm mb-3">
              Risk is elevated. Ask a question only the real person would know before continuing.
            </p>
            <div className="font-mono text-sm text-ink border-l-2 border-rust pl-3">
              {SAFE_VERIFICATION_QUESTIONS[history.length % SAFE_VERIFICATION_QUESTIONS.length]}
            </div>
          </div>
        )}

        {latest && (
          <div className="bg-panel border border-line p-5">
            <h2 className="font-sans text-sm text-dim tracking-wide mb-3">WHY THIS SCORE</h2>
            <ul className="space-y-2 font-mono text-xs">
              {latest.reasons.map((r, i) => (
                <li key={i} className="text-dim">
                  <span className={r.delta < 0 ? 'text-rust' : 'text-cyan'}>
                    {r.delta < 0 ? '▼' : '▲'} {r.delta.toFixed(1)}
                  </span>{' '}
                  {r.detail}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  )
}
