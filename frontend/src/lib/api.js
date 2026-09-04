const BASE = import.meta.env.VITE_API_BASE || ''
const DEFAULT_TIMEOUT = 10000 // 10 seconds

async function req(path, opts = {}) {
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), DEFAULT_TIMEOUT)
  
  try {
    const res = await fetch(`${BASE}${path}`, { ...opts, signal: controller.signal })
    clearTimeout(timeoutId)
    
    if (!res.ok) {
      const text = await res.text().catch(() => '')
      throw new Error(`${res.status} ${res.statusText}: ${text}`)
    }
    return res.json()
  } catch (e) {
    clearTimeout(timeoutId)
    if (e.name === 'AbortError') {
      throw new Error(`Request timeout after ${DEFAULT_TIMEOUT}ms`)
    }
    throw e
  }
}

export const api = {
  listSpeakers: () => req('/api/speakers'),
  enrollSpeaker: (speakerId, name, audioBlob) => {
    const form = new FormData()
    form.append('speaker_id', speakerId)
    form.append('name', name)
    form.append('audio', audioBlob, 'enroll.wav')
    return req('/api/speakers/enroll', { method: 'POST', body: form })
  },
  listCalls: () => req('/api/calls'),
  getCall: (id) => req(`/api/calls/${id}`),
  listDemoScenarios: () => req('/api/demo/scenarios'),
  
  // Settings
  getSettings: () => req('/api/settings'),
  updateSetting: (key, value) => req(`/api/settings/${key}`, { 
    method: 'PUT', 
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ value })
  }),
}

export function wsUrl(path) {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
  const base = import.meta.env.VITE_WS_BASE || `${proto}://${window.location.hostname}:8000`
  return `${base}${path}`
}
