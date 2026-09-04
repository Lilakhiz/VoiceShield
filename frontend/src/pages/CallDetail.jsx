import { useEffect, useState } from 'react'
import { api } from '../lib/api'
import TrustContinuity from '../components/TrustContinuity'

export default function CallDetail({ callId, onBack }) {
  const [state, setState] = useState('loading') // 'loading' | 'success' | 'error' | 'empty'
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false
    setState('loading')
    setError(null)
    setData(null)
    
    api.getCall(callId)
      .then((result) => {
        if (cancelled) return
        if (!result || !result.call) {
          setState('empty')
          return
        }
        setData(result)
        setState('success')
      })
      .catch((e) => {
        if (cancelled) return
        setError(e.message)
        setState('error')
      })

    return () => { cancelled = true }
  }, [callId])

  if (state === 'loading') {
    return <div className="text-dim text-sm font-mono">Loading…</div>
  }

  if (state === 'error') {
    return (
      <div className="space-y-4">
        <button onClick={onBack} className="text-dim text-sm hover:text-ink">← Back to call history</button>
        <div className="bg-panel border border-line p-5">
          <div className="font-sans text-sm text-rust tracking-wide mb-2">Error loading call details</div>
          <div className="text-dim text-sm font-mono">{error || 'Unknown error'}</div>
          <button 
            onClick={() => setState('loading')} 
            className="mt-3 bg-amber text-void text-sm font-semibold px-4 py-2"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  if (state === 'empty') {
    return (
      <div className="space-y-4">
        <button onClick={onBack} className="text-dim text-sm hover:text-ink">← Back to call history</button>
        <div className="bg-panel border border-line p-5">
          <div className="font-sans text-sm text-dim tracking-wide">No call data found</div>
          <div className="text-dim text-sm font-mono mt-1">The call may have been deleted or never existed.</div>
        </div>
      </div>
    )
  }

  const { call, snapshots } = data

  return (
    <div className="space-y-6">
      <button onClick={onBack} className="text-dim text-sm hover:text-ink">← Back to call history</button>

      <div className="bg-panel border border-line p-5">
        <h2 className="font-sans text-sm text-dim tracking-wide mb-2">CALL {call.id.slice(0, 8)}</h2>
        <div className="font-mono text-sm text-dim">
          Started {new Date(call.started_at).toLocaleString()}
          {call.claimed_speaker_id && <> · claimed identity: {call.claimed_speaker_id}</>}
          {call.demo_scenario && <> · demo scenario: {call.demo_scenario}</>}
        </div>
      </div>

      <div className="bg-panel border border-line p-5">
        <h2 className="font-sans text-sm text-dim tracking-wide mb-3">TRUST CONTINUITY</h2>
        <TrustContinuity history={snapshots} />
      </div>

      <div className="bg-panel border border-line p-5">
        <h2 className="font-sans text-sm text-dim tracking-wide mb-3">FULL SNAPSHOT LOG</h2>
        <div className="space-y-4">
          {snapshots.map((s, i) => (
            <div key={i} className="border border-line p-3 font-mono text-xs">
              <div className="flex justify-between text-dim mb-2">
                <span>t={s.elapsed_seconds.toFixed(1)}s</span>
                <span>risk {s.risk_score} · trust {s.trust_score} · {s.risk_level}</span>
              </div>
              {s.language?.transcript && (
                <div className="text-ink mb-2">"{s.language.transcript}"</div>
              )}
              <ul className="space-y-1">
                {s.reasons.map((r, j) => (
                  <li key={j} className="text-dim">
                    {r.delta < 0 ? '▼' : '▲'} {r.detail}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
