import { useEffect, useState } from 'react'
import { api } from '../lib/api'

const LEVEL_COLOR = { LOW: '#4FD1C5', MEDIUM: '#E8C34D', HIGH: '#E2574C', CRITICAL: '#E2574C' }

export default function CallHistory({ onOpenCall }) {
  const [calls, setCalls] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.listCalls().then(setCalls).catch(() => {}).finally(() => setLoading(false))
  }, [])

  return (
    <div className="bg-panel border border-line p-5">
      <h2 className="font-sans text-sm text-dim tracking-wide mb-4">CALL HISTORY</h2>
      {loading && <div className="text-dim text-sm font-mono">Loading…</div>}
      {!loading && calls.length === 0 && (
        <div className="text-dim text-sm font-mono">
          No calls yet. Run a demo scenario or start a live call from the Dashboard.
        </div>
      )}
      {calls.length > 0 && (
        <table className="w-full text-sm font-mono">
          <thead className="text-dim text-xs border-b border-line">
            <tr>
              <th className="text-left py-2">Started</th>
              <th className="text-left py-2">Claimed identity</th>
              <th className="text-left py-2">Scenario</th>
              <th className="text-left py-2">Final risk</th>
              <th className="text-left py-2"></th>
            </tr>
          </thead>
          <tbody>
            {calls.map((c) => (
              <tr key={c.id} className="border-b border-line hover:bg-panel2">
                <td className="py-2 text-dim">{new Date(c.started_at).toLocaleString()}</td>
                <td className="py-2 text-ink">{c.claimed_speaker_id || '—'}</td>
                <td className="py-2 text-dim">{c.demo_scenario || 'live'}</td>
                <td className="py-2">
                  {c.final_risk_level ? (
                    <span style={{ color: LEVEL_COLOR[c.final_risk_level] }}>
                      {c.final_risk_score} · {c.final_risk_level}
                    </span>
                  ) : '—'}
                </td>
                <td className="py-2">
                  <button onClick={() => onOpenCall(c.id)}
                    className="text-amber hover:underline">View details</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
