function trustColor(score) {
  if (score >= 70) return '#4FD1C5'
  if (score >= 45) return '#E8C34D'
  return '#E2574C'
}

function fmtTime(seconds) {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

/**
 * Renders the call's trust-continuity history as a stepped readout,
 * mirroring the brief's own example format:
 *   00:10  Speaker verified        Trust: 94
 *   00:32  Voice anomaly detected  Trust: 78
 * Each row shows *why* the score moved, not just the number.
 */
export default function TrustContinuity({ history }) {
  if (!history.length) {
    return (
      <div className="text-dim text-sm font-mono py-6 text-center border border-dashed border-line">
        Trust continuity will appear here once the call starts.
      </div>
    )
  }

  return (
    <div className="font-mono text-sm">
      {/* sparkline-style bar strip */}
      <div className="flex items-end gap-[2px] h-16 mb-3">
        {history.map((snap, i) => {
          const h = Math.max(4, (snap.trust_score / 100) * 64)
          return (
            <div
              key={i}
              title={`${fmtTime(snap.elapsed_seconds)} — trust ${snap.trust_score}`}
              style={{ height: `${h}px`, backgroundColor: trustColor(snap.trust_score), width: '6px' }}
            />
          )
        })}
      </div>

      <div className="space-y-2 max-h-64 overflow-y-auto pr-1">
        {history.slice().reverse().map((snap, i) => {
          const topReason = snap.reasons[0]
          return (
            <div key={i} className="flex items-center gap-3 border-b border-line pb-2">
              <span className="text-dim w-12 shrink-0">{fmtTime(snap.elapsed_seconds)}</span>
              <span className="flex-1 text-ink truncate">
                {topReason ? topReason.detail : 'No change'}
              </span>
              <span className="shrink-0 font-semibold" style={{ color: trustColor(snap.trust_score) }}>
                Trust: {snap.trust_score}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
