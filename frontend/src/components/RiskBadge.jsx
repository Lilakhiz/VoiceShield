const STYLES = {
  LOW: { color: '#4FD1C5', label: 'LOW' },
  MEDIUM: { color: '#E8C34D', label: 'MEDIUM' },
  HIGH: { color: '#E2574C', label: 'HIGH' },
  CRITICAL: { color: '#E2574C', label: 'CRITICAL' },
}

export default function RiskBadge({ level, score }) {
  const s = STYLES[level] || STYLES.LOW
  return (
    <div className="flex items-center gap-3">
      <div
        className="w-24 h-24 rounded-full border-4 flex flex-col items-center justify-center font-mono"
        style={{ borderColor: s.color }}
      >
        <span className="text-2xl font-semibold" style={{ color: s.color }}>{score}</span>
        <span className="text-[10px] text-dim">/ 100</span>
      </div>
      <div>
        <div className="text-dim text-xs font-sans">Impersonation risk</div>
        <div className="font-mono text-lg font-semibold" style={{ color: s.color }}>
          {s.label}
          {level === 'CRITICAL' && <span className="live-dot ml-2">●</span>}
        </div>
      </div>
    </div>
  )
}
