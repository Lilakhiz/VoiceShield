function colorFor(value, invert) {
  const v = invert ? 1 - value : value
  if (v >= 0.75) return '#E2574C'
  if (v >= 0.45) return '#E8C34D'
  return '#4FD1C5'
}

/**
 * A horizontal instrument meter with tick marks -- used for every
 * probability/score in the Live Call view instead of a rounded
 * "SaaS card" stat tile, to read like lab/telecom test equipment.
 *
 * value: 0..1
 * invert: true when a HIGH value is GOOD (e.g. speaker similarity)
 */
export default function Meter({ label, value, invert = false, suffix = '' }) {
  const pct = Math.round(Math.max(0, Math.min(1, value)) * 100)
  const color = colorFor(value, invert)
  return (
    <div>
      <div className="flex items-baseline justify-between mb-1">
        <span className="text-dim text-xs tracking-wide font-sans">{label}</span>
        <span className="font-mono text-sm" style={{ color }}>
          {pct}{suffix}
        </span>
      </div>
      <div className="meter-track">
        <div className="meter-fill" style={{ width: `${pct}%`, backgroundColor: color }} />
        <div className="meter-ticks" />
      </div>
    </div>
  )
}
