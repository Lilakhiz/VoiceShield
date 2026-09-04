import { useEffect, useState } from 'react'
import { api } from '../lib/api'

const DEFAULTS = {
  verificationThreshold: 0.5,
  highRiskThreshold: 55,
  criticalRiskThreshold: 80,
}

const SETTING_KEYS = {
  verificationThreshold: 'speaker_verification_threshold',
  highRiskThreshold: 'risk_high_threshold',
  criticalRiskThreshold: 'risk_critical_threshold',
}

export default function Settings() {
  const [settings, setSettings] = useState(DEFAULTS)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [saveStatus, setSaveStatus] = useState(null) // null, 'success', 'error'

  // Load settings from backend
  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setSaveStatus(null)
    
    api.getSettings()
      .then(data => {
        if (cancelled) return
        // Map backend keys to frontend state keys
        const newSettings = { ...DEFAULTS }
        if (data?.speaker_verification_threshold?.value !== undefined) {
          newSettings.verificationThreshold = Number(data.speaker_verification_threshold.value)
        }
        if (data?.risk_high_threshold?.value !== undefined) {
          newSettings.highRiskThreshold = Number(data.risk_high_threshold.value)
        }
        if (data?.risk_critical_threshold?.value !== undefined) {
          newSettings.criticalRiskThreshold = Number(data.risk_critical_threshold.value)
        }
        setSettings(newSettings)
      })
      .catch(err => {
        if (cancelled) return
        console.error('Failed to load settings:', err)
        // Keep defaults on error
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    
    return () => { cancelled = true }
  }, [])

  const update = (key) => (e) => {
    const value = Number(e.target.value)
    setSettings((s) => ({ ...s, [key]: value }))
    setSaveStatus(null)
  }

  const save = async () => {
    setSaving(true)
    setSaveStatus(null)
    
    try {
      // Save each setting
      await Promise.all([
        api.updateSetting('speaker_verification_threshold', settings.verificationThreshold),
        api.updateSetting('risk_high_threshold', settings.highRiskThreshold),
        api.updateSetting('risk_critical_threshold', settings.criticalRiskThreshold),
      ])
      setSaveStatus('success')
    } catch (err) {
      console.error('Failed to save settings:', err)
      setSaveStatus('error')
    } finally {
      setSaving(false)
    }
  }

  if (loading) {
    return (
      <div className="grid grid-cols-2 gap-6">
        <div className="bg-panel border border-line p-5">
          <div className="text-dim text-sm font-mono animate-pulse">Loading settings…</div>
        </div>
      </div>
    )
  }

  return (
    <div className="grid grid-cols-2 gap-6">
      <div className="bg-panel border border-line p-5">
        <div className="flex items-center justify-between mb-4">
          <h2 className="font-sans text-sm text-dim tracking-wide">RISK THRESHOLDS</h2>
          <button
            onClick={save}
            disabled={saving}
            className="bg-amber text-void text-xs font-semibold px-3 py-1 disabled:opacity-50"
          >
            {saving ? 'Saving…' : 'Save Changes'}
          </button>
        </div>
        {saveStatus === 'success' && (
          <div className="mb-4 text-cyan text-xs font-mono">Settings saved successfully</div>
        )}
        {saveStatus === 'error' && (
          <div className="mb-4 text-rust text-xs font-mono">Failed to save settings</div>
        )}
        <div className="space-y-5 font-mono text-sm">
          <div>
            <div className="flex justify-between mb-1">
              <span className="text-dim">Speaker verification cutoff (cosine similarity)</span>
              <span className="text-amber">{settings.verificationThreshold.toFixed(2)}</span>
            </div>
            <input type="range" min="0" max="1" step="0.01" value={settings.verificationThreshold}
              onChange={update('verificationThreshold')} className="w-full accent-amber" />
          </div>
          <div>
            <div className="flex justify-between mb-1">
              <span className="text-dim">HIGH risk threshold</span>
              <span className="text-yellow">{settings.highRiskThreshold}</span>
            </div>
            <input type="range" min="0" max="100" value={settings.highRiskThreshold}
              onChange={update('highRiskThreshold')} className="w-full accent-yellow" />
          </div>
          <div>
            <div className="flex justify-between mb-1">
              <span className="text-dim">CRITICAL risk threshold</span>
              <span className="text-rust">{settings.criticalRiskThreshold}</span>
            </div>
            <input type="range" min="0" max="100" value={settings.criticalRiskThreshold}
              onChange={update('criticalRiskThreshold')} className="w-full accent-rust" />
          </div>
          <p className="text-dim text-xs pt-2 border-t border-line">
            These map to the backend's <code>SPEAKER_VERIFICATION_THRESHOLD</code> env var and
            the risk-level cutoffs in <code>app/services/risk_engine.py</code>. Changes take
            effect immediately for new calls.
          </p>
        </div>
      </div>

      <div className="bg-panel border border-line p-5">
        <h2 className="font-sans text-sm text-dim tracking-wide mb-4">MODEL STATUS</h2>
        <ul className="space-y-3 font-mono text-sm">
          <li className="flex justify-between border-b border-line pb-2">
            <span className="text-dim">Speech-to-text / language ID</span>
            <span className="text-dim">OpenAI Whisper (small)</span>
          </li>
          <li className="flex justify-between border-b border-line pb-2">
            <span className="text-dim">Speaker verification</span>
            <span className="text-dim">SpeechBrain ECAPA-TDNN</span>
          </li>
          <li className="flex justify-between border-b border-line pb-2">
            <span className="text-dim">Deepfake detection</span>
            <span className="text-dim">Signal-heuristic fallback (pretrained hook available)</span>
          </li>
          <li className="flex justify-between border-b border-line pb-2">
            <span className="text-dim">Replay detection</span>
            <span className="text-dim">Spectral-heuristic (offline)</span>
          </li>
          <li className="flex justify-between">
            <span className="text-dim">Sensitive-request NLP</span>
            <span className="text-dim">Rule-based (5 languages) + optional zero-shot ML</span>
          </li>
        </ul>
        <p className="text-dim text-xs pt-3 mt-3 border-t border-line">
          Live model availability is reported per-call via each signal's
          <code> available</code> flag — check the backend logs if a
          panel reads 0 unexpectedly.
        </p>
      </div>
    </div>
  )
}
