import { useEffect, useRef, useState } from 'react'
import { api } from '../lib/api'

function useRecorder() {
  const mediaRecorderRef = useRef(null)
  const chunksRef = useRef([])
  const [recording, setRecording] = useState(false)
  const [blob, setBlob] = useState(null)

  const start = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    const mr = new MediaRecorder(stream)
    chunksRef.current = []
    mr.ondataavailable = (e) => chunksRef.current.push(e.data)
    mr.onstop = () => {
      setBlob(new Blob(chunksRef.current, { type: 'audio/webm' }))
      stream.getTracks().forEach((t) => t.stop())
    }
    mr.start()
    mediaRecorderRef.current = mr
    setRecording(true)
  }

  const stop = () => {
    mediaRecorderRef.current?.stop()
    setRecording(false)
  }

  return { start, stop, recording, blob, reset: () => setBlob(null) }
}

export default function TrustedSpeakers() {
  const [speakers, setSpeakers] = useState([])
  const [name, setName] = useState('')
  const [speakerId, setSpeakerId] = useState('')
  const [status, setStatus] = useState(null)
  const { start, stop, recording, blob, reset } = useRecorder()

  const refresh = () => api.listSpeakers().then(setSpeakers).catch(() => {})
  useEffect(() => { refresh() }, [])

  const enroll = async () => {
    if (!blob || !name || !speakerId) return
    setStatus('enrolling')
    try {
      await api.enrollSpeaker(speakerId, name, blob)
      setStatus('done')
      setName(''); setSpeakerId(''); reset()
      refresh()
    } catch (e) {
      setStatus('error: ' + e.message)
    }
  }

  return (
    <div className="grid grid-cols-2 gap-6">
      <div className="bg-panel border border-line p-5">
        <h2 className="font-sans text-sm text-dim tracking-wide mb-4">ENROLL A TRUSTED SPEAKER</h2>
        <div className="space-y-3">
          <input
            placeholder="Speaker ID (e.g. dad, mom, manager_priya)"
            value={speakerId}
            onChange={(e) => setSpeakerId(e.target.value)}
            className="w-full bg-void border border-line text-ink text-sm px-3 py-2 font-mono"
          />
          <input
            placeholder="Display name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="w-full bg-void border border-line text-ink text-sm px-3 py-2 font-mono"
          />
          <div className="flex items-center gap-3">
            {!recording ? (
              <button onClick={start}
                className="bg-amber text-void text-sm font-semibold px-4 py-2">
                {blob ? 'Re-record' : 'Record voice sample'}
              </button>
            ) : (
              <button onClick={stop} className="border border-rust text-rust text-sm px-4 py-2">
                <span className="live-dot mr-2">●</span>Stop recording
              </button>
            )}
            {blob && !recording && (
              <audio controls src={URL.createObjectURL(blob)} className="h-8" />
            )}
          </div>
          <p className="text-dim text-xs">
            Speak naturally for 5–10 seconds. This clip becomes the reference
            voiceprint (ECAPA-TDNN embedding) used for verification during calls.
          </p>
          <button onClick={enroll} disabled={!blob || !name || !speakerId}
            className="bg-cyan text-void text-sm font-semibold px-4 py-2 disabled:opacity-30">
            Save voiceprint
          </button>
          {status && <div className="text-dim text-xs font-mono">{status}</div>}
        </div>
      </div>

      <div className="bg-panel border border-line p-5">
        <h2 className="font-sans text-sm text-dim tracking-wide mb-4">ENROLLED SPEAKERS</h2>
        {speakers.length === 0 ? (
          <div className="text-dim text-sm font-mono">No trusted speakers enrolled yet.</div>
        ) : (
          <table className="w-full text-sm font-mono">
            <thead className="text-dim text-xs border-b border-line">
              <tr>
                <th className="text-left py-2">Name</th>
                <th className="text-left py-2">ID</th>
                <th className="text-left py-2">Clips</th>
                <th className="text-left py-2">Enrolled</th>
              </tr>
            </thead>
            <tbody>
              {speakers.map((s) => (
                <tr key={s.id} className="border-b border-line">
                  <td className="py-2 text-ink">{s.name}</td>
                  <td className="py-2 text-dim">{s.id}</td>
                  <td className="py-2 text-dim">{s.enrollment_clips}</td>
                  <td className="py-2 text-dim">{new Date(s.created_at).toLocaleDateString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
