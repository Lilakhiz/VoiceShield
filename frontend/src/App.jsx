import { useState } from 'react'
import LiveCall from './pages/LiveCall.jsx'
import TrustedSpeakers from './pages/TrustedSpeakers.jsx'
import CallHistory from './pages/CallHistory.jsx'
import CallDetail from './pages/CallDetail.jsx'
import Settings from './pages/Settings.jsx'

const NAV = [
  { key: 'live', label: 'Live Call', glyph: '◉' },
  { key: 'speakers', label: 'Trusted Speakers', glyph: '☰' },
  { key: 'history', label: 'Call History', glyph: '▤' },
  { key: 'settings', label: 'Settings', glyph: '⚙' },
]

export default function App() {
  const [page, setPage] = useState('live')
  const [openCallId, setOpenCallId] = useState(null)

  const goHistory = () => { setOpenCallId(null); setPage('history') }

  return (
    <div className="min-h-screen flex">
      <aside className="w-56 shrink-0 bg-panel border-r border-line flex flex-col">
        <div className="px-5 py-6 border-b border-line">
          <div className="font-mono text-amber text-lg leading-none">VOICEGUARD</div>
          <div className="text-dim text-[11px] mt-1 font-sans">SIH26104 · Impersonation Console</div>
        </div>
        <nav className="flex-1 py-4">
          {NAV.map((item) => (
            <button
              key={item.key}
              onClick={() => { setPage(item.key); setOpenCallId(null) }}
              className={`w-full text-left px-5 py-3 flex items-center gap-3 font-sans text-sm border-l-2 ${
                page === item.key
                  ? 'border-amber text-ink bg-panel2'
                  : 'border-transparent text-dim hover:text-ink'
              }`}
            >
              <span className="font-mono">{item.glyph}</span>
              {item.label}
            </button>
          ))}
        </nav>
        <div className="px-5 py-4 border-t border-line text-dim text-[11px] font-mono">
          Backend: <span className="text-cyan">connected</span>
        </div>
      </aside>

      <main className="flex-1 p-8 grid-texture bg-void">
        {page === 'live' && <LiveCall />}
        {page === 'speakers' && <TrustedSpeakers />}
        {page === 'history' && !openCallId && <CallHistory onOpenCall={setOpenCallId} />}
        {page === 'history' && openCallId && (
          <CallDetail callId={openCallId} onBack={goHistory} />
        )}
        {page === 'settings' && <Settings />}
      </main>
    </div>
  )
}
