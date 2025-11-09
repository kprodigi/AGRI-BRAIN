// frontend/src/main.jsx
import React from 'react'
import ReactDOM from 'react-dom/client'
import './index.css'

// ⬇️ If your root UI is elsewhere, change this import (e.g. './ui/App.jsx')
import App from './ui/App.jsx'
import AdminPanel from './mvp/AdminPanel.jsx'

import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import { ToastProvider } from './ui/components/UiPrimitives.jsx'
const router = createBrowserRouter([
    { path: '/', element: <App /> },          // Ops / Quality / Decisions
    { path: '/admin', element: <AdminPanel /> }
])

// ---------- Minimal on-page "Decision Memos" list (no component edits) ----------
function ensureMemoPanel() {
    const all = Array.from(document.querySelectorAll('h1,h2,h3,div,section,header,p,span,strong,b'))
    const heading = all.find(el => /(^|\s)decision\s+memos(\s|$)/i.test((el.textContent || '').trim()))
    let host

    if (heading?.parentElement) {
        host = heading.parentElement.querySelector('#memo-list')
        if (!host) {
            host = document.createElement('div')
            host.id = 'memo-list'
            host.style.marginTop = '8px'
            heading.parentElement.appendChild(host)
            console.log('[MVP] memo panel inserted after "Decision Memos" heading')
        }
    } else {
        host = document.getElementById('memo-list')
        if (!host) {
            host = document.createElement('div')
            host.id = 'memo-list'
            const root = document.getElementById('root')
            root && root.appendChild(host)
            console.log('[MVP] memo panel appended to #root (heading not found)')
        }
    }
    return host
}

function addMemoCard(m) {
    const host = ensureMemoPanel()
    if (!host) return
    const ts = new Date(((m.ts ?? Date.now() / 1000)) * 1000)

    const card = document.createElement('div')
    card.className = 'rounded-xl border border-gray-200 p-4 my-3 shadow-sm'
    card.style.borderRadius = '16px'
    card.innerHTML = `
    <div class="flex items-center justify-between">
      <div class="font-semibold">Agent: ${m.agent ?? 'farm'}</div>
      <div class="text-xs text-gray-500">${ts.toLocaleString()}</div>
    </div>
    <div class="mt-2">
      <span class="inline-block px-2 py-1 rounded-full bg-black text-white text-xs">${m.action ?? 'decision'}</span>
    </div>
    <dl class="mt-3 grid grid-cols-3 gap-4 text-sm">
      <div><dt class="text-gray-500">SLCA</dt><dd class="font-medium">${m.slca_score ?? '—'}</dd></div>
      <div><dt class="text-gray-500">CO₂</dt><dd class="font-medium">${m.carbon_kg ?? '—'} kg</dd></div>
      <div><dt class="text-gray-500">Tx hash</dt><dd class="font-mono">${m.tx_hash ?? '—'}</dd></div>
    </dl>
    <p class="mt-2 text-sm text-gray-700">${m.reason ?? ''}</p>
  `
    host.prepend(card) // newest first
}

document.addEventListener('decision:new', (e) => {
    addMemoCard(e.detail || {})
})

    // ---------- Global "Take decision" handler (safe, idempotent) ----------
    ; (function installTakeDecision() {
        if (window.__takeDecisionInstalled) return
        window.__takeDecisionInstalled = true

        const base = (window.API_BASE || localStorage.getItem('API_BASE') || 'http://127.0.0.1:8111').replace(/\/$/, '')
        console.log('[MVP] decision handler installed; API_BASE =', base)

        async function callAny() {
            const tries = [
                [`${base}/decision/take`, {}],
                [`${base}/decision/take`, { method: 'POST' }],
                [`${base}/decide`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' }],
                [`${base}/decisions/take`, {}],
                [`${base}/case/decide`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' }],
                [`${base}/api/decision/take`, { method: 'POST' }]
            ]
            for (const [url, init] of tries) {
                try { const r = await fetch(url, init); if (r.ok) return await r.json() } catch { }
            }
            throw new Error('No decision endpoint responded with 200')
        }

        document.addEventListener('click', async (e) => {
            const el = e.target.closest('button, a, [role="button"]')
            if (!el) return

            // ⬇️ Do NOT intercept clicks on the Admin route or buttons marked to skip
            if (location.pathname.startsWith('/admin')) return
            if (el.closest('[data-skip-global-take]')) return

            const label = (el.textContent || '').replace(/\s+/g, ' ').trim()
            if (!/^\s*take\s+decision\s*$/i.test(label)) return

            e.preventDefault()
            try {
                const memo = await callAny()
                console.log('Decision memo:', memo)
                document.dispatchEvent(new CustomEvent('decision:new', { detail: memo }))
            } catch (err) {
                console.warn(err)
                alert('Could not take decision. Check backend routes and API_BASE.')
            }
        }, true) // capture phase
    })()

    // ---------- Fix the "Download Decision Memo (PDF)" button ----------
    ; (function fixMemoDownloadButton() {
        const base = (window.API_BASE || localStorage.getItem('API_BASE') || 'http://127.0.0.1:8111').replace(/\/$/, '')
        const url = `${base}/report/pdf`  // ← use local report

        function attach() {
            const btn = Array.from(document.querySelectorAll('a,button'))
                .find(el => /download\s+decision\s+memo\s*\(pdf\)/i.test((el.textContent || '').trim()))
            if (!btn || btn.dataset.memoBound === '1') return

            if (btn.tagName === 'A') {
                btn.setAttribute('href', url)
                btn.setAttribute('target', '_blank')
                btn.setAttribute('rel', 'noopener')
            } else {
                btn.addEventListener('click', (e) => {
                    e.preventDefault()
                    window.open(url, '_blank', 'noopener')
                })
            }
            btn.dataset.memoBound = '1'
            console.log('[MVP] wired Download Decision Memo button →', url)
        }

        attach()
        window.addEventListener('load', attach)
        const mo = new MutationObserver(() => attach())
        mo.observe(document.documentElement, { childList: true, subtree: true })
    })()

// ---------- Mount app ----------
const rootEl = document.getElementById('root')
if (!rootEl) throw new Error('Root element #root not found')

ReactDOM.createRoot(rootEl).render(
    <ToastProvider>
        <RouterProvider router={router} />
    </ToastProvider>
)
