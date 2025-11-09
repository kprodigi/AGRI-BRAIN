import React, { useEffect, useState } from 'react'
import Ops from './tabs/Ops'
import Quality from './tabs/Quality'
import Decisions from './tabs/Decisions'
// Optional: uncomment next line to expose the Admin tools as a 4th tab
// import AdminPanel from './mvp/AdminPanel.jsx'

const API = (window.API_BASE || localStorage.getItem('API_BASE') || 'http://127.0.0.1:8100').replace(/\/$/, '')

export default function App() {
    // Add 'Admin' to show the Admin tab; or remove it to keep just the original three
    const tabs = ['Operations', 'Quality', 'Decisions' /*, 'Admin'*/]
    const [tab, setTab] = useState('Operations')

    // keep your original warm-up call
    useEffect(() => {
        fetch(`${API}/case/load`, { method: 'POST' }).catch(() => { })
    }, [])

    return (
        <div className='max-w-7xl mx-auto p-6 space-y-6'>
            <header className='flex items-center justify-between'>
                <h1 className='text-3xl font-bold'>AGRI-BRAIN — Spinach</h1>
                <div className='text-sm'>API: {API}</div>
            </header>

            {/* Pills with active (black) styling — uses the .nav-tabs CSS you added */}
            <nav className='nav-tabs'>
                {tabs.map((t) => (
                    <a
                        key={t}
                        href='#'
                        className={tab === t ? 'active' : ''}
                        onClick={(e) => {
                            e.preventDefault()
                            setTab(t)
                        }}
                    >
                        {t}
                    </a>
                ))}
            </nav>

            {tab === 'Operations' && <Ops API={API} />}
            {tab === 'Quality' && <Quality API={API} />}
            {tab === 'Decisions' && <Decisions API={API} />}

            {/* Uncomment to render Admin inside your tabs */}
            {/* {tab === 'Admin' && <AdminPanel />} */}
        </div>
    )
}
