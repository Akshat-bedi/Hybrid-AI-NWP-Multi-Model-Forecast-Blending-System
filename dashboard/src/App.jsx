import React, { useState, useEffect } from 'react';
import ForecastMap from './components/ForecastMap';
import ControlPanel from './components/ControlPanel';
import AlertPanel from './components/AlertPanel';
import SkillOverlay from './components/SkillOverlay';
import BottomBar from './components/BottomBar';
import { useHealth } from './hooks/useForecastData';

function App() {
  const [activeTab, setActiveTab] = useState('FORECAST'); // FORECAST, WEIGHTS, SKILL
  const [variable, setVariable] = useState('t2m'); // t2m, tp, wind_speed, alert_level
  const [leadHours, setLeadHours] = useState(72);
  const [toast, setToast] = useState(null);
  
  const { health } = useHealth();

  const handleSetToast = React.useCallback((msg) => {
    setToast(msg);
    setTimeout(() => setToast(null), 4000);
  }, []);

  return (
    <div style={{ position: 'relative', width: '100vw', height: '100vh', overflow: 'hidden' }}>
      
      {/* MAP LAYER */}
      <ForecastMap 
        activeTab={activeTab} 
        variable={variable} 
        leadHours={leadHours} 
        onError={React.useCallback(() => handleSetToast("API Unavailable — showing cached data"), [handleSetToast])}
      />

      {/* TOP HEADER BAR */}
      <div 
        className="glass-panel" 
        style={{
          position: 'absolute', top: 0, left: 0, width: '100%', height: '56px',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '0 20px', boxSizing: 'border-box', zIndex: 1000,
          borderTop: 'none', borderLeft: 'none', borderRight: 'none', borderRadius: 0
        }}
      >
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '18px' }}>☁️</span>
            <span style={{ fontWeight: 'bold', fontSize: '16px', letterSpacing: '1px' }}>MEGH VISION</span>
          </div>
          <span style={{ fontSize: '11px', color: 'rgba(255,255,255,0.6)' }}>Hybrid AI-NWP Forecast System</span>
        </div>

        <div style={{ display: 'flex', gap: '30px', height: '100%' }}>
          {['FORECAST', 'WEIGHTS', 'SKILL'].map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              style={{
                background: 'transparent',
                border: 'none',
                height: '100%',
                padding: '0 4px',
                color: activeTab === tab ? '#38bdf8' : 'rgba(255,255,255,0.4)',
                fontWeight: activeTab === tab ? 'bold' : 'normal',
                borderBottom: activeTab === tab ? '2px solid #38bdf8' : '2px solid transparent',
                cursor: 'pointer',
                letterSpacing: '1px',
                transition: 'all 0.2s',
                backgroundImage: activeTab === tab ? 'linear-gradient(90deg, #38bdf8, #818cf8)' : 'none',
                WebkitBackgroundClip: activeTab === tab ? 'text' : 'border-box',
                WebkitTextFillColor: activeTab === tab ? 'transparent' : 'rgba(255,255,255,0.4)'
              }}
            >
              {tab}
            </button>
          ))}
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end' }}>
          <div style={{ 
            display: 'flex', alignItems: 'center', gap: '6px', 
            background: health?.status === 'ok' ? 'rgba(0,255,100,0.1)' : 'rgba(251,191,36,0.1)',
            padding: '4px 10px', borderRadius: '12px'
          }}>
            <div style={{ 
              width: '8px', height: '8px', borderRadius: '50%', 
              background: health?.status === 'ok' ? 'var(--green)' : 'var(--amber)' 
            }}></div>
            <span style={{ 
              fontSize: '11px', fontWeight: 'bold', 
              color: health?.status === 'ok' ? 'var(--green)' : 'var(--amber)' 
            }}>
              {health?.status === 'ok' ? 'LIVE' : 'OFFLINE'}
            </span>
          </div>
          <span style={{ fontSize: '11px', color: 'rgba(255,255,255,0.4)', marginTop: '4px' }}>
            {health?.latest_run ? `Run: ${health.latest_run}` : 'Checking...'}
          </span>
        </div>
      </div>

      {/* LEFT CONTROL PANEL */}
      <ControlPanel 
        variable={variable} 
        setVariable={setVariable} 
        leadHours={leadHours} 
        setLeadHours={setLeadHours} 
      />

      {/* RIGHT ALERTS PANEL */}
      <AlertPanel />

      {/* SKILL OVERLAY (conditionally rendered) */}
      {activeTab === 'SKILL' && (
        <SkillOverlay variable={variable} leadHours={leadHours} />
      )}

      {/* ERROR TOAST */}
      {toast && (
        <div style={{
          position: 'fixed', bottom: '60px', left: '50%', transform: 'translateX(-50%)',
          background: 'rgba(248,113,113,0.15)', border: '1px solid rgba(248,113,113,0.3)',
          padding: '8px 16px', borderRadius: '8px', zIndex: 2000,
          fontSize: '12px', color: 'white', display: 'flex', alignItems: 'center', gap: '8px'
        }}>
          ⚠️ {toast}
        </div>
      )}

      {/* BOTTOM INFO BAR */}
      <BottomBar variable={variable} />
    </div>
  );
}

export default App;
