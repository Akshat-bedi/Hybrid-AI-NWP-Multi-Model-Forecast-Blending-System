import React from 'react';

const VARIABLES = [
  { id: 't2m', label: 'TEMPERATURE' },
  { id: 'tp', label: 'PRECIPITATION' },
  { id: 'wind_speed', label: 'WIND SPEED' },
  { id: 'alert_level', label: 'ALERT LEVEL' }
];

const LEAD_HOURS = [0, 24, 48, 72, 96, 120];

export default function ControlPanel({ variable, setVariable, leadHours, setLeadHours }) {
  const getDayLabel = (h) => {
    if (h === 0) return 'CURRENT';
    return `DAY ${Math.floor(h/24) + 1} — ${h}h`;
  };

  return (
    <div 
      className="glass-panel" 
      style={{ position: 'fixed', left: '16px', top: '72px', width: '180px', padding: '16px', zIndex: 1000 }}
    >
      <div style={{ fontSize: '10px', color: 'rgba(255,255,255,0.4)', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '12px' }}>
        VARIABLE
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
        {VARIABLES.map(v => {
          const isActive = variable === v.id;
          return (
            <button
              key={v.id}
              onClick={() => setVariable(v.id)}
              style={{
                width: '100%',
                height: '36px',
                borderRadius: '8px',
                border: isActive ? '1px solid rgba(56,189,248,0.5)' : '1px solid rgba(255,255,255,0.05)',
                background: isActive ? 'linear-gradient(135deg, rgba(56,189,248,0.3), rgba(129,140,248,0.3))' : 'transparent',
                color: isActive ? '#38bdf8' : 'rgba(255,255,255,0.4)',
                textShadow: isActive ? '0 0 10px rgba(56,189,248,0.5)' : 'none',
                cursor: 'pointer',
                fontWeight: isActive ? 'bold' : 'normal',
                fontSize: '11px',
                transition: 'all 0.2s',
              }}
              onMouseEnter={(e) => {
                if (!isActive) e.target.style.background = 'rgba(255,255,255,0.05)';
              }}
              onMouseLeave={(e) => {
                if (!isActive) e.target.style.background = 'transparent';
              }}
            >
              {v.label}
            </button>
          );
        })}
      </div>

      <div style={{ height: '1px', background: 'rgba(255,255,255,0.1)', margin: '16px 0' }}></div>

      <div style={{ fontSize: '10px', color: 'rgba(255,255,255,0.4)', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '8px' }}>
        LEAD TIME
      </div>
      
      <div style={{ fontSize: '14px', fontWeight: 'bold', color: '#38bdf8', marginBottom: '12px' }}>
        {getDayLabel(leadHours)}
      </div>

      <input 
        type="range" 
        min="0" 
        max="120" 
        step="24" 
        value={leadHours} 
        onChange={(e) => setLeadHours(parseInt(e.target.value))}
        className="custom-slider"
      />
      
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '6px' }}>
        {LEAD_HOURS.map(h => (
          <span key={h} style={{ fontSize: '9px', color: 'rgba(255,255,255,0.4)' }}>{h}h</span>
        ))}
      </div>

      <div style={{ height: '1px', background: 'rgba(255,255,255,0.1)', margin: '16px 0' }}></div>

      <div style={{ fontSize: '10px', color: 'rgba(255,255,255,0.4)', textAlign: 'center', marginBottom: '4px' }}>
        990 pts · 0.25°
      </div>
      <div style={{ fontSize: '10px', color: 'rgba(255,255,255,0.4)', textAlign: 'center' }}>
        India Domain
      </div>
    </div>
  );
}
