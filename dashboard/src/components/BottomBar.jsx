import React from 'react';

const VAR_CONFIG = {
  t2m: { min: 280, max: 318, label: 'TEMPERATURE (K)', gradient: 'linear-gradient(90deg, #1e40af, #3b82f6, #fde68a, #f97316, #dc2626)' },
  tp: { min: 0, max: 15, label: 'PRECIPITATION (mm/hr)', gradient: 'linear-gradient(90deg, #f0f9ff, #7dd3fc, #0ea5e9, #1d4ed8, #312e81)' },
  wind_speed: { min: 0, max: 25, label: 'WIND SPEED (m/s)', gradient: 'linear-gradient(90deg, #ecfdf5, #6ee7b7, #059669, #064e3b)' },
  alert_level: { min: 0, max: 2, label: 'ALERT LEVEL', gradient: 'linear-gradient(90deg, #052e16, #92400e, #7f1d1d)' }
};

export default function BottomBar({ variable }) {
  const config = VAR_CONFIG[variable] || VAR_CONFIG['t2m'];

  return (
    <div 
      className="glass-panel"
      style={{
        position: 'absolute', bottom: 0, left: 0, width: '100%', height: '40px',
        borderBottom: 'none', borderLeft: 'none', borderRight: 'none', borderRadius: 0,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '0 20px', boxSizing: 'border-box', zIndex: 1000
      }}
    >
      {/* LEFT: LEGEND */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <span className="mono" style={{ fontSize: '10px', color: 'rgba(255,255,255,0.6)' }}>{config.min}</span>
        <div style={{ width: '120px', height: '6px', borderRadius: '3px', background: config.gradient }}></div>
        <span className="mono" style={{ fontSize: '10px', color: 'rgba(255,255,255,0.6)' }}>{config.max}</span>
        <span style={{ fontSize: '10px', color: 'rgba(255,255,255,0.8)', marginLeft: '8px', fontWeight: 'bold' }}>
          {config.label}
        </span>
      </div>

      {/* CENTER: CREDITS */}
      <div style={{ fontSize: '10px', color: 'rgba(255,255,255,0.4)', letterSpacing: '1px' }}>
        SIH 2025 · Problem 26081 · NCMRWF / MoES
      </div>

      {/* RIGHT: MODEL COLORS */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px', fontSize: '10px', color: 'rgba(255,255,255,0.8)' }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}><span style={{ color: '#3b82f6' }}>●</span> GFS</span>
        <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}><span style={{ color: '#eab308' }}>●</span> ECMWF</span>
        <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}><span style={{ color: '#a855f7' }}>●</span> PANGU</span>
        <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}><span style={{ color: '#ef4444' }}>●</span> GRAPHCAST</span>
        <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}><span style={{ color: '#4ade80' }}>●</span> BLEND</span>
      </div>
    </div>
  );
}
