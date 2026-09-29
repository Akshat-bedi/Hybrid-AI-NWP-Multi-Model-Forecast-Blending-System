import React from 'react';
import { useAlerts } from '../hooks/useForecastData';
import { useMap } from 'react-leaflet';

export default function AlertPanel() {
  const { alerts, loading } = useAlerts();

  return (
    <div 
      className={`glass-panel ${loading ? 'loading-shimmer' : ''}`} 
      style={{ position: 'fixed', right: '16px', top: '72px', width: '260px', maxHeight: 'calc(100vh - 120px)', padding: '16px', zIndex: 1000, overflowY: 'auto' }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <div style={{ fontSize: '12px', fontWeight: 'bold' }}>⚡ EXTREME ALERTS</div>
        <div style={{ background: 'var(--red)', color: 'white', fontSize: '11px', fontWeight: 'bold', padding: '2px 8px', borderRadius: '12px' }}>
          {alerts?.length || 0}
        </div>
      </div>

      {!loading && (!alerts || alerts.length === 0) && (
        <div style={{ textAlign: 'center', marginTop: '30px', marginBottom: '20px' }}>
          <div style={{ color: 'var(--green)', fontSize: '13px', marginBottom: '4px' }}>All clear ✓</div>
          <div style={{ color: 'rgba(255,255,255,0.4)', fontSize: '11px' }}>No active warnings</div>
        </div>
      )}

      {alerts && alerts.map((alert, idx) => {
        const isRed = alert.severity === 2;
        const color = isRed ? 'var(--red)' : 'var(--amber)';
        const bg = isRed ? 'rgba(248,113,113,0.15)' : 'rgba(251,191,36,0.15)';
        const border = isRed ? 'rgba(248,113,113,0.3)' : 'rgba(251,191,36,0.3)';

        return (
          <div 
            key={idx}
            style={{
              background: 'rgba(255,255,255,0.03)',
              borderRadius: '8px',
              borderLeft: `3px solid ${color}`,
              padding: '10px 12px',
              marginBottom: '8px',
              cursor: 'pointer'
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
              <div style={{ fontSize: '11px', fontWeight: 'bold', textTransform: 'uppercase' }}>
                {alert.alert_type.replace('_', ' ')}
              </div>
              <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.4)', textAlign: 'right' }}>
                {alert.region}
              </div>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ 
                background: bg, color: color, border: `1px solid ${border}`,
                fontSize: '9px', fontWeight: 'bold', padding: '2px 6px', borderRadius: '12px'
              }}>
                {isRed ? 'RED' : 'AMBER'}
              </div>
              <div style={{ fontSize: '10px', color: 'rgba(255,255,255,0.4)' }}>
                {alert.valid_time}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
