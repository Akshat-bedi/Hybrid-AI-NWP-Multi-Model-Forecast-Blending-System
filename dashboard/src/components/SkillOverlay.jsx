import React, { useEffect, useState } from 'react';
import { useSkill } from '../hooks/useForecastData';

const MODEL_COLORS = {
  gfs: '#3b82f6',
  ecmwf: '#eab308',
  pangu: '#a855f7',
  graphcast: '#ef4444',
  hybrid_blend: '#4ade80'
};

export default function SkillOverlay({ variable, leadHours }) {
  const { skill, loading } = useSkill(variable, leadHours);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    // Small delay to trigger CSS animation on mount
    setTimeout(() => setMounted(true), 100);
    return () => setMounted(false);
  }, [skill]);

  // Aggregate mean RMSE across all regions per model
  const aggScores = {};
  if (skill) {
    skill.forEach(row => {
      if (!aggScores[row.model]) {
        aggScores[row.model] = { sum: 0, count: 0 };
      }
      aggScores[row.model].sum += row.rmse;
      aggScores[row.model].count += 1;
    });
  }

  const chartData = Object.entries(aggScores).map(([model, stats]) => ({
    model,
    rmse: stats.sum / stats.count,
    isBlend: model === 'hybrid_blend'
  })).sort((a, b) => {
    // Sort logic to put blend at bottom, others alphabetical
    if (a.isBlend) return 1;
    if (b.isBlend) return -1;
    return a.model.localeCompare(b.model);
  });

  const maxRmse = Math.max(...chartData.map(d => d.rmse), 0.001);

  let improvement = 0;
  if (chartData.length > 1) {
    const blendScore = chartData.find(d => d.isBlend)?.rmse;
    const others = chartData.filter(d => !d.isBlend).map(d => d.rmse);
    if (blendScore !== undefined && others.length > 0) {
      const bestOther = Math.min(...others);
      improvement = ((bestOther - blendScore) / bestOther) * 100;
      improvement = Math.max(0, improvement); // Ensure no negative improvement text natively
    }
  }

  return (
    <div 
      className={`glass-panel ${loading ? 'loading-shimmer' : ''}`}
      style={{ 
        position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)',
        width: '600px', padding: '24px', zIndex: 1000
      }}
    >
      <div style={{ textAlign: 'center', marginBottom: '24px' }}>
        <h2 style={{ 
          margin: '0 0 4px 0', fontSize: '16px', fontWeight: 'bold', letterSpacing: '1px',
          background: 'linear-gradient(90deg, #38bdf8, #a78bfa)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent'
        }}>
          MODEL SKILL COMPARISON
        </h2>
        <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)', textTransform: 'uppercase' }}>
          {variable.replace('_', ' ')} · LEAD {leadHours}H
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        {chartData.map((d, idx) => {
          // Invert: lower RMSE is better -> longer bar. (1 - normalizedRMSE)
          // Actually, RMSE min is best.
          // Let's normalize so best (min) is 100%, worst (max) is 20%.
          const normalized = maxRmse === 0 ? 1 : d.rmse / maxRmse;
          const barWidth = mounted ? `${(1 - normalized * 0.8) * 100}%` : '0%';
          const color = d.isBlend ? 'linear-gradient(90deg, #4ade80, #38bdf8)' : (MODEL_COLORS[d.model] || '#666');

          return (
            <div key={d.model} style={{ display: 'flex', alignItems: 'center' }}>
              <div style={{ width: '120px', fontSize: '12px', display: 'flex', alignItems: 'center', gap: '8px', color: d.isBlend ? '#4ade80' : 'white', fontWeight: d.isBlend ? 'bold' : 'normal' }}>
                {!d.isBlend && <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: color }}></div>}
                {d.isBlend ? 'BLEND ★' : d.model.toUpperCase()}
              </div>
              <div style={{ flex: 1, height: '12px', background: 'rgba(255,255,255,0.05)', borderRadius: '6px', overflow: 'hidden' }}>
                <div style={{ 
                  height: '100%', width: barWidth, background: color, borderRadius: '6px',
                  transition: 'width 1s cubic-bezier(0.4, 0, 0.2, 1)'
                }}></div>
              </div>
              <div className="mono" style={{ width: '60px', textAlign: 'right', fontSize: '12px', color: 'rgba(255,255,255,0.8)' }}>
                {d.rmse.toFixed(3)}
              </div>
            </div>
          );
        })}
      </div>

      {!loading && improvement > 0 && (
        <div style={{ 
          marginTop: '24px', background: 'rgba(74,222,128,0.1)', border: '1px solid rgba(74,222,128,0.3)',
          padding: '12px', borderRadius: '8px', textAlign: 'center', fontSize: '12px', color: '#4ade80'
        }}>
          Blended forecast reduces RMSE by <strong>~{improvement.toFixed(1)}%</strong> vs best individual model
        </div>
      )}
    </div>
  );
}
