import React, { useEffect } from 'react';
import { useMap } from 'react-leaflet';
import L from 'leaflet';
import { useWeights } from '../hooks/useForecastData';

// Hardcoded rough bounds for Indian subregions just for visualization
// since the API returns weight per region string, but leaflet needs rects.
const REGION_BOUNDS = {
  north_west: [[23.0, 68.0], [38.0, 78.0]],
  north_east: [[21.0, 88.0], [29.0, 97.0]],
  central: [[20.0, 78.0], [28.0, 88.0]],
  peninsular: [[8.0, 72.0], [20.0, 85.0]],
  all_india: [[6.0, 68.0], [38.25, 97.25]] // Fallback
};

export default function WeightLayer({ variable, leadHours }) {
  const map = useMap();
  const { weights } = useWeights(variable, leadHours);

  useEffect(() => {
    if (!map || !weights || !weights.regions) return;

    const layers = [];

    Object.entries(weights.regions).forEach(([regionName, models]) => {
      // Find dominant model
      if (!models || models.length === 0) return;
      
      const dominant = [...models].sort((a, b) => b.weight - a.weight)[0];
      const bounds = REGION_BOUNDS[regionName] || REGION_BOUNDS['all_india'];
      
      const rect = L.rectangle(bounds, {
        color: 'white',
        weight: 1,
        opacity: 0.3,
        fillColor: dominant.color,
        fillOpacity: 0.35
      }).addTo(map);

      // Construct tooltip content for the hover popup
      let tooltipHtml = `<div style="font-family:system-ui; width: 140px; background: rgba(5,10,20,0.9); padding: 8px; border-radius: 4px; color: white;">`;
      tooltipHtml += `<strong style="font-size: 12px; border-bottom: 1px solid rgba(255,255,255,0.2); padding-bottom: 4px; margin-bottom: 6px; display: block;">${regionName.toUpperCase()}</strong>`;
      
      models.forEach(m => {
        const pct = (m.weight * 100).toFixed(1);
        tooltipHtml += `
          <div style="display: flex; align-items: center; justify-content: space-between; font-size: 10px; margin-bottom: 4px;">
            <span>${m.model_name}</span>
            <span class="mono">${pct}%</span>
          </div>
          <div style="width: 100%; height: 4px; background: rgba(255,255,255,0.1); border-radius: 2px; margin-bottom: 6px;">
            <div style="width: ${pct}%; height: 100%; background: ${m.color}; border-radius: 2px;"></div>
          </div>
        `;
      });
      tooltipHtml += `</div>`;

      rect.bindTooltip(tooltipHtml, { sticky: true, className: 'custom-tooltip' });
      layers.push(rect);
    });

    return () => {
      layers.forEach(l => map.removeLayer(l));
    };
  }, [map, weights]);

  return null;
}
