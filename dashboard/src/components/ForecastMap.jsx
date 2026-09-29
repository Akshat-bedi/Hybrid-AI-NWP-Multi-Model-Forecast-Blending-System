import React, { useEffect, useRef } from 'react';
import { MapContainer, TileLayer, useMap } from 'react-leaflet';
import L from 'leaflet';
import { useForecastData, useAlerts } from '../hooks/useForecastData';
import WeightLayer from './WeightLayer';

// Constants for normalization
const VAR_CONFIG = {
  t2m: { min: 280, max: 318, gradient: { 0: '#1e40af', 0.3: '#3b82f6', 0.5: '#fde68a', 0.75: '#f97316', 1: '#dc2626' } },
  tp: { min: 0, max: 15, gradient: { 0: '#f0f9ff', 0.2: '#7dd3fc', 0.5: '#0ea5e9', 0.8: '#1d4ed8', 1: '#312e81' } },
  wind_speed: { min: 0, max: 25, gradient: { 0: '#ecfdf5', 0.3: '#6ee7b7', 0.6: '#059669', 1: '#064e3b' } },
  alert_level: { min: 0, max: 2, gradient: { 0: '#052e16', 0.5: '#92400e', 1: '#7f1d1d' } }
};

// Heatmap component
function HeatmapLayer({ variable, leadHours, onError }) {
  const map = useMap();
  const heatLayerRef = useRef(null);
  const { data, loading, error } = useForecastData(variable, leadHours);

  useEffect(() => {
    if (error) {
      onError(error);
    }
  }, [error, onError]);

  useEffect(() => {
    if (window._heatLayer && (!window.L || !window.L.heatLayer)) {
      if (!window.L) window.L = L;
      window.L.heatLayer = window._heatLayer;
      window.L.HeatLayer = window._HeatLayerClass;
    }
    console.log("Heatmap effect triggered! " + JSON.stringify({ map: !!map, data: !!data, heatData: data?.heatData?.length, hasL: !!window.L, hasHeat: !!window.L?.heatLayer }));
    if (!map || !data || !data.heatData || !window.L || !window.L.heatLayer) return;

    const config = VAR_CONFIG[variable] || VAR_CONFIG['t2m'];
    
    if (heatLayerRef.current) {
      map.removeLayer(heatLayerRef.current);
    }

    console.log("Adding heatlayer to map with data points:", data.heatData.length);
    heatLayerRef.current = window.L.heatLayer(data.heatData, {
      radius: 22,
      blur: 18,
      maxZoom: 8,
      gradient: config.gradient
    }).addTo(map);

    return () => {
      if (heatLayerRef.current && map) {
        map.removeLayer(heatLayerRef.current);
      }
    };
  }, [map, data, variable]);

  return null;
}

// Custom Alert Markers
function AlertMarkers() {
  const map = useMap();
  const { alerts } = useAlerts();

  useEffect(() => {
    if (!map || !alerts) return;

    const markers = [];

    alerts.forEach(alert => {
      const iconHtml = `
        <div class="alert-marker severity-${alert.severity}">
          <div class="ring"></div>
          <div class="dot"></div>
        </div>
      `;
      
      const icon = L.divIcon({
        html: iconHtml,
        className: '', // reset default leaflet class
        iconSize: [20, 20],
        iconAnchor: [10, 10]
      });

      const marker = L.marker([alert.lat_center, alert.lon_center], { icon }).addTo(map);
      
      // Popup
      marker.bindPopup(`
        <div style="font-family: system-ui; padding: 4px;">
          <strong style="color: ${alert.severity === 2 ? '#f87171' : '#fbbf24'};">${alert.alert_type.replace('_', ' ').toUpperCase()}</strong><br/>
          <span style="color: #666; font-size: 11px;">${alert.region}</span><br/>
          <span style="color: #333; font-size: 10px;">Valid: ${alert.valid_time}</span>
        </div>
      `);

      marker.on('click', () => {
        map.flyTo([alert.lat_center, alert.lon_center], 7, { duration: 1.5 });
      });

      markers.push(marker);
    });

    return () => {
      markers.forEach(m => map.removeLayer(m));
    };
  }, [map, alerts]);

  return null;
}

// Custom Glass Zoom Control
function GlassZoomControl() {
  const map = useMap();
  
  return (
    <div style={{ position: 'absolute', bottom: '60px', right: '20px', zIndex: 1000, display: 'flex', flexDirection: 'column', gap: '8px' }}>
      <button 
        className="glass-panel" 
        style={{ width: '40px', height: '40px', color: 'white', border: '1px solid rgba(255,255,255,0.2)', fontSize: '20px', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
        onClick={() => map.zoomIn()}
      >
        +
      </button>
      <button 
        className="glass-panel" 
        style={{ width: '40px', height: '40px', color: 'white', border: '1px solid rgba(255,255,255,0.2)', fontSize: '20px', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
        onClick={() => map.zoomOut()}
      >
        −
      </button>
    </div>
  );
}

// Main Component
export default function ForecastMap({ activeTab, variable, leadHours, onError }) {
  return (
    <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', zIndex: 0, opacity: activeTab === 'SKILL' ? 0.3 : 1, transition: 'opacity 0.3s' }}>
      <MapContainer 
        center={[20.5, 78.9]} 
        zoom={5} 
        minZoom={4} 
        maxZoom={8} 
        zoomControl={false} 
        style={{ width: '100%', height: '100%', background: '#050a14' }}
      >
        <TileLayer
          attribution='&copy; Stadia Maps'
          url='https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png'
        />
        
        {activeTab !== 'WEIGHTS' && (
          <HeatmapLayer variable={variable} leadHours={leadHours} onError={onError} />
        )}
        
        {activeTab === 'WEIGHTS' && (
          <WeightLayer variable={variable} leadHours={leadHours} />
        )}
        
        <AlertMarkers />
        <GlassZoomControl />
      </MapContainer>
    </div>
  );
}
