import { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE = 'http://localhost:8000/api/v1';

export function useForecastData(variable, leadHours) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let isMounted = true;
    
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const response = await axios.get(`${API_BASE}/forecast/${variable}/${leadHours}`);
        if (isMounted) {
          // Move the transformation logic here as requested
          const rawData = response.data;
          
          // Define config for normalization
          const VAR_CONFIG = {
            t2m: { min: 280, max: 318 },
            tp: { min: 0, max: 15 },
            wind_speed: { min: 0, max: 25 },
            alert_level: { min: 0, max: 2 }
          };
          const config = VAR_CONFIG[variable] || VAR_CONFIG['t2m'];
          
          const heatData = rawData.features.map(f => {
            const [lon, lat] = f.geometry.coordinates;
            const val = f.properties.value;
            let intensity = (val - config.min) / (config.max - config.min);
            intensity = Math.max(0, Math.min(1, intensity));
            return [lat, lon, intensity];
          });
          
          setData({ ...rawData, heatData });
          setLoading(false);
        }
      } catch (err) {
        if (isMounted) {
          console.error("Forecast fetch error:", err);
          setError(err);
          setLoading(false);
        }
      }
    };

    fetchData();

    return () => {
      isMounted = false;
    };
  }, [variable, leadHours]);

  return { data, loading, error };
}

export function useAlerts() {
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchAlerts = async () => {
      try {
        const response = await axios.get(`${API_BASE}/alerts/extreme`);
        setAlerts(response.data);
      } catch (err) {
        console.error("Alerts fetch error:", err);
        setAlerts([]);
      } finally {
        setLoading(false);
      }
    };
    fetchAlerts();
  }, []);

  return { alerts, loading };
}

export function useHealth() {
  const [health, setHealth] = useState(null);

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        // Health is at the root in our API main.py
        const response = await axios.get(`${API_BASE}/health`);
        setHealth(response.data);
      } catch (err) {
        setHealth({ status: "offline", models_loaded: [] });
      }
    };
    fetchHealth();
    
    // Poll health every 30s
    const interval = setInterval(fetchHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  return { health };
}

export function useWeights(variable, leadHours) {
  const [weights, setWeights] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchWeights = async () => {
      setLoading(true);
      try {
        // Map derived variables to their underlying NWP variables for weights
        let queryVar = variable;
        if (variable === 'wind_speed') queryVar = 'u10';
        if (variable === 'alert_level') queryVar = 'tp';
        
        const response = await axios.get(`${API_BASE}/weights/${queryVar}?lead_hours=${leadHours}&season=JJA`);
        setWeights(response.data);
      } catch (err) {
        console.error("Weights fetch error:", err);
        setWeights(null);
      } finally {
        setLoading(false);
      }
    };
    fetchWeights();
  }, [variable, leadHours]);

  return { weights, loading };
}

export function useSkill(variable, leadHours) {
  const [skill, setSkill] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchSkill = async () => {
      setLoading(true);
      try {
        // Map derived variables to their underlying NWP variables for skill
        let queryVar = variable;
        if (variable === 'wind_speed') queryVar = 'u10';
        if (variable === 'alert_level') queryVar = 'tp';

        const response = await axios.get(`${API_BASE}/skill/scores?variable=${queryVar}&lead_hours=${leadHours}`);
        setSkill(response.data);
      } catch (err) {
        console.error("Skill fetch error:", err);
        setSkill([]);
      } finally {
        setLoading(false);
      }
    };
    fetchSkill();
  }, [variable, leadHours]);

  return { skill, loading };
}
