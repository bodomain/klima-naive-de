import { useEffect, useState } from "react";
import type { StationInfo } from "../types";
import { getNearestStation } from "../api";

interface LocationPanelProps {
  point: { lat: number; lon: number } | null;
}

export default function LocationPanel({ point }: LocationPanelProps) {
  const [station, setStation] = useState<StationInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>("");

  useEffect(() => {
    if (!point) {
      setStation(null);
      return;
    }
    setLoading(true);
    setError("");
    getNearestStation(point.lat, point.lon)
      .then((s) => {
        setStation(s);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.response?.data?.detail || "Failed to load station");
        setLoading(false);
      });
  }, [point]);

  if (!point) {
    return (
      <div className="panel">
        <h2>Location</h2>
        <p className="hint">Click on the map to select a point.</p>
      </div>
    );
  }

  return (
    <div className="panel">
      <h2>Location</h2>
      <div className="coords">
        <span>Lat: {point.lat.toFixed(4)}</span>
        <span>Lon: {point.lon.toFixed(4)}</span>
      </div>

      <h3>Nearest DWD Station</h3>
      {loading && <p className="hint">Loading station...</p>}
      {error && <p className="error">{error}</p>}
      {station && (
        <div className="station-info">
          <p>
            <strong>{station.name || station.id}</strong>
            {station.distance_km !== undefined && (
              <span className="distance"> ({station.distance_km} km)</span>
            )}
          </p>
          <p className="meta">
            ID: {station.id} | Years: {station.start_year}–{station.end_year}
          </p>
          <p className="meta">
            Available: {station.has_temp ? "Temp" : ""}
            {station.has_sunshine ? ", Sunshine" : ""}
            {station.has_precip ? ", Precip" : ""}
          </p>
        </div>
      )}
    </div>
  );
}
