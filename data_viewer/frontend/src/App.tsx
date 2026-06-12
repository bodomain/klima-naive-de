import { useEffect, useState } from "react";
import Map from "./components/Map";
import LocationPanel from "./components/LocationPanel";
import VariableSelector from "./components/VariableSelector";
import TimeseriesChart from "./components/TimeseriesChart";
import { getVariables, getHyrasPoint, getNearestStation, getStationTimeseries } from "./api";
import type { VariablesResponse, TimeseriesData } from "./types";
import "./App.css";

function App() {
  const [selectedPoint, setSelectedPoint] = useState<{ lat: number; lon: number } | null>(null);
  const [variables, setVariables] = useState<VariablesResponse | null>(null);
  const [chartData, setChartData] = useState<TimeseriesData | null>(null);
  const [chartLabel, setChartLabel] = useState("");
  const [chartUnit, setChartUnit] = useState("");
  const [chartLoading, setChartLoading] = useState(false);
  const [chartError, setChartError] = useState("");

  useEffect(() => {
    getVariables()
      .then(setVariables)
      .catch((err) => console.error("Failed to load variables:", err));
  }, []);

  const handleSelectPoint = (lat: number, lon: number) => {
    setSelectedPoint({ lat, lon });
    setChartData(null);
    setChartError("");
  };

  const handlePlot = async (source: "hyras" | "station", variable: string, stat?: string) => {
    if (!selectedPoint) return;
    setChartLoading(true);
    setChartError("");
    setChartData(null);

    try {
      if (source === "hyras") {
        const res = await getHyrasPoint(selectedPoint.lat, selectedPoint.lon, variable, stat!);
        setChartData(res.timeseries);
        const meta = variables?.hyras[variable];
        const statMeta = meta?.stats?.[stat!];
        setChartLabel(`${meta?.label || variable} — ${statMeta?.label || stat}`);
        setChartUnit(statMeta?.unit || meta?.unit || "");
      } else {
        const station = await getNearestStation(selectedPoint.lat, selectedPoint.lon, variable);
        const res = await getStationTimeseries(station.id, variable);
        setChartData(res.timeseries);
        const meta = variables?.stations[variable];
        setChartLabel(`${meta?.label || variable} (Station ${station.name || station.id})`);
        setChartUnit(meta?.unit || "");
      }
    } catch (err: any) {
      setChartError(err.response?.data?.detail || "Failed to load data");
    } finally {
      setChartLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>HYRAS Data Viewer</h1>
        <p className="subtitle">
          Explore historical German climate data from DWD HYRAS gridded dataset and DWD station observations.
        </p>
      </header>

      <main className="app-main">
        <div className="map-section">
          <Map onSelectPoint={handleSelectPoint} selectedPoint={selectedPoint} />
        </div>

        <div className="sidebar">
          <LocationPanel point={selectedPoint} />
          <VariableSelector variables={variables} onPlot={handlePlot} />
          {chartError && <div className="error-banner">{chartError}</div>}
          <TimeseriesChart
            years={chartData?.years || []}
            values={chartData?.values || []}
            label={chartLabel}
            unit={chartUnit}
            loading={chartLoading}
          />
        </div>
      </main>
    </div>
  );
}

export default App;
