import { useState } from "react";
import type { VariablesResponse } from "../types";

interface VariableSelectorProps {
  variables: VariablesResponse | null;
  onPlot: (source: "hyras" | "station", variable: string, stat?: string) => void;
}

export default function VariableSelector({ variables, onPlot }: VariableSelectorProps) {
  const [source, setSource] = useState<"hyras" | "station">("hyras");
  const [selectedVar, setSelectedVar] = useState<string>("");
  const [selectedStat, setSelectedStat] = useState<string>("");

  const handlePlot = () => {
    if (!selectedVar) return;
    if (source === "hyras" && !selectedStat) return;
    onPlot(source, selectedVar, selectedStat || undefined);
  };

  const varMeta = variables
    ? source === "hyras"
      ? variables.hyras
      : variables.stations
    : {};

  const stats =
    source === "hyras" && selectedVar && variables?.hyras[selectedVar]
      ? variables.hyras[selectedVar].stats
      : null;

  return (
    <div className="panel variable-selector">
      <h2>Choose Data</h2>

      <div className="source-toggle">
        <button
          className={source === "hyras" ? "active" : ""}
          onClick={() => {
            setSource("hyras");
            setSelectedVar("");
            setSelectedStat("");
          }}
        >
          HYRAS Gridded
        </button>
        <button
          className={source === "station" ? "active" : ""}
          onClick={() => {
            setSource("station");
            setSelectedVar("");
            setSelectedStat("");
          }}
        >
          DWD Station
        </button>
      </div>

      <div className="var-list">
        <h4>{source === "hyras" ? "HYRAS Variable" : "Station Variable"}</h4>
        {Object.entries(varMeta).map(([key, meta]) => (
          <label key={key} className="radio-label">
            <input
              type="radio"
              name="variable"
              value={key}
              checked={selectedVar === key}
              onChange={(e) => {
                setSelectedVar(e.target.value);
                setSelectedStat("");
              }}
            />
            <span>
              {meta.label} <small>({meta.unit})</small>
            </span>
          </label>
        ))}
      </div>

      {stats && (
        <div className="stat-list">
          <h4>Statistic</h4>
          {Object.entries(stats).map(([key, meta]) => (
            <label key={key} className="radio-label">
              <input
                type="radio"
                name="stat"
                value={key}
                checked={selectedStat === key}
                onChange={(e) => setSelectedStat(e.target.value)}
              />
              <span>
                {meta.label} <small>({meta.unit})</small>
              </span>
            </label>
          ))}
        </div>
      )}

      <button className="plot-btn" onClick={handlePlot} disabled={!selectedVar || (source === "hyras" && !selectedStat)}>
        Plot Timeseries
      </button>
    </div>
  );
}
