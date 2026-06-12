import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

interface TimeseriesChartProps {
  years: number[];
  values: (number | null)[];
  label: string;
  unit: string;
  loading?: boolean;
}

export default function TimeseriesChart({ years, values, label, unit, loading }: TimeseriesChartProps) {
  const data = years.map((year, i) => ({
    year,
    value: values[i] !== null && values[i] !== undefined ? Number(values[i]) : null,
  }));

  if (loading) {
    return (
      <div className="chart-panel">
        <h3>Loading chart...</h3>
        <div className="skeleton-chart" />
      </div>
    );
  }

  if (data.length === 0) {
    return (
      <div className="chart-panel">
        <h3>Timeseries</h3>
        <p className="hint">Select a variable and click Plot to view the chart.</p>
      </div>
    );
  }

  return (
    <div className="chart-panel">
      <h3>
        {label} <small>({unit})</small>
      </h3>
      <div className="chart-container">
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={data} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              dataKey="year"
              type="number"
              domain={["dataMin", "dataMax"]}
              tick={{ fontSize: 12 }}
              label={{ value: "Year", position: "insideBottom", offset: -5 }}
            />
            <YAxis
              tick={{ fontSize: 12 }}
              label={{ value: unit, angle: -90, position: "insideLeft" }}
            />
            <Tooltip
              formatter={(value: any) => [Number(value).toFixed(2), label]}
              labelFormatter={(label: any) => `Year: ${label}`}
            />
            <Line
              type="monotone"
              dataKey="value"
              stroke="#2563eb"
              strokeWidth={2}
              dot={false}
              connectNulls
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
