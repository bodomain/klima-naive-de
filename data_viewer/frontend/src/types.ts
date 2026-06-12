export interface VariableInfo {
  label: string;
  unit: string;
  stats?: Record<string, { label: string; unit: string }>;
}

export interface VariablesResponse {
  hyras: Record<string, VariableInfo>;
  stations: Record<string, VariableInfo>;
}

export interface TimeseriesData {
  years: number[];
  values: (number | null)[];
}

export interface HyrasPointResponse {
  variable: string;
  stat: string;
  requested: { lat: number; lon: number };
  cell: { lat: number; lon: number };
  timeseries: TimeseriesData;
}

export interface StationInfo {
  id: string;
  name: string;
  lat: number | null;
  lon: number | null;
  elevation: number | null;
  start_year: number;
  end_year: number;
  has_temp: number;
  has_sunshine: number;
  has_precip: number;
  distance_km?: number;
}

export interface StationTimeseriesResponse {
  station_id: string;
  variable: string;
  timeseries: TimeseriesData;
}

export interface GermanyGeoJSON {
  type: "Feature";
  geometry: {
    type: "Polygon";
    coordinates: number[][][];
  };
  properties: Record<string, string>;
}
