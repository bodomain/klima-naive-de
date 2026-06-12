import axios from "axios";
import type {
  VariablesResponse,
  HyrasPointResponse,
  StationInfo,
  StationTimeseriesResponse,
  GermanyGeoJSON,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

const api = axios.create({
  baseURL: `${API_BASE}/api`,
  timeout: 30000,
});

export async function getVariables(): Promise<VariablesResponse> {
  const res = await api.get("/variables");
  return res.data;
}

export async function getHyrasPoint(
  lat: number,
  lon: number,
  variable: string,
  stat: string
): Promise<HyrasPointResponse> {
  const res = await api.get("/hyras/point", {
    params: { lat, lon, variable, stat },
  });
  return res.data;
}

export async function getNearestStation(
  lat: number,
  lon: number,
  variable?: string
): Promise<StationInfo> {
  const res = await api.get("/stations/nearest", {
    params: { lat, lon, variable },
  });
  return res.data;
}

export async function getStationTimeseries(
  stationId: string,
  variable: string
): Promise<StationTimeseriesResponse> {
  const res = await api.get(`/stations/${stationId}/timeseries`, {
    params: { variable },
  });
  return res.data;
}

export async function getGermanyBoundary(): Promise<GermanyGeoJSON> {
  const res = await api.get("/germany");
  return res.data;
}
