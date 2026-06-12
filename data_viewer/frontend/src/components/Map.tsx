import { useEffect, useState } from "react";
import { MapContainer, TileLayer, GeoJSON, Marker, useMapEvents } from "react-leaflet";
import L from "leaflet";
import type { GermanyGeoJSON } from "../types";
import { getGermanyBoundary } from "../api";

interface MapProps {
  onSelectPoint: (lat: number, lon: number) => void;
  selectedPoint: { lat: number; lon: number } | null;
}

function ClickHandler({ onSelect }: { onSelect: (lat: number, lon: number) => void }) {
  useMapEvents({
    click(e) {
      onSelect(e.latlng.lat, e.latlng.lng);
    },
  });
  return null;
}

const pinIcon = new L.Icon({
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
});

export default function Map({ onSelectPoint, selectedPoint }: MapProps) {
  const [boundary, setBoundary] = useState<GermanyGeoJSON | null>(null);

  useEffect(() => {
    getGermanyBoundary()
      .then(setBoundary)
      .catch((err) => console.error("Failed to load boundary:", err));
  }, []);

  return (
    <MapContainer
      center={[51.1657, 10.4515]}
      zoom={6}
      style={{ height: "100%", width: "100%", borderRadius: "8px" }}
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      {boundary && (
        <GeoJSON
          data={boundary}
          style={{
            color: "#2563eb",
            weight: 2,
            fillOpacity: 0.05,
          }}
        />
      )}
      <ClickHandler onSelect={onSelectPoint} />
      {selectedPoint && (
        <Marker position={[selectedPoint.lat, selectedPoint.lon]} icon={pinIcon} />
      )}
    </MapContainer>
  );
}
