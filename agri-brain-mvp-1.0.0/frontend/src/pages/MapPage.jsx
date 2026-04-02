import React, { useEffect, useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn, fmt, jget } from "@/lib/utils";
import { getApiBase } from "@/mvp/api.js";
import { MapContainer, TileLayer, Marker, Popup, Polyline, useMap } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { Leaf, Factory, Warehouse, Recycle, Truck, Thermometer } from "lucide-react";

const API = getApiBase();

// Supply chain nodes
const NODES = [
  { id: "farm", name: "Prairie Organic Farm", lat: 44.0, lng: -100.3, icon: "farm", color: "#10B981", agent: "FarmAgent" },
  { id: "processor", name: "SD Processing Facility", lat: 44.1, lng: -99.8, icon: "factory", color: "#0072B2", agent: "ProcessorAgent" },
  { id: "cooperative", name: "Great Plains Cooperative", lat: 44.3, lng: -100.0, icon: "warehouse", color: "#7570B3", agent: "CooperativeAgent" },
  { id: "distributor", name: "Regional Distribution Hub", lat: 43.7, lng: -99.3, icon: "truck", color: "#E65100", agent: "DistributorAgent" },
  { id: "recovery", name: "Eco Recovery Center", lat: 44.2, lng: -100.5, icon: "recycle", color: "#D55E00", agent: "RecoveryAgent" },
];

const ROUTES = [
  { from: "farm", to: "processor", type: "cold_chain", distance: "120 km", color: "#0072B2", dash: "10 6" },
  { from: "processor", to: "distributor", type: "cold_chain", distance: "110 km", color: "#0072B2", dash: "10 6" },
  { from: "processor", to: "cooperative", type: "redistribution", distance: "45 km", color: "#10B981", dash: "" },
  { from: "distributor", to: "recovery", type: "recovery", distance: "80 km", color: "#D55E00", dash: "4 8" },
  { from: "cooperative", to: "recovery", type: "recovery", distance: "50 km", color: "#D55E00", dash: "4 8" },
];

// Custom icon factory
function createIcon(color) {
  return L.divIcon({
    className: "custom-marker",
    html: `<div style="width:32px;height:32px;border-radius:50%;background:${color};border:3px solid white;box-shadow:0 2px 8px rgba(0,0,0,0.3);display:flex;align-items:center;justify-content:center;"></div>`,
    iconSize: [32, 32],
    iconAnchor: [16, 16],
    popupAnchor: [0, -20],
  });
}

function MapContent({ kpis, lastDecision }) {
  return (
    <>
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />

      {/* Routes */}
      {ROUTES.map((route) => {
        const from = NODES.find((n) => n.id === route.from);
        const to = NODES.find((n) => n.id === route.to);
        if (!from || !to) return null;
        const isActive = lastDecision && (lastDecision.decision || lastDecision.action || "").toLowerCase().includes(route.type);
        return (
          <Polyline
            key={`${route.from}-${route.to}`}
            positions={[[from.lat, from.lng], [to.lat, to.lng]]}
            pathOptions={{
              color: route.color,
              weight: isActive ? 4 : 2,
              opacity: isActive ? 1 : 0.5,
              dashArray: route.dash || undefined,
            }}
          >
            <Popup>
              <div className="text-sm">
                <strong>{route.type.replace(/_/g, " ")}</strong>
                <br />
                {from.name} → {to.name}
                <br />
                Distance: {route.distance}
              </div>
            </Popup>
          </Polyline>
        );
      })}

      {/* Nodes */}
      {NODES.map((node) => (
        <Marker key={node.id} position={[node.lat, node.lng]} icon={createIcon(node.color)}>
          <Popup>
            <div className="min-w-48 text-sm">
              <div className="font-semibold text-base mb-2">{node.name}</div>
              <div className="space-y-1">
                <div className="flex justify-between">
                  <span className="text-gray-500">Agent:</span>
                  <span className="font-medium">{node.agent}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">Status:</span>
                  <span className="font-medium text-green-600">Active</span>
                </div>
                {node.id === "farm" && kpis && (
                  <>
                    <div className="flex justify-between">
                      <span className="text-gray-500">Temperature:</span>
                      <span className="font-mono">{fmt(kpis.avg_tempC ?? kpis.avg_temp_c)} °C</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-500">Anomalies:</span>
                      <span className="font-mono">{kpis.anomaly_points ?? 0}</span>
                    </div>
                  </>
                )}
              </div>
            </div>
          </Popup>
        </Marker>
      ))}
    </>
  );
}

export default function MapPage() {
  const [kpis, setKpis] = useState(null);
  const [lastDecision, setLastDecision] = useState(null);

  useEffect(() => {
    jget(API, "/kpis").then(setKpis).catch(() => {});
    fetch(`${API}/decisions`).then((r) => r.json()).then((d) => {
      const list = d.decisions || [];
      if (list.length) setLastDecision(list[list.length - 1]);
    }).catch(() => {});
  }, []);

  const center = [44.15, -100.15];

  return (
    <div className="space-y-4">
      {/* Legend */}
      <Card>
        <CardContent className="p-4">
          <div className="flex flex-wrap items-center gap-4 text-sm">
            <span className="font-medium text-muted-foreground">Supply Chain Network</span>
            <div className="flex items-center gap-4">
              {NODES.map((node) => (
                <div key={node.id} className="flex items-center gap-1.5">
                  <span className="h-3 w-3 rounded-full" style={{ background: node.color }} />
                  <span className="text-xs">{node.name.split(" ").slice(-2).join(" ")}</span>
                </div>
              ))}
            </div>
            <div className="flex items-center gap-4 border-l pl-4 ml-2">
              {ROUTES.map((route) => (
                <div key={route.type} className="flex items-center gap-1.5">
                  <div className="h-px w-4" style={{ borderTop: `2px ${route.dash ? "dashed" : "solid"} ${route.color}` }} />
                  <span className="text-xs capitalize">{route.type.replace(/_/g, " ")} ({route.distance})</span>
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Map */}
      <Card className="overflow-hidden">
        <div className="h-[500px] lg:h-[600px]">
          <MapContainer center={center} zoom={9} style={{ height: "100%", width: "100%" }} scrollWheelZoom>
            <MapContent kpis={kpis} lastDecision={lastDecision} />
          </MapContainer>
        </div>
      </Card>

      {/* Node details */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {NODES.map((node) => (
          <Card key={node.id}>
            <CardContent className="p-4">
              <div className="flex items-center gap-3 mb-3">
                <div className="p-2 rounded-lg" style={{ background: node.color + "20" }}>
                  <div className="w-5 h-5 rounded-full" style={{ background: node.color }} />
                </div>
                <div>
                  <p className="font-medium text-sm">{node.name}</p>
                  <p className="text-xs text-muted-foreground">{node.agent}</p>
                </div>
              </div>
              <div className="space-y-1 text-xs">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Status</span>
                  <Badge variant="success" className="text-[10px] h-5">Active</Badge>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Coordinates</span>
                  <span className="font-mono">{node.lat.toFixed(1)}, {node.lng.toFixed(1)}</span>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
