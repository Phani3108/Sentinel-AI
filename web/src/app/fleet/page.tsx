"use client";

import { useState, useEffect } from "react";
import { Activity, Globe, Cpu, Radio, Shield, MapPin, Zap } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

type EdgeNode = {
  id: string;
  status: "ONLINE" | "OFFLINE";
  connected_at: string;
  last_heartbeat: string;
  location: { lat: number; lng: number };
  region: string;
};

export default function FleetPage() {
  const [fleet, setFleet] = useState<EdgeNode[]>([]);
  const [totalOnline, setTotalOnline] = useState(0);

  useEffect(() => {
    const fetchFleet = async () => {
      try {
        const res = await fetch("http://localhost:8080/fleet/status");
        const data = await res.json();
        setFleet(data.nodes || []);
        setTotalOnline(data.total_active || 0);
      } catch (e) {
        console.error("Fleet telemetry offline", e);
      }
    };
    
    fetchFleet();
    const iv = setInterval(fetchFleet, 2000);
    return () => clearInterval(iv);
  }, []);

  return (
    <div className="flex-1 w-full h-full flex flex-col p-8 gap-8 max-w-7xl mx-auto overflow-y-auto">
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight text-foreground flex items-center gap-3">
            <Globe className="text-primary" size={28} /> Global Fleet Command
          </h1>
          <p className="text-muted-foreground mt-1 text-sm">
            Macroscopic telemetry aggregator tracking physical Sentinel Edge Nodes spanning multi-region IoT meshes.
          </p>
        </div>
        
        <div className="flex items-center gap-4">
          <div className="bg-card-bg border border-border px-6 py-2 rounded-xl flex items-center gap-3 shadow-sm">
             <div className="relative flex items-center justify-center">
                <span className="absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-20 animate-ping"></span>
                <Radio className="text-emerald-500 relative" size={18} />
             </div>
             <div>
                <p className="text-[10px] text-muted-foreground uppercase tracking-widest font-bold">Active Sensors</p>
                <p className="text-lg font-mono font-bold leading-none">{totalOnline}</p>
             </div>
          </div>
        </div>
      </motion.div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <AnimatePresence>
          {fleet.map((node) => (
            <motion.div
              key={node.id}
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className={`relative overflow-hidden rounded-2xl border ${node.status === "ONLINE" ? 'bg-card-bg border-border shadow-labs' : 'bg-background border-border opacity-60'} p-6 transition-all duration-300`}
            >
              {/* Node Header */}
              <div className="flex justify-between items-start mb-6 border-b border-border pb-4">
                <div className="flex items-center gap-3">
                  <div className={`p-2 rounded-lg ${node.status === "ONLINE" ? 'bg-emerald-500/10 text-emerald-500' : 'bg-neutral-800 text-neutral-500'}`}>
                    <Cpu size={24} />
                  </div>
                  <div>
                    <h3 className="font-bold text-lg font-mono">{node.id}</h3>
                    <p className="text-xs text-muted-foreground tracking-widest uppercase">{node.region}</p>
                  </div>
                </div>
                {node.status === "ONLINE" ? (
                  <span className="bg-emerald-500/10 text-emerald-500 text-[10px] font-bold px-2 py-1 rounded-full border border-emerald-500/20 uppercase tracking-widest flex items-center gap-1">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span> Streaming
                  </span>
                ) : (
                  <span className="bg-neutral-800 text-neutral-400 text-[10px] font-bold px-2 py-1 rounded-full border border-neutral-700 uppercase tracking-widest">
                    Disconnected
                  </span>
                )}
              </div>

              {/* Node Telemetry */}
              <div className="space-y-4">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground flex items-center gap-2"><MapPin size={14} /> Coordinates</span>
                  <span className="font-mono text-foreground">{node.location.lat.toFixed(4)}, {node.location.lng.toFixed(4)}</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground flex items-center gap-2"><Activity size={14} /> Swarm Status</span>
                  <span className={node.status === "ONLINE" ? "text-primary font-medium" : "text-muted-foreground"}>
                    {node.status === "ONLINE" ? "Bound to Control Plane" : "Subspace Severed"}
                  </span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground flex items-center gap-2"><Zap size={14} /> Last Heartbeat</span>
                  <span className="font-mono text-xs text-muted-foreground">{node.last_heartbeat.split('T')[1]?.substring(0,8) || "N/A"}</span>
                </div>
              </div>

              {/* Visual Flare */}
              {node.status === "ONLINE" && (
                <div className="absolute top-0 right-0 w-32 h-32 bg-primary/5 rounded-full blur-3xl -mr-16 -mt-16 pointer-events-none"></div>
              )}
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
      
      {fleet.length === 0 && (
         <div className="flex-1 flex flex-col items-center justify-center border border-dashed border-border rounded-3xl bg-background/50">
            <Globe className="text-muted-foreground mb-4 opacity-50" size={48} />
            <p className="text-muted-foreground font-medium">No Edge Nodes detected in the Global Mesh.</p>
            <p className="text-xs text-muted-foreground mt-2">To provision a node, execute `edge_node.py` on remote hardware.</p>
         </div>
      )}
    </div>
  );
}
