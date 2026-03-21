"use client";

import { useState, useEffect } from "react";
import { Activity, Cpu, Server, Database, TrendingUp, AlertCircle } from "lucide-react";
import { motion } from "framer-motion";

export default function MonitoringPage() {
  const [sysHealth, setSysHealth] = useState<any>(null);
  const [ragStats, setRagStats] = useState<any>(null);

  useEffect(() => {
    fetch("http://localhost:8080/health").then(r => r.json()).then(setSysHealth).catch(console.error);
    fetch("http://localhost:8080/rag/stats").then(r => r.json()).then(setRagStats).catch(console.error);
  }, []);

  const metrics = [
    { label: "Pipeline Latency (P99)", value: "Connected", trend: "Live", icon: Activity, color: "text-primary", bg: "bg-primary/10" },
    { label: "Active Inference Model", value: sysHealth?.vision_model || "Loading...", trend: "VRAM Locked", icon: Cpu, color: "text-blue-500", bg: "bg-blue-500/10" },
    { label: "Active Generative LLM", value: sysHealth?.llm_model || "Loading...", trend: "Mapped", icon: TrendingUp, color: "text-green-500", bg: "bg-green-500/10" },
    { label: "Hardware Accel", value: sysHealth?.device?.toUpperCase() || "CPU", trend: "Operational", icon: Server, color: "text-purple-500", bg: "bg-purple-500/10" },
    { label: "ChromaDB Nodes", value: ragStats ? `${ragStats.document_count} docs` : "0", trend: "Indexed", icon: Database, color: "text-orange-500", bg: "bg-orange-500/10" },
  ];

  return (
    <div className="flex-1 overflow-y-auto w-full p-8 md:p-12">
      <div className="max-w-6xl mx-auto space-y-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-semibold tracking-tight text-foreground flex items-center gap-3">
              <Activity className="text-primary" size={28} /> System Telemetry
            </h1>
            <p className="text-muted-foreground mt-1 text-sm">Real-time infrastructure health and Prometheus metrics aggregation.</p>
          </div>
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-green-50 text-green-700 border border-green-200 text-sm font-medium">
            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            All Systems Operational
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4">
          {metrics.map((m, i) => (
            <motion.div 
              key={m.label}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.1 }}
              className="bg-card-bg border border-border rounded-2xl p-5 shadow-labs"
            >
              <div className="flex items-start justify-between mb-4">
                <div className={`w-10 h-10 rounded-xl ${m.bg} ${m.color} flex items-center justify-center`}>
                  <m.icon size={20} />
                </div>
              </div>
              <div>
                <h4 className="text-2xl font-semibold text-foreground tracking-tight">{m.value}</h4>
                <div className="flex items-center justify-between mt-1">
                  <p className="text-xs text-muted-foreground font-medium">{m.label}</p>
                  <span className="text-[10px] font-mono text-muted-foreground">{m.trend}</span>
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-card-bg border border-border rounded-2xl p-6 shadow-labs min-h-[300px] flex flex-col">
            <h3 className="font-medium text-foreground mb-4">Inference Latency Window</h3>
            <div className="flex-1 border border-border border-dashed rounded-xl bg-muted/50 flex items-center justify-center">
              <p className="text-muted-foreground text-sm flex items-center gap-2">
                <AlertCircle size={16} /> Grafana Embeds Disabled in Development
              </p>
            </div>
          </div>
          
          <div className="bg-card-bg border border-border rounded-2xl p-6 shadow-labs min-h-[300px] flex flex-col">
            <h3 className="font-medium text-foreground mb-4">Node Resource Utilization</h3>
            <div className="flex-1 border border-border border-dashed rounded-xl bg-muted/50 flex items-center justify-center">
              <p className="text-muted-foreground text-sm flex items-center gap-2">
                <AlertCircle size={16} /> Grafana Embeds Disabled in Development
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
