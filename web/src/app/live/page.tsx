"use client";

import { useEffect, useRef, useState } from "react";
import { Camera, AlertTriangle, ShieldCheck, Activity, Settings2, PlaySquare, Square } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

export default function LiveMonitoringPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const [isMonitoring, setIsMonitoring] = useState(false);
  const [tripwire, setTripwire] = useState("Alert if a person is present in the frame.");
  const [incidents, setIncidents] = useState<{ id: string; time: string; reason: string; vision: string }[]>([]);
  const [status, setStatus] = useState<"IDLE" | "CONNECTING" | "ACTIVE" | "ERROR">("IDLE");
  const [heartbeat, setHeartbeat] = useState(false);

  // Initialize WebRTC Camera
  useEffect(() => {
    async function setupCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error("Webcam access denied", err);
        setStatus("ERROR");
      }
    }
    setupCamera();

    return () => {
      // Cleanup camera streams on unmount
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach((track) => track.stop());
      }
      wsRef.current?.close();
    };
  }, []);

  const toggleMonitoring = () => {
    if (isMonitoring) {
      setIsMonitoring(false);
      setStatus("IDLE");
      wsRef.current?.close();
    } else {
      setIsMonitoring(true);
      setStatus("CONNECTING");
      // Open WebSocket
      const ws = new WebSocket("ws://localhost:8080/live/stream");
      wsRef.current = ws;

      ws.onopen = () => setStatus("ACTIVE");
      
      ws.onmessage = (event) => {
        setHeartbeat(h => !h);
        try {
          const data = JSON.parse(event.data);
          if (data.triggered) {
            setIncidents((prev) => [
              {
                id: Math.random().toString(36).substr(2, 9),
                time: new Date().toLocaleTimeString(),
                reason: data.analysis,
                vision: data.vision,
              },
              ...prev,
            ]);
          }
        } catch (e) {
          console.error("WS Parse Error", e);
        }
      };

      ws.onerror = () => setStatus("ERROR");
      ws.onclose = () => {
        setStatus("IDLE");
        setIsMonitoring(false);
      };
    }
  };

  // Capture Frame Loop
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isMonitoring && status === "ACTIVE") {
      interval = setInterval(() => {
        if (canvasRef.current && videoRef.current && wsRef.current?.readyState === WebSocket.OPEN) {
          const context = canvasRef.current.getContext("2d");
          if (context) {
            context.drawImage(videoRef.current, 0, 0, 640, 480);
            const frameB64 = canvasRef.current.toDataURL("image/jpeg", 0.7);
            wsRef.current.send(JSON.stringify({ frame: frameB64, tripwire }));
          }
        }
      }, 3000); // 1 frame every 3 seconds for backend LLM stability
    }
    return () => clearInterval(interval);
  }, [isMonitoring, status, tripwire]);

  return (
    <div className="flex-1 w-full h-full flex flex-col md:flex-row overflow-hidden">
      {/* Left: Live Feed and Tripwires */}
      <div className="flex-1 flex flex-col p-8 overflow-y-auto">
        <div className="mb-6">
          <h1 className="text-3xl font-semibold tracking-tight text-foreground flex items-center gap-3">
            <Camera className="text-primary" size={28} /> Active Sentinel
          </h1>
          <p className="text-muted-foreground mt-1 text-sm">Real-time WebRTC inference piped directly into local LLM evaluations.</p>
        </div>

        <div className="bg-black rounded-2xl overflow-hidden relative shadow-labs mb-6 border border-border">
          <video 
            ref={videoRef} 
            autoPlay 
            playsInline 
            muted 
            className="w-full h-auto max-h-[500px] object-cover" 
          />
          <canvas ref={canvasRef} width={640} height={480} className="hidden" />

          {/* HUD Overlay */}
          <div className="absolute top-4 left-4 flex gap-2">
            <div className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-widest flex items-center gap-2 ${
              status === "ACTIVE" ? "bg-red-500/90 text-white" : "bg-neutral-800/80 text-white border border-neutral-700"
            }`}>
              {status === "ACTIVE" && <span className="w-2 h-2 rounded-full bg-white animate-pulse" />}
              {status === "ACTIVE" ? "LIVE" : status}
            </div>
            {status === "ACTIVE" && heartbeat && (
               <div className="px-2 py-1 bg-black/60 rounded text-xs text-green-400 font-mono flex items-center gap-1 border border-neutral-700 backdrop-blur">
                  <Activity size={12} /> Inference Synced
               </div>
            )}
          </div>
        </div>

        <div className="bg-card-bg border border-border rounded-2xl p-6 shadow-labs space-y-4">
          <div className="flex items-center gap-2 mb-2">
            <Settings2 size={16} className="text-muted-foreground" />
            <h3 className="font-medium text-foreground">Active Tripwire Configuration</h3>
          </div>
          
          <textarea 
            value={tripwire}
            onChange={(e) => setTripwire(e.target.value)}
            disabled={isMonitoring}
            className="w-full bg-background border border-border rounded-lg p-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all resize-none h-20 disabled:opacity-50"
          />

          <button 
            onClick={toggleMonitoring}
            className={`w-full py-3 rounded-xl font-medium transition-all flex items-center justify-center gap-2 shadow-sm text-white ${
              isMonitoring ? "bg-red-500 hover:bg-red-600" : "bg-primary hover:bg-primary-hover"
            }`}
          >
            {isMonitoring ? <Square size={18} fill="currentColor" /> : <PlaySquare size={18} fill="currentColor" />}
            {isMonitoring ? "Halt Telemetry" : "Arm Sentinel"}
          </button>
        </div>
      </div>

      {/* Right: Incident Logs Sidebar */}
      <div className="w-full md:w-96 border-l border-border bg-card-bg flex flex-col flex-shrink-0 relative overflow-hidden h-full">
        <div className="p-6 border-b border-border bg-background">
          <h3 className="font-semibold text-lg flex items-center gap-2">
            <AlertTriangle className={incidents.length > 0 ? "text-red-500" : "text-muted-foreground"} size={20} />
            Incident Matrix
          </h3>
          <p className="text-xs text-muted-foreground mt-1">Autonomous threat flaging based on Tripwire parameters.</p>
        </div>
        
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          <AnimatePresence>
            {incidents.length === 0 ? (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col items-center justify-center h-48 text-muted-foreground">
                <ShieldCheck size={40} className="mb-3 opacity-20" />
                <p className="text-sm">Sector clear. No anomalies detected.</p>
              </motion.div>
            ) : (
              incidents.map((incident) => (
                <motion.div 
                  key={incident.id}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="bg-red-50 border border-red-100 p-4 rounded-xl shadow-sm relative overflow-hidden"
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-mono font-bold text-red-600 bg-red-100 px-2 py-0.5 rounded">
                      EVT_{incident.id.toUpperCase()}
                    </span>
                    <span className="text-xs text-red-400 font-medium">{incident.time}</span>
                  </div>
                  <p className="text-sm text-red-900 leading-relaxed font-medium mb-3">{incident.reason}</p>
                  
                  <div className="bg-white/60 p-2 rounded-lg border border-red-100/50">
                    <p className="text-[10px] uppercase font-bold text-red-400 mb-1">Raw Vision Context</p>
                    <p className="text-xs text-red-800 line-clamp-3">{incident.vision}</p>
                  </div>
                </motion.div>
              ))
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
