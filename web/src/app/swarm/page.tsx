"use client";

import { useState, useRef, useEffect } from "react";
import { Users, Bot, ShieldAlert, Cpu, Network, UploadCloud, Loader2, AlertTriangle, Lock, Key } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

type SwarmEvent = {
  id: string;
  node: string;
  message: string;
};

export default function SwarmArenaPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [tripwire, setTripwire] = useState("Scan the visual manifold. Is there a critical anomaly that violates internal policies?");
  
  const [isSwarming, setIsSwarming] = useState(false);
  const [webhookFired, setWebhookFired] = useState(false);
  
  // Phase 22: Containment Interlocks
  const [lockedIncident, setLockedIncident] = useState<string | null>(null);
  
  const [events, setEvents] = useState<SwarmEvent[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [events]);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const f = e.target.files[0];
      setFile(f);
      setPreview(URL.createObjectURL(f));
    }
  };

  const deploySwarm = async () => {
    if (!file) return;
    setIsSwarming(true);
    setWebhookFired(false);
    setLockedIncident(null);
    setEvents([]);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("tripwire", tripwire);

    try {
      const response = await fetch("http://localhost:8080/swarm/stream", {
        method: "POST",
        body: formData,
      });

      if (!response.body) throw new Error("No response body.");

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        
        let boundary = buffer.indexOf("\n\n");
        while (boundary !== -1) {
          const chunk = buffer.slice(0, boundary).replace(/^data:\s*/, "");
          buffer = buffer.slice(boundary + 2);

          try {
            const data = JSON.parse(chunk);
            if (data.message === "[DONE]") {
              setIsSwarming(false);
              break;
            }
            
            if (data.node === "action_agent" && data.message.includes("CRITICAL payload to Enterprise Webhook Hub")) {
              setWebhookFired(true);
            }
            
            // Phase 22: Regex explicitly extracting the Python dict `incident_id` if present
            if (data.message.includes("[CONTAINMENT_LOCK]")) {
               const match = data.message.match(/\[(INC-[A-Z0-9]{6})\]/);
               if (match) setLockedIncident(match[1]);
            }
            
            setEvents((prev) => [
              ...prev, 
              { id: Math.random().toString(), node: data.node, message: data.message }
            ]);
            
          } catch (e) {
            console.error("Failed to parse SSE chunk", chunk);
          }

          boundary = buffer.indexOf("\n\n");
        }
      }
    } catch (e) {
      console.error(e);
      setEvents((prev) => [...prev, { id: "err", node: "system", message: "Swarm encountered a fatal error." }]);
      setIsSwarming(false);
    }
  };

  const getNodeColor = (node: string) => {
    if (node === "director") return "bg-blue-500 text-blue-50 border-blue-600";
    if (node === "vision_agent") return "bg-purple-500 text-purple-50 border-purple-600";
    if (node === "intel_agent") return "bg-orange-500 text-orange-50 border-orange-600";
    if (node === "action_agent") return "bg-red-500 text-red-50 border-red-600 shadow-[0_0_15px_rgba(239,68,68,0.5)]";
    return "bg-neutral-800 text-white border-neutral-700";
  };
  
  const getNodeIcon = (node: string) => {
    if (node === "director") return <Network size={16} />;
    if (node === "vision_agent") return <Cpu size={16} />;
    if (node === "intel_agent") return <ShieldAlert size={16} />;
    if (node === "action_agent") return <AlertTriangle size={16} className="animate-pulse" />;
    return <Bot size={16} />;
  };
  
  const approveContainment = async () => {
     if (!lockedIncident) return;
     try {
       const res = await fetch("http://localhost:8080/containment/approve", {
         method: "POST",
         headers: { "Content-Type": "application/json" },
         body: JSON.stringify({ incident_id: lockedIncident })
       });
       const data = await res.json();
       if (data.status === "success") {
          setLockedIncident(null);
          setWebhookFired(true); // Physically executes post-unlock
          setEvents(prev => [...prev, { id: Math.random().toString(), node: "action_agent", message: `[ACTION_AGENT]: Admin Key Accepted. ${data.message}` }]);
       }
     } catch (e) {
       console.error("Lock override failed", e);
     }
  };

  return (
    <div className="flex-1 w-full h-full flex flex-col overflow-hidden p-8 gap-6 max-w-7xl mx-auto">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight text-foreground flex items-center gap-3">
          <Users className="text-primary" size={28} /> Multi-Agent Swarm Arena
        </h1>
        <p className="text-muted-foreground mt-1 text-sm">Deploy hierarchical LLM agents operating via Cyclical LangGraph mechanics to rigorously interrogate an incident.</p>
      </div>

      <div className="flex flex-col md:flex-row gap-6 flex-1 min-h-0">
        {/* Left: Configuration & Preview */}
        <div className="w-full md:w-1/3 flex flex-col gap-6">
           <div className="bg-card-bg border border-border rounded-2xl p-6 shadow-labs">
            <h3 className="font-medium text-foreground mb-4">Threat Intel Ingestion</h3>
            <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-dashed border-border rounded-xl cursor-pointer hover:bg-muted/50 transition-colors mb-4">
              <div className="flex flex-col items-center justify-center pt-5 pb-6">
                <UploadCloud className="text-muted-foreground mb-2" size={24} />
                <p className="text-sm text-muted-foreground">Drop incident frame</p>
              </div>
              <input type="file" className="hidden" accept="image/*" onChange={handleFileUpload} />
            </label>

            {preview && (
              <div className="mb-4 rounded-xl overflow-hidden border border-border">
                <img src={preview} alt="Preview" className="w-full h-40 object-cover" />
              </div>
            )}

            <div className="mb-4">
              <label className="block text-xs uppercase tracking-wider font-bold text-muted-foreground mb-2">Director's Directive (Tripwire)</label>
              <textarea 
                value={tripwire}
                onChange={(e) => setTripwire(e.target.value)}
                className="w-full bg-background border border-border rounded-lg p-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary/20 h-24 resize-none"
              />
            </div>

            <button 
              onClick={deploySwarm}
              disabled={!file || isSwarming}
              className="w-full py-3 rounded-xl font-medium text-white transition-all flex items-center justify-center gap-2 shadow-sm bg-primary hover:bg-primary-hover disabled:bg-muted-foreground"
            >
              {isSwarming ? <Loader2 size={18} className="animate-spin" /> : <Network size={18} />}
              {isSwarming ? "Swarm Executing..." : "Deploy Swarm"}
            </button>
           </div>
        </div>

        {/* Right: The Arena (Agent Chat) */}
        <div className="w-full md:w-2/3 bg-background border border-border rounded-2xl shadow-labs flex flex-col overflow-hidden relative">
          
          <AnimatePresence>
            {webhookFired && (
              <motion.div 
                initial={{ opacity: 0, y: -50 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.95 }}
                className="absolute top-4 left-4 right-4 z-50 bg-red-500/90 text-white p-4 rounded-xl shadow-[0_0_30px_rgba(239,68,68,0.4)] border border-red-400 flex items-center gap-4 backdrop-blur-md"
              >
                <AlertTriangle size={32} className="animate-pulse flex-shrink-0" />
                <div>
                   <h4 className="font-bold text-lg leading-tight uppercase tracking-widest">Autonomous Remediation Triggered</h4>
                   <p className="text-red-100 text-sm">Action Agent has physically dispatched the incident payload via Enterprise Webhook Hub.</p>
                </div>
              </motion.div>
            )}
            
            {/* Phase 22: Containment Approval Core */}
            {lockedIncident && (
              <motion.div 
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="absolute inset-0 z-50 bg-black/80 backdrop-blur-md flex items-center justify-center p-6"
              >
                <div className="bg-[#1C1C1E] border border-orange-500/50 rounded-2xl p-8 max-w-md w-full shadow-[0_0_50px_rgba(249,115,22,0.2)] flex flex-col items-center text-center">
                   <div className="w-16 h-16 rounded-full bg-orange-500/10 flex items-center justify-center mb-6 border border-orange-500/20">
                     <Lock className="text-orange-500" size={32} />
                   </div>
                   <h3 className="text-xl font-bold text-white mb-2">Security Clearance Required</h3>
                   <p className="text-muted-foreground text-sm mb-6">
                     The Swarm evaluated the matrix as CRITICAL. Your Operator role restricts autonomous action dispatch. 
                     <br/><br/>
                     <span className="font-mono text-orange-400 bg-orange-400/10 px-2 py-1 rounded">ID: {lockedIncident}</span>
                   </p>
                   
                   <div className="flex w-full gap-4">
                      <button onClick={() => setLockedIncident(null)} className="flex-1 px-4 py-3 rounded-xl border border-border text-white hover:bg-white/5 transition-all">Abort</button>
                      <button onClick={approveContainment} className="flex-1 px-4 py-3 rounded-xl bg-orange-500 hover:bg-orange-600 text-white font-bold flex items-center justify-center gap-2 transition-all">
                        <Key size={16} /> Admin Override
                      </button>
                   </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          <div className="border-b border-border bg-card-bg p-4 flex items-center justify-between z-10">
            <h3 className="font-semibold text-foreground flex items-center gap-2">
              <Network className="text-primary" size={18} /> LangGraph Execution Trace
            </h3>
            {isSwarming && (
              <div className="flex items-center gap-2 text-xs font-mono text-purple-500 bg-purple-500/10 px-3 py-1 rounded-full border border-purple-200">
                <span className="w-2 h-2 rounded-full bg-purple-500 animate-pulse" />
                Processing Graph State
              </div>
            )}
          </div>

          <div 
            ref={scrollRef} 
            className="flex-1 overflow-y-auto p-6 space-y-4 font-mono scroll-smooth bg-[url('/grid.svg')] bg-center"
          >
             {events.length === 0 && !isSwarming && (
               <div className="flex flex-col items-center justify-center h-full text-muted-foreground opacity-30">
                 <Bot size={64} className="mb-4" />
                 <p className="text-sm">Agents awaiting deployment constraints.</p>
               </div>
             )}

             <AnimatePresence>
                {events.map((evt) => (
                  <motion.div 
                    key={evt.id}
                    initial={{ opacity: 0, scale: 0.95, y: 10 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    className={`p-4 rounded-xl border max-w-[85%] ${getNodeColor(evt.node)} shadow-sm ${evt.node === 'director' ? 'ml-auto text-right' : 'mr-auto text-left'}`}
                  >
                    <div className={`flex items-center gap-2 mb-1 opacity-70 ${evt.node === 'director' ? 'justify-end' : 'justify-start'}`}>
                      {getNodeIcon(evt.node)}
                      <span className="text-[10px] uppercase font-bold tracking-widest">{evt.node}</span>
                    </div>
                    <p className="text-sm leading-relaxed whitespace-pre-wrap">
                      {evt.message.replace(/\[.*?\]:\s/, '')}
                    </p>
                  </motion.div>
                ))}
             </AnimatePresence>
          </div>
        </div>
      </div>
    </div>
  );
}
