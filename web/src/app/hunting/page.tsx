"use client";

import { useState } from "react";
import { Search, Map, Database, Bot, Clock, Crosshair, Terminal, Loader2 } from "lucide-react";
import { motion } from "framer-motion";

export default function ThreatHuntingPage() {
  const [query, setQuery] = useState("");
  const [analysis, setAnalysis] = useState("");
  const [isHunting, setIsHunting] = useState(false);

  const dispatchHunt = async () => {
    if (!query) return;
    setIsHunting(true);
    setAnalysis("");

    try {
      const response = await fetch("http://localhost:8080/hunting/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      const data = await response.json();
      setAnalysis(data.analysis);
    } catch (e) {
      console.error(e);
      setAnalysis("FATAL: Control Plane unreachable. Threat mapping aborted.");
    } finally {
      setIsHunting(false);
    }
  };

  return (
    <div className="flex-1 w-full h-full flex flex-col overflow-y-auto p-8 gap-8 max-w-5xl mx-auto">
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="text-3xl font-semibold tracking-tight text-foreground flex items-center gap-3">
          <Map className="text-primary" size={28} /> Palantir Threat Hunting
        </h1>
        <p className="text-muted-foreground mt-1 text-sm">
          Deploy an autonomous ReAct forensic analyst to synthesize macroscopic temporal trends across the ChromaDB Vector Lake and the Flywheel SQL Immutable Ledger.
        </p>
      </motion.div>

      <div className="bg-card-bg border border-border rounded-3xl p-6 shadow-labs space-y-4">
        <div className="relative flex-1">
          <Terminal className="absolute left-4 top-1/2 -translate-y-1/2 text-primary/70" size={20} />
          <input 
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && dispatchHunt()}
            placeholder="e.g., Map out every incident over the last week involving unauthorized vehicles near the North Gate."
            className="w-full bg-background border border-border rounded-xl py-4 pl-12 pr-4 font-mono text-sm focus:outline-none focus:ring-2 focus:ring-primary/20 transition-all font-medium text-foreground"
          />
        </div>
        
        <div className="flex justify-end">
          <button 
            onClick={dispatchHunt}
            disabled={!query || isHunting}
            className="bg-primary hover:bg-primary-hover disabled:bg-muted-foreground text-white px-8 py-3 rounded-xl font-medium transition-all flex items-center gap-2 shadow-sm"
          >
            {isHunting ? <Loader2 size={18} className="animate-spin" /> : <Crosshair size={18} />}
            {isHunting ? "Extracting Manifolds..." : "Deploy ReAct Agent"}
          </button>
        </div>
      </div>

      {(analysis || isHunting) && (
         <motion.div 
            initial={{ opacity: 0, scale: 0.98 }} 
            animate={{ opacity: 1, scale: 1 }}
            className="flex-1 bg-black/90 p-8 rounded-3xl border border-border shadow-[0_0_50px_rgba(0,0,0,0.5)] flex flex-col mt-4"
          >
            <div className="flex items-center justify-between mb-6 pb-4 border-b border-white/10">
               <div className="flex items-center gap-3 text-white/80 uppercase tracking-widest text-xs font-bold">
                 <Database size={16} className="text-primary" />
                 <span>Sentinel Forensic Synthesis</span>
               </div>
               <Clock size={16} className="text-white/30" />
            </div>

            {isHunting ? (
               <div className="flex-1 flex flex-col items-center justify-center min-h-[200px] text-white/50 space-y-4 font-mono">
                 <Bot size={48} className="animate-bounce text-primary" />
                 <p className="text-sm tracking-widest animate-pulse">ReAct Agent Interrogating SQL & Vector Clusters...</p>
               </div>
            ) : (
               <div className="flex-1 text-white/90 font-mono text-sm leading-relaxed whitespace-pre-wrap selection:bg-primary/30">
                 {analysis}
               </div>
            )}
         </motion.div>
      )}
    </div>
  );
}
