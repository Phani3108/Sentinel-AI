"use client";

import Link from "next/link";
import { Shield, Image as ImageIcon, Video, Search, Activity, Cpu } from "lucide-react";
import { motion } from "framer-motion";

export default function Home() {
  const tools = [
    {
      name: "Image Analysis",
      description: "Extract deterministic semantic descriptions and reason over visual input.",
      icon: ImageIcon,
      href: "/image",
    },
    {
      name: "Video Pipeline",
      description: "Analyze temporal sequences, extract frames, and generate continuous narration.",
      icon: Video,
      href: "/video",
    },
    {
      name: "RAG Explorer",
      description: "Semantic search across your local embedded knowledge graph.",
      icon: Search,
      href: "/rag",
    },
    {
      name: "System Telemetry",
      description: "Monitor end-to-end latencies and background queue vitals.",
      icon: Activity,
      href: "/monitoring",
    },
  ];

  return (
    <div className="flex-1 overflow-y-auto w-full p-8 md:p-16">
      <div className="max-w-5xl mx-auto space-y-12">
        <motion.div 
          initial={{ opacity: 0, y: 10 }} 
          animate={{ opacity: 1, y: 0 }}
          className="space-y-4"
        >
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 text-primary text-sm font-semibold">
            <Cpu size={16} /> Sentinel AI v10.0
          </div>
          <h1 className="text-4xl md:text-5xl font-semibold tracking-tight text-foreground">
            Enterprise Multimodal Engine
          </h1>
          <p className="text-lg text-muted-foreground max-w-2xl leading-relaxed">
            A state-of-the-art deployment interface for visual reasoning, temporal video 
            understanding, and continuous RLHF learning. Build with deterministic bounds.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {tools.map((tool, idx) => (
            <motion.div
              key={tool.name}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 * idx }}
            >
              <Link href={tool.href}>
                <div className="group h-full p-6 bg-card-bg border border-border rounded-2xl shadow-labs hover:shadow-labs-hover transition-all duration-300">
                  <div className="w-12 h-12 rounded-full bg-muted flex items-center justify-center mb-4 group-hover:bg-primary/10 group-hover:text-primary transition-colors">
                    <tool.icon size={24} className="text-muted-foreground group-hover:text-primary transition-colors" />
                  </div>
                  <h3 className="text-xl font-medium mb-2">{tool.name}</h3>
                  <p className="text-muted-foreground leading-relaxed">{tool.description}</p>
                </div>
              </Link>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
}
