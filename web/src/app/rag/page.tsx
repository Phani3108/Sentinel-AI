"use client";

import { useState } from "react";
import { Search, Database, Fingerprint, Loader2 } from "lucide-react";
import { motion } from "framer-motion";

export default function RAGPage() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  // Executing raw semantic search against local ChromaDB embedding space
  const handleSearch = async () => {
    if (!query) return;
    setLoading(true);
    
    try {
      const response = await fetch("http://localhost:8080/rag/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query, top_k: 5 }),
      });
      const data = await response.json();
      if (data.results) {
        setResults(data.results.map((r: any) => ({
          id: r.id || Math.random().toString(),
          score: r.score || 0.0,
          content: r.page_content,
          metadata: r.metadata || { source: "Unknown Document", page: "N/A" }
        })));
      } else {
        setResults([]);
      }
    } catch (e) {
      console.error("RAG Query Failed", e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex-1 overflow-y-auto w-full p-8 md:p-12">
      <div className="max-w-4xl mx-auto space-y-8">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight text-foreground flex items-center gap-3">
            <Search className="text-primary" size={28} /> RAG Knowledge Explorer
          </h1>
          <p className="text-muted-foreground mt-1 text-sm">Direct retrieval interface for the embedded ChromaDB collections.</p>
        </div>

        <div className="bg-card-bg border border-border rounded-2xl p-6 shadow-labs flex gap-3">
          <div className="relative flex-1">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-muted-foreground" size={20} />
            <input 
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              placeholder="Search embedded documents via semantic similarity..."
              className="w-full bg-background border border-border rounded-xl py-3 pl-12 pr-4 text-sm focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all"
            />
          </div>
          <button 
            onClick={handleSearch}
            disabled={!query || loading}
            className="bg-primary hover:bg-primary-hover disabled:bg-muted-foreground text-white px-6 py-3 rounded-xl font-medium transition-colors flex items-center gap-2 shadow-sm"
          >
            {loading ? <Loader2 size={18} className="animate-spin" /> : <Database size={18} />}
            Retrieve
          </button>
        </div>

        {results.length > 0 && (
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-4">
            <h3 className="font-medium text-foreground flex items-center gap-2 px-2">
              <Fingerprint size={16} className="text-muted-foreground" /> Semantic Matches ({results.length})
            </h3>
            
            <div className="grid gap-4">
              {results.map((res, i) => (
                <div key={res.id} className="bg-card-bg border border-border rounded-2xl p-6 shadow-labs hover:shadow-labs-hover transition-all">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-xs font-mono font-medium text-primary bg-primary/10 px-2 py-1 rounded">
                      Dist: {res.score.toFixed(3)}
                    </span>
                    <span className="text-xs text-muted-foreground flex items-center gap-1.5">
                      {res.metadata.source} · pg {res.metadata.page}
                    </span>
                  </div>
                  <p className="text-sm text-foreground leading-relaxed">{res.content}</p>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
}
