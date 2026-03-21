"use client";

import { useState, useRef, FormEvent } from "react";
import { UploadCloud, CheckCircle2, AlertCircle, Play, Settings2 } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

export default function ImageAnalysisPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [prompt, setPrompt] = useState("What do you observe in this image?");
  const [visionPrompt, setVisionPrompt] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [streamText, setStreamText] = useState("");
  const [isFinished, setIsFinished] = useState(false);
  const [latency, setLatency] = useState(0);

  // RLHF State
  const [feedbackLoading, setFeedbackLoading] = useState(false);
  const [feedbackSuccess, setFeedbackSuccess] = useState(false);
  const [correction, setCorrection] = useState("");

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selected = e.target.files[0];
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
      // Reset state
      setStreamText("");
      setIsFinished(false);
      setError(null);
      setFeedbackSuccess(false);
    }
  };

  const handleRun = async () => {
    if (!file) return;
    setLoading(true);
    setStreamText("");
    setIsFinished(false);
    setError(null);
    setLatency(0);
    setFeedbackSuccess(false);
    
    const startTime = Date.now();

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("prompt", prompt);
      if (visionPrompt.trim()) {
        formData.append("vision_prompt", visionPrompt.trim());
      }

      const response = await fetch("http://localhost:8080/analyze/image/stream", {
        method: "POST",
        body: formData,
        // Optional headers if required by backend, e.g., API keys
        headers: {
          "X-API-Key": "sk-admin-secret-key-123"
        }
      });

      if (!response.ok) {
        throw new Error(`API returned status ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (reader) {
        let isDone = false;
        while (!isDone) {
          const { value, done } = await reader.read();
          isDone = done;
          if (value) {
            const chunk = decoder.decode(value);
            const lines = chunk.split("\n");
            for (const line of lines) {
              if (line.startsWith("data: ")) {
                const token = line.slice(6);
                if (token === "[DONE]") {
                  isDone = true;
                  break;
                }
                setStreamText((prev) => prev + token);
              }
            }
          }
        }
      }
      setIsFinished(true);
      setLatency(Date.now() - startTime);
    } catch (err: any) {
      setError(err.message || "Failed to reach the Sentinel API.");
    } finally {
      setLoading(false);
    }
  };

  const submitFeedback = async (rating: number) => {
    setFeedbackLoading(true);
    try {
      const payload = {
        prompt,
        original_answer: streamText,
        rating,
        image_hash: "react_frontend_hash",
        user_correction: correction.trim() || undefined
      };
      
      const res = await fetch("http://localhost:8080/feedback/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-API-Key": "sk-admin-secret-key-123"
        },
        body: JSON.stringify(payload)
      });
      
      if (res.ok) {
        setFeedbackSuccess(true);
      } else {
        throw new Error("Feedback submission failed");
      }
    } catch (err) {
      console.error(err);
    } finally {
      setFeedbackLoading(false);
    }
  };

  return (
    <div className="flex-1 overflow-y-auto w-full p-8 md:p-12">
      <div className="max-w-6xl mx-auto space-y-8">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight text-foreground">Multimodal Analysis Sandbox</h1>
          <p className="text-muted-foreground mt-1 text-sm">Upload visual media to extract deterministic insights using the Hybrid Pipeline.</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* Left Column: Config & Upload */}
          <div className="lg:col-span-4 flex flex-col gap-6">
            <div 
              className="bg-card-bg border border-border rounded-2xl p-6 shadow-labs flex flex-col items-center justify-center text-center cursor-pointer hover:border-primary/50 transition-colors"
              onClick={() => fileInputRef.current?.click()}
            >
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileChange} 
                className="hidden" 
                accept="image/jpeg, image/png, image/webp"
              />
              {preview ? (
                <img src={preview} alt="Upload Preview" className="w-full h-48 object-cover rounded-xl" />
              ) : (
                <div className="py-8">
                  <div className="w-12 h-12 bg-primary/10 rounded-full flex items-center justify-center mx-auto mb-4">
                    <UploadCloud className="text-primary" size={24} />
                  </div>
                  <h3 className="font-medium text-foreground">Drag & Drop Image</h3>
                  <p className="text-sm text-muted-foreground mt-1">JPEG, PNG, WebP up to 10MB</p>
                </div>
              )}
            </div>

            <div className="bg-card-bg border border-border rounded-2xl p-6 shadow-labs space-y-4">
              <div className="flex items-center gap-2 mb-2">
                <Settings2 size={16} className="text-muted-foreground" />
                <h3 className="font-medium text-foreground">Parameters</h3>
              </div>
              
              <div className="space-y-1">
                <label className="text-xs font-medium tracking-wide text-muted-foreground uppercase">General Prompt</label>
                <textarea 
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  className="w-full bg-background border border-border rounded-lg p-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all resize-none h-24"
                />
              </div>

              <div className="space-y-1">
                <label className="text-xs font-medium tracking-wide text-muted-foreground uppercase">Vision Override (Optional)</label>
                <input 
                  type="text"
                  value={visionPrompt}
                  placeholder="Leave blank for auto-description"
                  onChange={(e) => setVisionPrompt(e.target.value)}
                  className="w-full bg-background border border-border rounded-lg p-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all"
                />
              </div>

              <button 
                onClick={handleRun}
                disabled={!file || loading}
                className="w-full mt-4 bg-primary hover:bg-primary-hover disabled:bg-muted-foreground disabled:cursor-not-allowed text-white font-medium py-3 rounded-xl transition-all flex items-center justify-center gap-2 shadow-sm"
              >
                {loading ? (
                  <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1, ease: "linear" }}>
                    <AlertCircle size={18} />
                  </motion.div>
                ) : (
                  <Play size={18} fill="currentColor" />
                )}
                {loading ? "Analyzing Pipeline..." : "Execute Reasoning"}
              </button>
            </div>
          </div>

          {/* Right Column: Execution Result & RLHF */}
          <div className="lg:col-span-8 flex flex-col gap-6">
            <div className="bg-card-bg border border-border rounded-2xl p-6 shadow-labs min-h-[400px] flex flex-col relative overflow-hidden">
              <h3 className="font-medium text-foreground mb-4 flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-primary animate-pulse" />
                Secure Stream Output
              </h3>

              {error ? (
                <div className="p-4 bg-red-50 text-red-600 rounded-xl border border-red-100 flex items-start gap-3">
                  <AlertCircle size={20} className="shrink-0 mt-0.5" />
                  <p className="text-sm leading-relaxed">{error}</p>
                </div>
              ) : !streamText && !loading ? (
                <div className="flex-1 flex flex-col items-center justify-center text-muted-foreground">
                  <Activity size={32} className="opacity-20 mb-4" />
                  <p>Awaiting execution payload...</p>
                </div>
              ) : (
                <div className="flex-1 overflow-y-auto">
                  <div className="prose prose-sm prose-slate max-w-none text-foreground leading-relaxed whitespace-pre-wrap font-mono text-sm">
                    {streamText}
                    {loading && <span className="inline-block w-2 h-4 bg-primary ml-1 animate-pulse" />}
                  </div>
                </div>
              )}

              {isFinished && (
                <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="mt-6 pt-4 border-t border-border flex items-center justify-between text-xs text-muted-foreground">
                  <span className="flex items-center gap-1.5 font-medium text-green-600">
                    <CheckCircle2 size={14} /> Pipeline Complete
                  </span>
                  <span>Latency: {(latency / 1000).toFixed(2)}s</span>
                </motion.div>
              )}
            </div>

            {/* Phase 9 Continuous Learning Flywheel UI */}
            <AnimatePresence>
              {isFinished && (
                <motion.div 
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  className="bg-card-bg border border-border rounded-2xl p-6 shadow-labs"
                >
                  <div className="flex items-center gap-2 mb-4">
                    <div className="bg-primary/10 text-primary p-1.5 rounded-md">
                      <Settings2 size={16} />
                    </div>
                    <div>
                      <h3 className="font-medium text-foreground">Continuous Learning Flywheel</h3>
                      <p className="text-xs text-muted-foreground">Your feedback corrects hallucinations and curates LoRA tuning datasets.</p>
                    </div>
                  </div>

                  {feedbackSuccess ? (
                    <div className="p-4 bg-green-50 text-green-700 rounded-xl border border-green-100 flex items-center justify-center gap-2 text-sm font-medium">
                      <CheckCircle2 size={18} /> Feedback successfully injected into the RLHF dataset!
                    </div>
                  ) : (
                    <div className="space-y-4">
                      <textarea 
                        placeholder="Provide the ground-truth correction here (Optional)"
                        value={correction}
                        onChange={(e) => setCorrection(e.target.value)}
                        className="w-full bg-background border border-border rounded-lg p-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all resize-none h-20"
                      />
                      <div className="flex items-center gap-3">
                        <button 
                          onClick={() => submitFeedback(1)}
                          disabled={feedbackLoading}
                          className="flex-1 py-2 rounded-xl bg-background border border-border hover:border-primary hover:text-primary transition-colors text-sm font-medium flex items-center justify-center gap-2 shadow-sm"
                        >
                          👍 Accurate
                        </button>
                        <button 
                          onClick={() => submitFeedback(-1)}
                          disabled={feedbackLoading}
                          className="flex-1 py-2 rounded-xl bg-background border border-border hover:border-red-500 hover:text-red-600 transition-colors text-sm font-medium flex items-center justify-center gap-2 shadow-sm"
                        >
                          👎 Hallucination
                        </button>
                      </div>
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>

          </div>
        </div>
      </div>
    </div>
  );
}
