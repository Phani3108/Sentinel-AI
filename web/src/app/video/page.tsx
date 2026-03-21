"use client";

import { useState, useRef, useEffect } from "react";
import { UploadCloud, CheckCircle2, AlertCircle, Play, Video as VideoIcon, Loader2, ListTree } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

export default function VideoAnalysisPage() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [prompt, setPrompt] = useState("Narrate the events happening in this video sequence.");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<"IDLE" | "SUBMITTING" | "PENDING" | "PROCESSING" | "SUCCESS" | "FAILURE">("IDLE");
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selected = e.target.files[0];
      setFile(selected);
      setPreviewUrl(URL.createObjectURL(selected));
      setStatus("IDLE");
      setJobId(null);
      setResult(null);
      setError(null);
    }
  };

  const handleRun = async () => {
    if (!file) return;
    setStatus("SUBMITTING");
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("prompt", prompt);

      const res = await fetch("http://localhost:8080/jobs/video", {
        method: "POST",
        body: formData,
        headers: { "X-API-Key": "sk-admin-secret-key-123" }
      });

      if (!res.ok) throw new Error("Failed to dispatch Celery Job");
      const data = await res.json();
      setJobId(data.job_id);
      setStatus("PENDING");
    } catch (err: any) {
      setError(err.message);
      setStatus("FAILURE");
    }
  };

  // Poll for job status
  useEffect(() => {
    if (!jobId || status === "SUCCESS" || status === "FAILURE") return;

    const interval = setInterval(async () => {
      try {
        const res = await fetch(`http://localhost:8080/jobs/${jobId}`, {
          headers: { "X-API-Key": "sk-admin-secret-key-123" }
        });
        if (res.ok) {
          const data = await res.json();
          if (data.status === "SUCCESS") {
            setResult(data.result);
            setStatus("SUCCESS");
          } else if (data.status === "FAILURE") {
            setError(data.result || "Celery worker failed.");
            setStatus("FAILURE");
          } else {
            setStatus("PROCESSING");
          }
        }
      } catch (e) {
        console.error("Polling error", e);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [jobId, status]);

  return (
    <div className="flex-1 overflow-y-auto w-full p-8 md:p-12">
      <div className="max-w-6xl mx-auto space-y-8">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight text-foreground flex items-center gap-3">
            <VideoIcon className="text-primary" size={28} /> Video Pipeline
          </h1>
          <p className="text-muted-foreground mt-1 text-sm">Asynchronous temporal analysis dispatched to detached Celery workers.</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
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
                accept="video/mp4, video/quicktime"
              />
              {previewUrl ? (
                <video src={previewUrl} controls className="w-full h-48 object-cover rounded-xl bg-black" />
              ) : (
                <div className="py-8">
                  <div className="w-12 h-12 bg-primary/10 rounded-full flex items-center justify-center mx-auto mb-4">
                    <UploadCloud className="text-primary" size={24} />
                  </div>
                  <h3 className="font-medium text-foreground">Upload Video</h3>
                  <p className="text-sm text-muted-foreground mt-1">MP4 or MOV</p>
                </div>
              )}
            </div>

            <div className="bg-card-bg border border-border rounded-2xl p-6 shadow-labs space-y-4">
              <div className="space-y-1">
                <label className="text-xs font-medium tracking-wide text-muted-foreground uppercase">Narration Prompt</label>
                <textarea 
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  className="w-full bg-background border border-border rounded-lg p-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all resize-none h-24"
                />
              </div>

              <button 
                onClick={handleRun}
                disabled={!file || status === "SUBMITTING" || status === "PENDING" || status === "PROCESSING"}
                className="w-full mt-4 bg-primary hover:bg-primary-hover disabled:bg-muted-foreground disabled:cursor-not-allowed text-white font-medium py-3 rounded-xl transition-all flex items-center justify-center gap-2 shadow-sm"
              >
                {status === "SUBMITTING" ? <Loader2 className="animate-spin" size={18} /> : <Play size={18} fill="currentColor" />}
                {status === "SUBMITTING" ? "Dispatching..." : "Enqueue Job"}
              </button>
            </div>
          </div>

          <div className="lg:col-span-8 flex flex-col gap-6">
            <div className="bg-card-bg border border-border rounded-2xl p-6 shadow-labs min-h-[400px] flex flex-col">
              <div className="flex items-center justify-between mb-6">
                <h3 className="font-medium text-foreground flex items-center gap-2">
                  <ListTree size={18} className="text-muted-foreground" /> Workflow Events
                </h3>
                {jobId && <span className="text-xs font-mono text-muted-foreground bg-muted px-2 py-1 rounded">Job ID: {jobId}</span>}
              </div>

              <div className="flex-1 flex flex-col">
                {status === "IDLE" && (
                  <div className="flex-1 flex flex-col items-center justify-center text-muted-foreground">
                    <p>Ready for dispatch.</p>
                  </div>
                )}
                
                {error && (
                  <div className="p-4 bg-red-50 text-red-600 rounded-xl border border-red-100 flex items-start gap-3">
                    <AlertCircle size={20} className="shrink-0 mt-0.5" />
                    <p className="text-sm leading-relaxed">{error}</p>
                  </div>
                )}

                {(status === "PENDING" || status === "PROCESSING") && (
                  <div className="flex-1 flex flex-col items-center justify-center">
                    <Loader2 className="w-8 h-8 text-primary animate-spin mb-4" />
                    <p className="text-foreground font-medium animate-pulse">
                      {status === "PENDING" ? "Job Queued in Redis..." : "Celery Worker Processing Frames..."}
                    </p>
                    <p className="text-muted-foreground text-sm mt-2 max-w-sm text-center">
                      Video analysis is extremely GPU intensive. The worker is extracting frames and inferencing visual descriptions temporally. 
                    </p>
                  </div>
                )}

                {status === "SUCCESS" && result && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
                    <div className="p-5 bg-background border border-border rounded-xl">
                      <h4 className="text-xs font-semibold uppercase tracking-wider text-primary mb-3">Temporal Descriptions</h4>
                      <ul className="space-y-3">
                        {result.frame_descriptions?.map((desc: any, idx: number) => (
                          <li key={idx} className="flex gap-4 items-start text-sm">
                            <span className="font-mono text-muted-foreground shrink-0 mt-0.5">Frame {desc.frame_index}</span>
                            <span className="text-foreground">{desc.description}</span>
                          </li>
                        ))}
                      </ul>
                    </div>

                    <div className="p-5 bg-primary/5 border border-primary/20 rounded-xl">
                      <h4 className="text-xs font-semibold uppercase tracking-wider text-primary mb-2">Final Narration</h4>
                      <p className="text-foreground leading-relaxed text-sm">{result.final_narration}</p>
                    </div>
                  </motion.div>
                )}
              </div>

            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
