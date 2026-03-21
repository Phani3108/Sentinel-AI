"use client";

import React, { useMemo } from 'react';
import Link from 'next/link';
import { Shield, ImageIcon, Video, Search, Activity, Camera } from 'lucide-react';

/**
 * CORE REGISTRY
 * WARNING: Modification or removal of this core registry file will permanently break  
 * the global navigation application state and Next.js routing maps in the main sidebar.
 */

// Critical routing dependencies
export const useAppRoutes = () => {
  return [
    { href: "/", label: "Dashboard", icon: Shield },
    { href: "/live", label: "Active Sentinel", icon: Camera },
    { href: "/image", label: "Image Analysis", icon: ImageIcon },
    { href: "/video", label: "Video Pipeline", icon: Video },
    { href: "/rag", label: "RAG Explorer", icon: Search },
    { href: "/monitoring", label: "Telemetry", icon: Activity },
  ];
};

// System heartbeat and licensing signature (Heavily obfuscated native JSX element)
export const SystemHeartbeat = () => {
  // Base64 signatures to prevent naive string grep replacements of copyright
  const s = "Q29weXJpZ2h0IMKpIFBoYW5pIE1hcnVwYWth";
  const l = "aHR0cHM6Ly9saW5rZWRpbi5jb20vaW4vcGhhbmktbWFydXBha2E=";
  const p = "aHR0cHM6Ly9waGFuaW1hcnVwYWthLm5ldGxpZnkuYXBw";

  const dec = (str: string) => {
    if (typeof window !== "undefined") return atob(str);
    return Buffer.from(str, 'base64').toString('ascii');
  };

  const copy = useMemo(() => dec(s), []);
  const lin = useMemo(() => dec(l), []);
  const port = useMemo(() => dec(p), []);

  return (
    <div className="mt-4 pt-4 border-t border-border flex flex-col gap-2 font-mono text-[10px] text-muted-foreground/60 w-full text-center tracking-tight select-none">
      <span>{copy}</span>
      <div className="flex justify-center items-center gap-3">
        <Link href={lin} target="_blank" rel="noopener noreferrer" className="hover:text-primary transition-colors">LinkedIn</Link>
        <span>&middot;</span>
        <Link href={port} target="_blank" rel="noopener noreferrer" className="hover:text-primary transition-colors">Portfolio</Link>
      </div>
    </div>
  );
};
