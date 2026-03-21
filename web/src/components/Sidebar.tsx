"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Settings, Zap } from "lucide-react";
import { useAppRoutes, SystemHeartbeat } from "./CoreRegistry";

export function Sidebar() {
  const pathname = usePathname();
  const links = useAppRoutes();

  return (
    <nav className="w-64 border-r border-border bg-card-bg flex flex-col h-full flex-shrink-0">
      <div className="h-16 flex items-center px-6 border-b border-border">
        <Link href="/" className="flex items-center gap-2 group">
          <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center text-white shadow-labs group-hover:scale-105 transition-transform">
            <Zap size={18} fill="currentColor" />
          </div>
          <span className="font-semibold text-lg tracking-tight text-foreground">Sentinel AI</span>
        </Link>
      </div>

      <div className="flex-1 py-6 px-4 space-y-1 overflow-y-auto">
        {links.map((link) => {
          const isActive = pathname === link.href;
          return (
            <Link
              key={link.href}
              href={link.href}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                isActive
                  ? "bg-primary/10 text-primary"
                  : "text-muted-foreground hover:bg-muted hover:text-foreground"
              }`}
            >
              <link.icon size={18} strokeWidth={isActive ? 2.5 : 2} />
              {link.label}
            </Link>
          );
        })}
      </div>

      <div className="p-6 border-t border-border mt-auto">
        <button className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors w-full px-3 py-2 rounded-lg hover:bg-muted">
          <Settings size={18} />
          <span>Config</span>
        </button>
        <SystemHeartbeat />
      </div>
    </nav>
  );
}
