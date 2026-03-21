import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Sentinel AI | Home",
  description: "Enterprise Multimodal Analysis Platform",
};

import { Sidebar } from "@/components/Sidebar";

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="flex h-screen overflow-hidden antialiased text-foreground bg-background">
        <Sidebar />
        <main className="flex-1 flex flex-col h-full relative overflow-hidden bg-background">
          {children}
        </main>
      </body>
    </html>
  );
}
