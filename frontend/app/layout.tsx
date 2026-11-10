import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AetherMed 4.0 — Clinical Digital Twin & Precision TDM Platform",
  description:
    "Production-grade Clinical Decision Support and In-Silico Clinical Trial Optimization for Narrow Therapeutic Index (NTI) drugs.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-[#030712] text-slate-100 antialiased selection:bg-cyan-500 selection:text-slate-900">
        {children}
      </body>
    </html>
  );
}
