"use client";

import React, { useState } from "react";
import { BookOpen, ShieldCheck, ExternalLink, Bookmark, CheckCircle2, FileText } from "lucide-react";
import { RAGResponse, CitationItem } from "@/types";

interface RAGConsultViewProps {
  rag: RAGResponse | null;
  drug: string;
}

export const RAGConsultView: React.FC<RAGConsultViewProps> = ({ rag, drug }) => {
  const [activeCitation, setActiveCitation] = useState<number | null>(null);

  if (!rag) {
    return (
      <div className="glass-panel rounded-2xl p-8 text-center text-slate-400">
        Run simulation to generate literature-grounded clinical consult reports.
      </div>
    );
  }

  // Format markdown sections cleanly into visual cards
  const sections = rag.report_markdown.split(/(?=# Section \d+:|# Executive Clinical Summary|## |### )/g);

  return (
    <div className="space-y-6">
      {/* Top Banner: Grounding Audit & Verdict */}
      <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center space-x-3">
            <div className="p-2.5 rounded-xl bg-teal-950/80 border border-teal-700/60 text-teal-400">
              <BookOpen className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="text-xs text-slate-400 uppercase tracking-wider font-semibold">
                  Evidence-Grounded Clinical Decision Support
                </span>
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-950 border border-emerald-700 text-emerald-300 font-mono">
                  FDA Label &amp; DailyMed Grounded
                </span>
              </div>
              <h3 className="text-sm font-bold text-slate-100">
                AetherMed Clinical Consult &amp; Citation Audit ({drug})
              </h3>
            </div>
          </div>

          {/* Grounding Coverage Meter */}
          <div className="glass-panel px-4 py-2.5 rounded-xl border border-teal-500/40 bg-teal-950/30">
            <div className="flex items-center justify-between text-xs space-x-3">
              <span className="text-slate-400 flex items-center space-x-1">
                <ShieldCheck className="w-3.5 h-3.5 text-teal-400" />
                <span>Lexical Grounding</span>
              </span>
              <span className="font-mono font-bold text-teal-300 text-sm">
                {rag.grounding_coverage_pct.toFixed(1)}%
              </span>
            </div>
            <div className="w-44 h-1.5 rounded-full bg-slate-900 border border-slate-800 mt-1.5 overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-teal-500 to-emerald-400 rounded-full"
                style={{ width: `${rag.grounding_coverage_pct}%` }}
              />
            </div>
          </div>
        </div>

        {/* Executive Clinical Verdict Callout */}
        <div className="p-3.5 rounded-xl bg-slate-900/90 border border-slate-800 flex items-start space-x-3">
          <CheckCircle2 className="w-5 h-5 text-emerald-400 mt-0.5 flex-shrink-0" />
          <div className="space-y-0.5">
            <span className="text-xs font-semibold text-emerald-300 uppercase tracking-wider">
              Executive Dosing Verdict
            </span>
            <p className="text-xs text-slate-200 leading-relaxed">{rag.executive_verdict}</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Columns: Full 5-Section Consultation Report */}
        <div className="lg:col-span-2 space-y-4">
          <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-5">
            <div className="flex items-center space-x-2 text-xs text-slate-400 border-b border-slate-800/80 pb-3">
              <FileText className="w-4 h-4 text-cyan-400" />
              <span className="font-semibold uppercase tracking-wider text-slate-200">
                5-Section Clinical Consultation Protocol
              </span>
            </div>

            <div className="prose prose-invert prose-xs max-w-none space-y-4 text-xs text-slate-300 leading-relaxed font-sans">
              {sections.length > 1 ? (
                sections.map((sec, idx) => (
                  <div key={idx} className="p-3.5 rounded-xl bg-slate-950/40 border border-slate-800/60 space-y-2">
                    <div className="whitespace-pre-wrap">{sec.trim()}</div>
                  </div>
                ))
              ) : (
                <div className="whitespace-pre-wrap p-4 rounded-xl bg-slate-950/40 border border-slate-800/60">
                  {rag.report_markdown}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right Column: Interactive Citations Inspector */}
        <div className="space-y-4">
          <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-3">
            <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
              <div className="flex items-center space-x-2 text-xs font-semibold text-slate-200 uppercase tracking-wider">
                <Bookmark className="w-4 h-4 text-indigo-400" />
                <span>Retrieved Citations ({rag.structured_citations.length})</span>
              </div>
              <span className="text-[10px] text-slate-400">Click to Inspect</span>
            </div>

            <div className="space-y-3 max-h-[600px] overflow-y-auto pr-1">
              {rag.structured_citations.map((cite) => {
                const isSelected = activeCitation === cite.citation_index;

                return (
                  <div
                    key={cite.citation_index}
                    onClick={() => setActiveCitation(isSelected ? null : cite.citation_index)}
                    className={`p-3.5 rounded-xl border text-xs cursor-pointer transition-all ${
                      isSelected
                        ? "bg-indigo-950/60 border-indigo-500 shadow-md shadow-indigo-950/50"
                        : "bg-slate-900/50 border-slate-800 hover:border-slate-700"
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-bold text-cyan-300 font-mono">
                        [{cite.citation_index}]
                      </span>
                      <span className="text-[10px] px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 border border-slate-700 font-mono">
                        {cite.source_type.toUpperCase()}
                      </span>
                    </div>

                    <h5 className="font-semibold text-slate-200 mt-1.5 leading-snug">
                      {cite.source_title}
                    </h5>

                    <p className="text-[11px] text-slate-400 mt-1 font-mono">
                      {cite.citation}
                    </p>

                    <div className="mt-2.5 p-2 rounded-lg bg-slate-950/70 border border-slate-800/60 text-[11px] text-slate-300 italic">
                      &ldquo;{cite.text_snippet}&rdquo;
                    </div>

                    {cite.supported_claims.length > 0 && (
                      <div className="mt-2 pt-2 border-t border-slate-800/60 space-y-1">
                        <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block">
                          Supported Clinical Claims:
                        </span>
                        {cite.supported_claims.map((claim, cIdx) => (
                          <div key={cIdx} className="text-[11px] text-emerald-300 flex items-center space-x-1.5">
                            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 flex-shrink-0" />
                            <span>{claim}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
