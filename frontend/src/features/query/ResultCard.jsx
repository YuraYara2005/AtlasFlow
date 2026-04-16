// src/features/query/ResultCard.jsx

import { cn } from "@/utils/formatters";

/**
 * AI Result Display Component.
 * Presents the engine's diagnosis and recommended action with a strong visual hierarchy.
 *
 * @param {object} props
 * @param {object} props.result - The AI decision object.
 */
export default function ResultCard({ result }) {
  // Graceful fallback if no result is passed
  if (!result) return null;

  // Format confidence and determine semantic color
  const confidencePercent = Math.round(result.confidence * 100);
  const confidenceColor =
    result.confidence >= 0.85 ? "text-emerald-400 bg-emerald-400/10 border-emerald-400/20" :
    result.confidence >= 0.60 ? "text-amber-400 bg-amber-400/10 border-amber-400/20" :
    "text-red-400 bg-red-400/10 border-red-400/20";

  return (
    <div className="w-full max-w-4xl mx-auto overflow-hidden rounded-2xl bg-slate-900/80 border border-slate-700/60 shadow-2xl backdrop-blur-sm">

      {/* Top Accent Line */}
      <div className="h-1 w-full bg-gradient-to-r from-sky-500 via-indigo-500 to-purple-500" />

      <div className="p-6 sm:p-8 flex flex-col gap-6">

        {/* --- HEADER: Issue & Confidence --- */}
        <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4">

          <div className="flex flex-col gap-1.5">
            <span className="text-xs font-bold text-slate-500 uppercase tracking-widest">
              Detected Issue
            </span>
            <h2 className="text-2xl font-bold text-slate-100 tracking-tight leading-snug">
              {result.issue}
            </h2>
          </div>

          {/* Confidence Badge */}
          <div className="shrink-0 flex flex-col items-end">
            <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1">
              Confidence
            </span>
            <div className={cn(
              "flex items-center justify-center px-3 py-1.5 rounded-lg border",
              confidenceColor
            )}>
              <span className="text-lg font-mono font-bold tabular-nums leading-none">
                {confidencePercent}%
              </span>
            </div>
          </div>

        </div>

        {/* --- DIVIDER --- */}
        <hr className="border-slate-800" />

        {/* --- BODY: Recommended Action --- */}
        <div className="flex flex-col gap-2 rounded-xl bg-slate-800/40 p-5 border border-slate-700/50">
          <div className="flex items-center gap-2 mb-1">
            {/* Simple icon purely made with Tailwind shapes for visual anchor */}
            <div className="w-2 h-2 rounded-full bg-sky-400 animate-pulse" />
            <span className="text-xs font-bold text-sky-400 uppercase tracking-widest">
              Recommended Action
            </span>
          </div>

          <p className="text-slate-200 text-base leading-relaxed">
            {result.recommended_action}
          </p>
        </div>

      </div>
    </div>
  );
}