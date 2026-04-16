// src/features/query/LogsViewer.jsx

import { cn } from "@/utils/formatters";

/**
 * Logs Viewer Component.
 * Renders a list of system telemetry logs that informed the AI's diagnosis.
 * * @param {object} props
 * @param {Array}  props.logs - Array of log objects to display.
 */
export default function LogsViewer({ logs = [] }) {
  // --- EMPTY STATE ---
  if (!logs || logs.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center p-8 border border-dashed border-slate-800 rounded-xl bg-slate-900/20">
        <span className="text-slate-500 text-sm font-medium italic">
          No logs available for this query
        </span>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      {/* Section Header */}
      <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest px-1">
        Evidence Logs
      </h3>

      {/* --- SCROLLABLE CONTAINER --- */}
      <div className="max-h-[350px] overflow-y-auto pr-2 flex flex-col gap-2 custom-scrollbar">
        {logs.map((log, index) => (
          <div
            key={`${log.timestamp}-${index}`}
            className={cn(
              "group relative flex flex-col sm:flex-row sm:items-center gap-3 p-3 rounded-lg border transition-all duration-200",

              // Standard Log Style
              !log.isImportant && "bg-slate-900/40 border-slate-800/60 hover:border-slate-700 hover:bg-slate-800/30",

              // Important Log Style: High contrast, accent border, and subtle glow
              log.isImportant && [
                "bg-sky-950/20 border-sky-500/30",
                "border-l-4 border-l-sky-400 shadow-[shadow:0_0_15px_rgba(56,189,248,0.05)]",
                "before:absolute before:inset-0 before:bg-sky-400/5 before:rounded-lg"
              ]
            )}
          >
            {/* Timestamp & Aisle - Technical Meta Data */}
            <div className="flex items-center gap-3 shrink-0">
              <span className="font-mono text-[11px] text-slate-500 tabular-nums">
                {log.timestamp}
              </span>
              <span className={cn(
                "font-mono text-[10px] font-bold px-1.5 py-0.5 rounded uppercase tracking-tighter",
                log.isImportant ? "bg-sky-400/20 text-sky-400" : "bg-slate-800 text-slate-400"
              )}>
                Aisle {log.aisle}
              </span>
            </div>

            {/* Message Content */}
            <div className="flex-1">
              <p className={cn(
                "text-sm leading-relaxed",
                log.isImportant ? "text-slate-100 font-medium" : "text-slate-400"
              )}>
                {log.message}
              </p>
            </div>

            {/* "Important" Indicator Tag (Visible only on highlighted logs) */}
            {log.isImportant && (
              <div className="shrink-0">
                <span className="text-[10px] font-black text-sky-500/80 uppercase tracking-tighter bg-sky-500/5 px-2 py-1 rounded-full border border-sky-500/10">
                  Critical Path
                </span>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}