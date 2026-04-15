// src/features/admin/ExplanationModal.jsx

import { useEffect, useRef } from "react";
import { cn } from "@/utils/formatters";
import Button from "@/components/common/Button";

/**
 * ExplanationModal component to display detailed AI reasoning.
 * Follows WAI-ARIA best practices for accessible dialogs.
 *
 * @param {object}   props
 * @param {boolean}  props.isOpen       - Toggles modal visibility.
 * @param {function} props.onClose      - Callback to close the modal.
 * @param {object}   props.explanation  - Data object containing the reasoning.
 */
export default function ExplanationModal({ isOpen, onClose, explanation }) {
  const modalRef = useRef(null);

  // --- Mock Data Fallback & Formatting ---
  const data = explanation || {
    issue: "Unknown Issue Flagged",
    confidence: 0,
    reasoning: ["No specific reasoning provided by the model."],
    logs: [],
  };

  const confidencePercentage = (data.confidence * 100).toFixed(1);

  // --- Accessibility & UX Effects ---
  useEffect(() => {
    // Handle Escape key to close
    const handleEscape = (e) => {
      if (e.key === "Escape" && isOpen) {
        onClose();
      }
    };

    if (isOpen) {
      // Prevent body scroll when modal is active
      document.body.style.overflow = "hidden";
      document.addEventListener("keydown", handleEscape);
      // Optional: Trap focus within modal (requires additional utility function)
    }

    return () => {
      // Restore body scroll and cleanup listener
      document.body.style.overflow = "unset";
      document.removeEventListener("keydown", handleEscape);
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    // --- Centereddimmed overlay ---
    <div
      className={cn(
        "fixed inset-0 z-50 flex items-center justify-center",
        "bg-slate-950/90 backdrop-blur-sm p-4",
        "transition-opacity duration-300"
      )}
      onClick={onClose} // Close on backdrop click
      role="dialog"
      aria-modal="true"
      aria-labelledby="explanation-title"
    >
      {/* --- Modal Content Container (Stops propagation) --- */}
      <div
        ref={modalRef}
        className={cn(
          "bg-slate-900 border border-slate-800 rounded-xl shadow-2xl",
          "w-full max-w-2xl max-h-[90vh] flex flex-col overflow-hidden",
          "scale-100 opacity-100 transition-all duration-300"
        )}
        onClick={(e) => e.stopPropagation()}
      >
        {/* --- HEADER --- */}
        <header className="px-6 py-5 border-b border-slate-800 flex items-center justify-between gap-4 sticky top-0 bg-slate-900 z-10">
          <div>
            <h2 id="explanation-title" className="text-xl font-bold text-slate-100 tracking-tight">
              Model reasoning
            </h2>
            <p className="text-sm text-slate-400 mt-1">
              Issue flag: <span className="font-semibold text-slate-200">{data.issue}</span>
            </p>
          </div>

          {/* Confidence Score Display */}
          <div className="flex flex-col items-end gap-1 px-4 py-2 rounded-lg bg-slate-800 border border-slate-700">
            <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">
              Confidence
            </span>
            <span className={cn("text-2xl font-extrabold tabular-nums",
              data.confidence > 0.8 ? "text-emerald-400" :
              data.confidence > 0.5 ? "text-amber-400" : "text-red-400"
            )}>
              {confidencePercentage}%
            </span>
          </div>
        </header>

        {/* --- SCROLLABLE CONTENT BODY --- */}
        <div className="flex-1 overflow-y-auto p-6 flex flex-col gap-8">

          {/* A) Reasoning Chain Section */}
          <section>
            <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-widest mb-4">
              Reasoning Chain
            </h3>
            <ul className="space-y-3 list-decimal list-outside pl-5">
              {data.reasoning.map((point, index) => (
                <li key={index} className="text-sm text-slate-300 leading-relaxed pl-1">
                  {point}
                </li>
              ))}
            </ul>
          </section>

          {/* B) Relevant Logs Section */}
          <section>
            <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-widest mb-4 sticky top-0 bg-slate-900 py-1">
              Supporting raw evidence (Raw Logs)
            </h3>
            <div className="flex flex-col gap-2.5">
              {data.logs.length === 0 ? (
                <div className="text-center py-6 text-slate-600 border border-dashed border-slate-800 rounded-lg">
                  No log evidence correlated.
                </div>
              ) : (
                data.logs.map((log, index) => (
                  <div
                    key={index}
                    className={cn(
                      "grid grid-cols-[100px_1fr_60px] gap-3 items-center",
                      "p-3 rounded-lg bg-slate-800/20 border border-slate-800",
                      "font-mono text-xs text-slate-300",
                      index < 3 && "border-sky-800 bg-sky-950/20 shadow-sm" // Highlighting 'top' logs
                    )}
                  >
                    <span className="text-slate-500 tabular-nums text-center">{log.timestamp}</span>
                    <span className="text-slate-200 leading-snug">{log.message}</span>
                    <span className="font-semibold text-sky-400 bg-sky-400/10 px-2 py-0.5 rounded text-center">
                      Aisle {log.aisle}
                    </span>
                  </div>
                ))
              )}
            </div>
          </section>

        </div>

        {/* --- FOOTER / ACTIONS --- sticky bottom */}
        <footer className="px-6 py-4 border-t border-slate-800 sticky bottom-0 bg-slate-900 z-10 flex justify-end">
          <Button variant="ghost" onClick={onClose}>
            Acknowledge & Close
          </Button>
        </footer>
      </div>
    </div>
  );
}