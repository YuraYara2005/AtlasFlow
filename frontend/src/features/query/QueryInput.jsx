// src/features/query/QueryInput.jsx

import { useState } from "react";
import { cn } from "@/utils/formatters";
import Button from "@/components/common/Button";

/**
 * AI Interaction Input Component.
 * Acts as the primary command bar for querying the AtlasFlow system.
 *
 * @param {object}   props
 * @param {function} props.onSubmit - Callback fired with the query string when submitted.
 */
export default function QueryInput({ onSubmit }) {
  const [query, setQuery] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault(); // Prevents the page from refreshing on submit

    const trimmedQuery = query.trim();
    if (!trimmedQuery) return; // Failsafe against empty submissions

    // Pass the question up to the parent component
    if (onSubmit) {
      onSubmit(trimmedQuery);
    }

    // Clear the input after submission
    setQuery("");
  };

  return (
    <div className="w-full max-w-4xl mx-auto">
      {/* The Container: Card-like, frosted glass effect.
        We use focus-within to highlight the entire container when the input is active.
      */}
      <div
        className={cn(
          "p-2 rounded-2xl bg-slate-900/60 border border-slate-700/60 shadow-lg backdrop-blur-md",
          "transition-all duration-300 ease-out",
          "focus-within:border-sky-500/50 focus-within:bg-slate-900/80 focus-within:shadow-[0_0_20px_rgba(56,189,248,0.1)]"
        )}
      >
        <form onSubmit={handleSubmit} className="flex items-center gap-3">

          {/* The Input Field */}
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask about system status, logs, or recent anomalies..."
            className={cn(
              "flex-1 bg-transparent px-4 py-3",
              "text-slate-100 placeholder-slate-500 text-base",
              "focus:outline-none",
              "w-full"
            )}
            autoComplete="off"
            spellCheck="false"
          />

          {/* The Submit Button */}
          <div className="pr-1">
            <Button
              type="submit"
              variant="primary"
              disabled={!query.trim()}
              className="px-6 py-2.5 rounded-xl font-medium tracking-wide"
            >
              Ask AI
            </Button>
          </div>

        </form>
      </div>

      {/* Optional Helper Text under the input */}
      <div className="mt-2 text-center">
        <p className="text-xs text-slate-500">
          Press <kbd className="font-mono bg-slate-800 px-1.5 py-0.5 rounded border border-slate-700 text-[10px]">Enter</kbd> to submit
        </p>
      </div>
    </div>
  );
}