// src/pages/Dashboard.jsx

import QueryInput from "@/features/query/QueryInput";
import ResultCard from "@/features/query/ResultCard";
import LogsViewer from "@/features/query/LogsViewer";
import useQuery from "@/features/query/useQuery";
import { cn } from "@/utils/formatters";

/**
 * Live Monitor Dashboard.
 * The primary interface for real-time AI interaction and system diagnostics.
 */
export default function Dashboard() {
  const { result, logs, loading, error, submitQuery } = useQuery();

  return (
    <div className="flex flex-col gap-10 w-full max-w-5xl mx-auto pb-20">

      {/* --- SECTION 1: SEARCH & COMMAND --- */}
      <section className="flex flex-col items-center text-center gap-4 pt-4">
        <div className="flex flex-col gap-2">
          <h1 className="text-3xl font-bold text-slate-100 tracking-tight">
            Live System Monitor
          </h1>
          <p className="text-slate-400 text-sm max-w-lg">
            Query the AtlasFlow engine for real-time spatiotemporal diagnostics
            and automated incident resolution.
          </p>
        </div>

        {/* The AI Command Bar */}
        <div className="w-full mt-4">
          <QueryInput onSubmit={submitQuery} />
        </div>
      </section>

      {/* --- SECTION 2: AI OUTPUT & EVIDENCE --- */}
      <section className="flex flex-col gap-8 min-h-[400px]">

        {/* LOADING STATE */}
        {loading && (
          <div className="flex flex-col items-center justify-center py-20 gap-4">
            <div className="w-12 h-12 border-4 border-sky-500/20 border-t-sky-500 rounded-full animate-spin" />
            <p className="text-slate-500 text-sm font-medium animate-pulse">
              AtlasFlow is analyzing spatiotemporal logs...
            </p>
          </div>
        )}

        {/* ERROR STATE */}
        {error && (
          <div className="p-4 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-center text-sm">
            {error}
          </div>
        )}

        {/* INITIAL STATE (When no query has been made yet) */}
        {!loading && !result && !error && (
          <div className="flex flex-col items-center justify-center py-20 opacity-40 grayscale">
             <div className="w-20 h-20 rounded-full border-2 border-dashed border-slate-700 flex items-center justify-center mb-4">
                <span className="text-2xl text-slate-700">?</span>
             </div>
             <p className="text-slate-500 text-sm">System idle. Awaiting operational query.</p>
          </div>
        )}

        {/* SUCCESS STATE: RESULTS & LOGS */}
        {!loading && result && (
          <div className={cn(
            "flex flex-col gap-8 animate-in fade-in slide-in-from-bottom-4 duration-700"
          )}>
            {/* The primary AI decision */}
            <ResultCard result={result} />

            {/* The supporting evidence logs */}
            <div className="bg-slate-900/40 border border-slate-800/60 rounded-2xl p-6">
               <LogsViewer logs={logs} />
            </div>
          </div>
        )}

      </section>
    </div>
  );
}