// src/features/admin/IncidentTable.jsx

import { cn } from "@/utils/formatters";

/**
 * Data table for rendering system incidents.
 * Designed for high legibility and quick scannability in an operations context.
 *
 * @param {object} props
 * @param {Array}  props.incidents - Array of incident objects.
 */
export default function IncidentTable({ incidents = [] }) {
  // --- EMPTY STATE ---
  if (!incidents || incidents.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center p-12 border border-dashed border-slate-700 rounded-lg bg-slate-900/30">
        <span className="text-slate-400 text-sm font-medium">
          No incidents detected
        </span>
      </div>
    );
  }

  // --- STYLE HELPER FOR BADGES ---
  const getStatusStyles = (status) => {
    switch (status) {
      case "critical":
        return "bg-red-400/10 text-red-400 border-red-400/20";
      case "warning":
        return "bg-amber-400/10 text-amber-400 border-amber-400/20";
      case "normal":
      default:
        return "bg-emerald-400/10 text-emerald-400 border-emerald-400/20";
    }
  };

  return (
    <div className="w-full overflow-x-auto rounded-lg border border-slate-800 bg-slate-900/50 shadow-sm">
      <table className="w-full text-left border-collapse">

        {/* --- TABLE HEADER --- */}
        <thead>
          <tr className="bg-slate-800/80 border-b border-slate-700 text-xs font-semibold text-slate-400 uppercase tracking-wider">
            <th className="px-4 py-3 font-medium">ID</th>
            <th className="px-4 py-3 font-medium">Timestamp</th>
            <th className="px-4 py-3 font-medium">Issue</th>
            <th className="px-4 py-3 font-medium">Confidence</th>
            <th className="px-4 py-3 font-medium">Recommended Action</th>
            <th className="px-4 py-3 font-medium text-right">Status</th>
          </tr>
        </thead>

        {/* --- TABLE BODY --- */}
        <tbody className="divide-y divide-slate-800">
          {incidents.map((incident) => (
            <tr
              key={incident.id}
              className="hover:bg-slate-800/40 transition-colors duration-150 group"
            >
              {/* ID Badge */}
              <td className="px-4 py-3 whitespace-nowrap">
                <span className="text-xs font-mono text-sky-400 bg-sky-400/10 px-2 py-1 rounded">
                  {incident.id}
                </span>
              </td>

              {/* Timestamp */}
              <td className="px-4 py-3 whitespace-nowrap text-sm text-slate-400">
                {incident.timestamp}
              </td>

              {/* Issue Description */}
              <td className="px-4 py-3 text-sm text-slate-200">
                {incident.issue}
              </td>

              {/* Confidence Percentage */}
              <td className="px-4 py-3 whitespace-nowrap text-sm">
                <span className="text-slate-300 font-medium">
                  {(incident.confidence * 100).toFixed(0)}%
                </span>
              </td>

              {/* Recommended Action */}
              <td className="px-4 py-3 text-sm text-slate-400">
                {incident.action}
              </td>

              {/* Status Badge */}
              <td className="px-4 py-3 whitespace-nowrap text-right">
                <span
                  className={cn(
                    "inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border capitalize",
                    getStatusStyles(incident.status)
                  )}
                >
                  {incident.status}
                </span>
              </td>
            </tr>
          ))}
        </tbody>

      </table>
    </div>
  );
}