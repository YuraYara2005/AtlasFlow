// src/features/admin/ExportCenter.jsx

import { useState } from "react";
import Card from "@/components/common/Card";
import Button from "@/components/common/Button";

/**
 * Export Center Utility Panel.
 * Handles the extraction of system logs and reports with simulated network latency.
 */
export default function ExportCenter() {
  const [isExporting, setIsExporting] = useState(false);

  // Mock export handler
  const handleExport = () => {
    setIsExporting(true);

    // Simulate a 2-second file generation delay from the backend
    setTimeout(() => {
      setIsExporting(false);
      // In production, this would trigger a Blob download or pre-signed URL
    }, 2000);
  };

  return (
    // Note: We use h-full so it stretches perfectly if placed in a CSS Grid
    <Card className="flex flex-col h-full justify-between">

      {/* Header Area */}
      <div className="mb-6">
        <h3 className="text-lg font-medium text-slate-200 mb-1">
          Export Reports
        </h3>
        <p className="text-sm text-slate-400">
          Download system telemetry and incident logs for offline compliance review.
        </p>
      </div>

      {/* Export Options List */}
      <div className="flex flex-col gap-4">

        {/* Primary Export: Excel / CSV */}
        <div className="flex items-center justify-between p-4 rounded-lg bg-slate-800/40 border border-slate-700/50">
          <div className="flex flex-col gap-1">
            <span className="text-sm font-semibold text-slate-200">
              System Telemetry (Excel)
            </span>
            <span className="text-xs text-slate-500">
              Raw sensor data and pipeline logs.
            </span>
          </div>
          <Button
            variant="success"
            onClick={handleExport}
            isLoading={isExporting}
          >
            {isExporting ? "Generating..." : "Export Data"}
          </Button>
        </div>

        {/* Placeholder for scaling: PDF Summaries */}
        <div className="flex items-center justify-between p-4 rounded-lg bg-slate-900/50 border border-slate-800/50 opacity-60 pointer-events-none select-none">
          <div className="flex flex-col gap-1">
            <span className="text-sm font-semibold text-slate-400">
              Executive Audit (PDF)
            </span>
            <span className="text-xs text-slate-600">
              High-level anomaly summaries.
            </span>
          </div>
          <Button variant="ghost" disabled>
            Coming Soon
          </Button>
        </div>

      </div>
    </Card>
  );
}