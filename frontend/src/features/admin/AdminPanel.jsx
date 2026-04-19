// src/features/admin/AdminPanel.jsx

import { useState } from "react";
import Card from "@/components/common/Card";
import Button from "@/components/common/Button";
import ExplanationModal from "./ExplanationModal";
import ExportCenter from "./ExportCenter";
import IncidentTable from "./IncidentTable";       // NEW: Imported the table
import SensitivityControl from "./SensitivityControl"; // NEW: Imported the slider
import useAdmin from "./useAdmin";                 // NEW: Imported the state hook

/**
 * Main Administrative Control Panel.
 * Composes individual feature sections into a unified operational view.
 */
export default function AdminPanel() {
  // --- STATE MANAGEMENT ---
  // Hook handles data fetching and logic automatically
  const { incidents, sensitivity, loading, updateSensitivity } = useAdmin();

  // Local state for the modal
  const [isModalOpen, setIsModalOpen] = useState(false);

  // Mock Explanation Data (Still used for the "Sample Explanation" button)
  const mockExplanation = {
    issue: "Anomaly-12: Unsynchronized Pallet Pickup Detected",
    confidence: 0.941,
    reasoning: [
      "Cross-attention weights spiked on Node B data pipeline during time window T4.",
      "Vibration sensor data correlated with anomalous acceleration curves in Node D (AGV).",
      "Model predicted a 94% probability of misaligned grab attempt based on historical training data."
    ],
    logs: [
      { timestamp: "10:42:01 AM", message: "Node B: Spike detected in Cross-Attention (weights > 0.8)", aisle: "C3" },
      { timestamp: "10:42:00 AM", message: "Node D: AGV reports 15% deviation in grab accuracy", aisle: "C3" },
      { timestamp: "10:41:59 AM", message: "Node D: Vibration sensor alert (Z-axis peak)", aisle: "C3" },
      { timestamp: "10:41:58 AM", message: "Node A: Log entry 'Pallet identified'", aisle: "C3" }
    ]
  };

  return (
    <div className="flex flex-col gap-8 max-w-5xl mx-auto w-full pb-10">

      {/* --- PAGE HEADER --- */}
      <header className="flex flex-col gap-2">
        <h1 className="text-3xl font-bold text-slate-100 tracking-tight">
          Admin Control Panel
        </h1>
        <p className="text-slate-400 text-sm">
          System monitoring, incident reporting, and compliance data export.
        </p>
      </header>

      {/* --- SECTION 1: Incident Overview --- */}
      <section>
        <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider mb-4">
          Incident Overview
        </h2>
        <Card className="flex flex-col gap-4">
          <p className="text-sm text-slate-400 mb-2">
            Recent anomalies detected by the AtlasFlow engine.
          </p>

          {/* NEW: Conditional rendering based on the loading state from our hook */}
          {loading ? (
            <div className="flex items-center justify-center py-12 border border-dashed border-slate-700 rounded-lg bg-slate-900/30">
              <span className="text-slate-400 text-sm font-medium animate-pulse">
                Fetching system telemetry...
              </span>
            </div>
          ) : (
            <IncidentTable incidents={incidents} />
          )}

          <div className="mt-2 flex justify-end">
            <Button variant="ghost">View Full Log History</Button>
          </div>
        </Card>
      </section>

      {/* --- GRID FOR SECONDARY ACTIONS --- */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">

        {/* --- SECTION 2: Data Export (Left Column) --- */}
        <section className="flex flex-col h-full">
          <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider mb-4">
            Data Extraction
          </h2>
          <ExportCenter />
        </section>

        {/* --- SECTION 3: Tuning & Interpretability (Right Column) --- */}
        <section className="flex flex-col h-full gap-8">

          {/* NEW: Sensitivity Control */}
          <div>
            <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider mb-4">
              Model Tuning
            </h2>
            <SensitivityControl
              value={sensitivity}
              onChange={updateSensitivity}
            />
          </div>

          {/* AI Explanations */}
          <div className="flex-1 flex flex-col">
            <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider mb-4">
              Model Interpretability
            </h2>
            <Card className="flex-1 flex flex-col justify-between gap-4">
              <div>
                <h3 className="text-lg font-medium text-slate-200 mb-1">Audit AI Reasoning</h3>
                <p className="text-sm text-slate-400">
                  Review the Cross-Attention Transformer's logic for flagged incidents.
                </p>
              </div>
              <div className="mt-4 border-t border-slate-700/50 pt-4">
                <Button variant="primary" className="w-full" onClick={() => setIsModalOpen(true)}>
                  View Sample Explanation
                </Button>
              </div>
            </Card>
          </div>

        </section>

      </div>

      {/* Mount the Modal and pass states/data */}
      <ExplanationModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        explanation={mockExplanation}
      />

    </div>
  );
}