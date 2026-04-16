// src/features/query/useQuery.js

import { useState, useCallback } from "react";

/**
 * Custom hook to manage AI query operations.
 * Handles the state lifecycle from the initial question to the final diagnosis and evidence.
 * * @returns {Object} Query state and the submit function.
 */
export default function useQuery() {
  const [result, setResult] = useState(null);
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  /**
   * Simulates an AI processing cycle.
   * In production, this would be an async POST request to your AI inference endpoint.
   * * @param {string} userQuestion - The natural language query from the user.
   */
  const submitQuery = useCallback(async (userQuestion) => {
    if (!userQuestion.trim()) return;

    setLoading(true);
    setError(null);

    // Clear previous results while "thinking" to give a fresh feel
    setResult(null);
    setLogs([]);

    try {
      // Simulate API latency (e.g., waiting for the Transformer model to process)
      await new Promise((resolve) => setTimeout(resolve, 1800));

      // --- MOCK RESPONSE DATA ---
      // This mimics the structure expected by ResultCard and LogsViewer
      const mockResult = {
        issue: "Spatiotemporal Desync in Aisle 12",
        confidence: 0.91,
        recommended_action: "Reset the spatial encoders on AGV-7 and re-sync the global clock across Node B. The model detected a 120ms drift in log timestamps."
      };

      const mockLogs = [
        {
          timestamp: "01:10:05 AM",
          message: "Node B: Clock drift detected exceeding 50ms threshold",
          aisle: 12,
          isImportant: true
        },
        {
          timestamp: "01:10:02 AM",
          message: "AGV-7: Reporting spatial positioning variance of 15cm",
          aisle: 12,
          isImportant: true
        },
        {
          timestamp: "01:09:55 AM",
          message: "System: Routine log collection heartbeat",
          aisle: 4,
          isImportant: false
        },
        {
          timestamp: "01:09:40 AM",
          message: "Node A: Data pipeline operational",
          aisle: 1,
          isImportant: false
        }
      ];

      setResult(mockResult);
      setLogs(mockLogs);
    } catch (err) {
      setError("Failed to communicate with the AtlasFlow engine. Please check your connection.");
      console.error("Query Error:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  return {
    result,
    logs,
    loading,
    error,
    submitQuery,
  };
}