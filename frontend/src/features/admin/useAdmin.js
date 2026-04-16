// src/features/admin/useAdmin.js

import { useState, useEffect, useCallback } from "react";

/**
 * Custom hook to manage Admin Control Panel state and operations.
 * Separates business logic and data fetching from UI presentation.
 * * @returns {Object} Admin state and control functions
 */
export default function useAdmin() {
  // --- STATE ---
  const [incidents, setIncidents] = useState([]);
  const [sensitivity, setSensitivity] = useState(0.7); // Default 70% threshold
  const [loading, setLoading] = useState(true);

  // --- ACTIONS ---

  /**
   * Simulates an API call to fetch recent AI incidents.
   * Wrapped in useCallback to maintain referential equality across renders.
   */
  const fetchIncidents = useCallback(() => {
    setLoading(true);

    // Simulate network latency (e.g., waiting for the Python backend)
    setTimeout(() => {
      const mockData = [
        {
          id: "INC-092",
          timestamp: "10:42:01 AM",
          issue: "Unsynchronized Pallet Pickup Detected",
          confidence: 0.94,
          action: "Halt AGV & Inspect",
          status: "critical",
        },
        {
          id: "INC-091",
          timestamp: "09:15:22 AM",
          issue: "Data Pipeline Sync Latency (Node B)",
          confidence: 0.88,
          action: "Restart Worker Node",
          status: "warning",
        },
        {
          id: "INC-090",
          timestamp: "08:30:00 AM",
          issue: "Minor Thermal Fluctuation in Server Rack 3",
          confidence: 0.65,
          action: "Monitor System Logs",
          status: "normal",
        },
      ];

      setIncidents(mockData);
      setLoading(false);
    }, 1200); // 1.2 second delay for realistic feel
  }, []);

  /**
   * Updates the global model sensitivity threshold.
   * @param {number} value - New sensitivity value (0.0 to 1.0)
   */
  const updateSensitivity = useCallback((value) => {
    setSensitivity(value);

    // In a production environment, changing the threshold would likely
    // trigger a re-evaluation of incidents. You could trigger a refetch here:
    // fetchIncidents();
  }, []);

  // --- INITIALIZATION ---

  // Automatically fetch incidents when the dashboard mounts
  useEffect(() => {
    fetchIncidents();
  }, [fetchIncidents]);

  // --- EXPORT ---
  return {
    incidents,
    sensitivity,
    loading,
    updateSensitivity,
    fetchIncidents,
  };
}