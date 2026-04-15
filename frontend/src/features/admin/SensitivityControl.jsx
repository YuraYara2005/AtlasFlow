// src/features/admin/SensitivityControl.jsx

import { cn } from "@/utils/formatters";

/**
 * Control component for tuning the AI model's confidence threshold.
 *
 * @param {object}   props
 * @param {number}   props.value    - Current sensitivity value (0 to 1).
 * @param {function} props.onChange - Callback fired when the slider moves.
 */
export default function SensitivityControl({ value = 0.85, onChange }) {
  // Convert 0-1 float to a clean percentage for display
  const displayPercentage = Math.round(value * 100);

  const handleSliderChange = (e) => {
    const newValue = parseFloat(e.target.value);
    if (onChange) {
      onChange(newValue);
    }
  };

  return (
    <div className="flex flex-col gap-4 p-5 rounded-lg bg-slate-800/40 border border-slate-700/60">

      {/* --- HEADER: Label & Value Display --- */}
      <div className="flex items-start justify-between">
        <div className="flex flex-col gap-1">
          <label
            htmlFor="model-sensitivity"
            className="text-sm font-semibold text-slate-200"
          >
            Model Sensitivity
          </label>
          <span className="text-xs text-slate-400">
            Adjust the baseline confidence threshold for flagging anomalies.
          </span>
        </div>

        {/* Real-time Value Readout */}
        <div className="flex items-center justify-center px-3 py-1 rounded-md bg-slate-900 border border-slate-700 shadow-inner">
          <span className="text-sm font-mono font-bold text-sky-400 tabular-nums">
            {displayPercentage}%
          </span>
        </div>
      </div>

      {/* --- SLIDER CONTROL --- */}
      <div className="flex flex-col gap-2 mt-1">
        <input
          id="model-sensitivity"
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={value}
          onChange={handleSliderChange}
          className={cn(
            "w-full h-1.5 rounded-lg appearance-none cursor-pointer bg-slate-700",
            // Tailwind's accent property natively styles the slider thumb
            "accent-sky-400",
            "focus:outline-none focus:ring-2 focus:ring-sky-400/50 focus:ring-offset-2 focus:ring-offset-slate-900",
            "transition-all duration-150"
          )}
          aria-valuemin="0"
          aria-valuemax="100"
          aria-valuenow={displayPercentage}
        />

        {/* Scale Indicators */}
        <div className="flex justify-between px-1 text-[10px] font-semibold text-slate-500 uppercase tracking-widest select-none">
          <span>Lenient</span>
          <span>Strict</span>
        </div>
      </div>

    </div>
  );
}