// src/components/common/Loader.jsx

import { cn } from "@/utils/formatters";

// ---------------------------------------------------------------------------
// Size mapping for the SVG spinner
// ---------------------------------------------------------------------------
const SIZE_MAP = {
  sm: "h-4 w-4 border-2",
  md: "h-8 w-8 border-3",
  lg: "h-12 w-12 border-4",
};

/**
 * Reusable Loading Indicator.
 * Supports inline rendering (for cards/sections) or full-screen overlays.
 *
 * @param {object}   props
 * @param {"sm"|"md"|"lg"} [props.size="md"] - Dimensions of the spinner.
 * @param {boolean}  [props.fullScreen=false] - If true, covers the entire viewport.
 * @param {string}   [props.label]            - Optional text displayed below the spinner.
 * @param {string}   [props.className]        - Extra Tailwind classes for the wrapper.
 */
export default function Loader({
  size = "md",
  fullScreen = false,
  label,
  className,
  ...props
}) {
  const spinnerSize = SIZE_MAP[size] || SIZE_MAP.md;

  // The core spinner and label (used in both inline and fullscreen modes)
  const LoaderContent = (
    <div
      className={cn("flex flex-col items-center justify-center gap-3", className)}
      {...props}
    >
      {/* Using a custom SVG instead of a simple border-spinner for a more
        "high-tech" look. Inherits text color (text-sky-400 by default).
      */}
      <svg
        className={cn("animate-spin text-sky-400", spinnerSize)}
        xmlns="http://www.w3.org/2000/svg"
        fill="none"
        viewBox="0 0 24 24"
        role="status"
        aria-label={label || "Loading"}
      >
        <circle
          className="opacity-20"
          cx="12"
          cy="12"
          r="10"
          stroke="currentColor"
          strokeWidth="4"
        />
        <path
          className="opacity-80"
          fill="currentColor"
          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
        />
      </svg>

      {/* Optional Label with a subtle pulse effect to indicate "thinking" */}
      {label ? (
        <span className="text-sm font-medium text-slate-300 animate-pulse select-none">
          {label}
        </span>
      ) : (
        /* Screen reader fallback if no visual label is provided */
        <span className="sr-only">Loading...</span>
      )}
    </div>
  );

  // If fullScreen is true, wrap it in a fixed glassmorphism overlay
  if (fullScreen) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/80 backdrop-blur-sm transition-all">
        {LoaderContent}
      </div>
    );
  }

  // Otherwise, return as an inline block
  return LoaderContent;
}