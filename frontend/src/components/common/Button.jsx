// src/components/common/Button.jsx

import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";
import { cn } from "@/utils/formatters";
/**
 * Merge Tailwind classes safely, resolving conflicts with tailwind-merge
 * and handling falsy values with clsx.
 */

// ---------------------------------------------------------------------------
// Variant base styles
// ---------------------------------------------------------------------------
const VARIANT_STYLES = {
  primary:
    "bg-sky-400 text-white hover:bg-sky-500 hover:shadow-[0_0_14px_rgba(56,189,248,0.55)] focus-visible:ring-sky-400",
  success:
    "bg-emerald-500 text-white hover:bg-emerald-600 focus-visible:ring-emerald-500",
  danger:
    "bg-red-500 text-white hover:bg-red-600 focus-visible:ring-red-500",
  ghost:
    "bg-transparent text-slate-200 border border-slate-600 hover:bg-slate-700 focus-visible:ring-slate-500",
};

// ---------------------------------------------------------------------------
// Spinner
// ---------------------------------------------------------------------------
function Spinner() {
  return (
    <svg
      className="animate-spin h-4 w-4"
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
      aria-hidden="true"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
      />
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Button
// ---------------------------------------------------------------------------

/**
 * Reusable Button component.
 *
 * @param {object}   props
 * @param {React.ReactNode} props.children      - Button label / content.
 * @param {function} [props.onClick]            - Click handler.
 * @param {"button"|"submit"|"reset"} [props.type="button"] - Native button type.
 * @param {"primary"|"success"|"danger"|"ghost"} [props.variant="primary"] - Visual style.
 * @param {boolean}  [props.isLoading=false]    - Show spinner and disable interaction.
 * @param {boolean}  [props.disabled=false]     - Disable the button.
 * @param {string}   [props.className]          - Extra Tailwind classes (merged safely).
 */
export default function Button({
  children,
  onClick,
  type = "button",
  variant = "primary",
  isLoading = false,
  disabled = false,
  className,
  ...props
}) {
  const isDisabled = disabled || isLoading;

  return (
    <button
      // eslint-disable-next-line react/button-has-type
      type={type}
      onClick={onClick}
      disabled={isDisabled}
      aria-disabled={isDisabled}
      aria-busy={isLoading}
      className={cn(
        // --- Base styles ---
        "inline-flex items-center justify-center gap-2",
        "rounded-md px-4 py-2",
        "font-medium text-sm",
        "transition-all duration-200",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-900",
        "select-none",

        // --- Variant ---
        VARIANT_STYLES[variant] ?? VARIANT_STYLES.primary,

        // --- Disabled / loading state ---
        isDisabled && "opacity-50 pointer-events-none cursor-not-allowed",

        // --- Caller overrides (merged last, wins over everything above) ---
        className
      )}
      {...props}
    >
      {isLoading && <Spinner />}
      {children}
    </button>
  );
}
