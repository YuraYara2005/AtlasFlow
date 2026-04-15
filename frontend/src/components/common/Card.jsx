// src/components/common/Card.jsx

import { cn } from "@/utils/formatters";

/**
 * Glassmorphism surface card for dashboard content.
 *
 * Provides a consistent elevated container with a slate/cyber aesthetic:
 * semi-transparent dark background, blurred backdrop, and a subtle border.
 *
 * @param {object}          props
 * @param {React.ReactNode} props.children   - Content to render inside the card.
 * @param {string}          [props.padding]  - Tailwind padding class. Defaults to "p-6".
 * @param {string}          [props.className] - Extra Tailwind classes merged via twMerge.
 */
export default function Card({ children, padding = "p-6", className, ...props }) {
  return (
    <div
      className={cn(
        // --- Shape ---
        "rounded-xl overflow-hidden",

        // --- Glassmorphism surface ---
        "bg-slate-800/50",          // semi-transparent dark slate
        "backdrop-blur-md",         // frosted-glass blur
        "border border-slate-700",  // subtle 1-px border

        // --- Elevation ---
        "shadow-[0_4px_24px_rgba(0,0,0,0.35)]", // soft deep shadow

        // --- Spacing (overridable via `padding` prop) ---
        padding,

        // --- Caller overrides (wins over everything above) ---
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
}
