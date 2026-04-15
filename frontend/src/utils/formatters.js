// src/utils/formatters.js
// Shared class-merging utility used across all components.

import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Merge Tailwind classes safely.
 * - clsx collapses falsy values and arrays.
 * - twMerge resolves Tailwind conflicts (e.g. p-4 + p-6 → p-6).
 *
 * @param {...(string|undefined|null|boolean|object)} inputs
 * @returns {string}
 */
export function cn(...inputs) {
  return twMerge(clsx(inputs));
}