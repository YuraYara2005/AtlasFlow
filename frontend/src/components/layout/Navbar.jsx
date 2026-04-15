// src/components/layout/Navbar.jsx

import { NavLink } from "react-router-dom";
import { cn } from "@/utils/formatters";

/**
 * Top Navigation Bar
 * Features a frosted-glass effect, dynamic active states, and semantic layout.
 */
export default function Navbar() {
  // Navigation items defined as an array for easy scaling later
  const navItems = [
    { name: "Live Monitor", path: "/" },
    { name: "Admin Analytics", path: "/admin" },
  ];

  return (
    <header
      className={cn(
        // Sticky positioning with z-index to stay above scrolling content
        "sticky top-0 z-40 w-full",
        // Glassmorphism effect blending into the Slate theme
        "bg-slate-900/80 backdrop-blur-md",
        // Subtle separation from the main content
        "border-b border-slate-800",
        // Soft shadow for depth
        "shadow-sm"
      )}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">

          {/* --- LEFT: Brand & Logo --- */}
          <div className="flex items-center gap-8">
            {/* Logo */}
            <div className="flex-shrink-0 flex items-center gap-2 cursor-pointer select-none">
              {/* Optional: Simple SVG geometric logo placeholder */}
              <div className="w-8 h-8 rounded bg-gradient-to-br from-sky-400 to-sky-600 flex items-center justify-center shadow-[0_0_10px_rgba(56,189,248,0.4)]">
                <span className="text-white font-bold text-lg leading-none">A</span>
              </div>
              <span className="text-xl font-bold tracking-tight text-slate-100">
                Atlas<span className="text-sky-400">Flow</span>
              </span>
            </div>

            {/* --- CENTER/LEFT: Navigation Links --- */}
            <nav className="hidden md:flex items-center gap-6">
              {navItems.map((item) => (
                <NavLink
                  key={item.name}
                  to={item.path}
                  className={({ isActive }) =>
                    cn(
                      "relative px-1 py-2 text-sm font-medium transition-colors duration-200",
                      // Inactive state
                      !isActive && "text-slate-400 hover:text-slate-200",
                      // Active state: Brighter text + custom bottom indicator bar
                      isActive && [
                        "text-sky-400",
                        "after:absolute after:bottom-[-17px] after:left-0 after:w-full after:h-[2px]",
                        "after:bg-sky-400 after:rounded-t-md after:shadow-[0_-2px_8px_rgba(56,189,248,0.5)]"
                      ]
                    )
                  }
                >
                  {item.name}
                </NavLink>
              ))}
            </nav>
          </div>

          {/* --- RIGHT: Actions & Profile Placeholder --- */}
          <div className="flex items-center gap-4">
            {/* System Health Indicator (Mocked for now) */}
            <div className="hidden sm:flex items-center gap-2 px-3 py-1 rounded-full bg-slate-800/50 border border-slate-700">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
              </span>
              <span className="text-xs font-medium text-slate-300">System Online</span>
            </div>

            {/* Profile Avatar Placeholder */}
            <button
              type="button"
              className="h-8 w-8 rounded-full bg-slate-700 border border-slate-600 flex items-center justify-center hover:border-sky-400 transition-colors focus:outline-none focus:ring-2 focus:ring-sky-400 focus:ring-offset-2 focus:ring-offset-slate-900"
            >
              <span className="text-xs font-medium text-slate-300">YM</span>
            </button>
          </div>

        </div>
      </div>
    </header>
  );
}