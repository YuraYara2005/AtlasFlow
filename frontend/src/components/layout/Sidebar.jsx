// src/components/layout/Sidebar.jsx

import { NavLink } from "react-router-dom";
import { cn } from "@/utils/formatters";

/**
 * Operational Sidebar Control Panel.
 * Functions as primary navigation while providing real-time system context,
 * status indicators, and quick metrics for the AtlasFlow engine.
 */
export default function Sidebar({ className }) {
  const navItems = [
    { name: "Live Monitor", path: "/" },
    { name: "Admin Analytics", path: "/admin" },
  ];

  // Mock data for Quick Insights
  const insights = [
    { label: "Active Alerts", value: "3", status: "text-amber-400" },
    { label: "Delays Detected", value: "1", status: "text-red-400" },
    { label: "System Load", value: "42%", status: "text-emerald-400" },
  ];

  return (
    <aside
      className={cn(
        "w-64 flex flex-col h-full",
        "border-r border-slate-800 bg-slate-900/80 backdrop-blur-md",
        "select-none",
        className
      )}
    >
      {/* Scrollable Content Area */}
      <div className="flex-1 overflow-y-auto flex flex-col">

        {/* --- SECTION A: Modules (Navigation) --- */}
        <section className="py-6 px-4">
          <h2 className="mb-3 px-2 text-[10px] font-bold text-slate-500 uppercase tracking-widest">
            System Modules
          </h2>
          <nav className="flex flex-col gap-1.5">
            {navItems.map((item) => (
              <NavLink
                key={item.name}
                to={item.path}
                className={({ isActive }) =>
                  cn(
                    "block px-3 py-2 text-sm font-medium transition-all duration-200",
                    "border-l-2 rounded-r-md",
                    !isActive && [
                      "border-transparent text-slate-400",
                      "hover:bg-slate-800/40 hover:text-slate-200"
                    ],
                    isActive && [
                      "bg-slate-800/80 text-sky-400 border-sky-400",
                      "shadow-[inset_1px_0_0_rgba(56,189,248,0.15)]"
                    ]
                  )
                }
              >
                {item.name}
              </NavLink>
            ))}
          </nav>
        </section>

        <hr className="border-slate-800 mx-4" />

        {/* --- SECTION B: System Status --- */}
        <section className="py-6 px-6">
          <h2 className="mb-4 text-[10px] font-bold text-slate-500 uppercase tracking-widest">
            Core Status
          </h2>
          <div className="flex items-center gap-3 px-3 py-2.5 rounded-md bg-slate-800/30 border border-slate-700/50">
            <span className="relative flex h-2.5 w-2.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500"></span>
            </span>
            <span className="text-sm font-medium text-slate-200">System Online</span>
          </div>
        </section>

        <hr className="border-slate-800 mx-4" />

        {/* --- SECTION C: Quick Insights --- */}
        <section className="py-6 px-6 flex-1">
          <h2 className="mb-4 text-[10px] font-bold text-slate-500 uppercase tracking-widest">
            Live Metrics
          </h2>
          <div className="flex flex-col gap-3">
            {insights.map((stat, idx) => (
              <div
                key={idx}
                className="flex items-center justify-between p-3 rounded-md bg-slate-800/20 border border-slate-800/60"
              >
                <span className="text-xs font-medium text-slate-400">{stat.label}</span>
                <span className={cn("text-sm font-bold", stat.status)}>
                  {stat.value}
                </span>
              </div>
            ))}
          </div>
        </section>

      </div>

      {/* --- SECTION D: Footer --- */}
      <footer className="p-4 border-t border-slate-800 bg-slate-900/50">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-slate-500">AtlasFlow Core</span>
          <span className="text-[10px] font-mono text-slate-600 bg-slate-800/50 px-2 py-0.5 rounded">v1.0.4</span>
        </div>
      </footer>
    </aside>
  );
}