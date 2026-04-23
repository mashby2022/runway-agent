/**
 * TvGuide.tsx — 24-hour TV Guide for Couture One
 *
 * Renders a pixel-perfect vertical timeline for a single broadcast day.
 * - Every minute of the 24-hour day is represented (1440 px tall at PX_PER_MIN=1)
 * - Real slots rendered in brand colours by tribe
 * - Off-Air / Continuous Stream fillers rendered as muted placeholders
 * - A "NOW" indicator line scrolls into view on mount
 * - Keyboard-accessible slot cards
 *
 * Usage:
 *   <TvGuide day="Wednesday" apiBase="http://localhost:8081" />
 *
 * Depends on: React 18+, TailwindCSS (or swap className for inline styles)
 */

import React, { useEffect, useRef, useState, useCallback } from "react";

// ── Constants ─────────────────────────────────────────────────────────────────

const PX_PER_MIN = 2;          // 2px per minute → 2880px for 24 hours
const HOUR_LABELS = Array.from({ length: 25 }, (_, i) => i); // 0–24

// Style Tribe → accent colour map
const TRIBE_COLORS: Record<string, string> = {
  "Avant-Garde":     "#9333ea", // purple-600
  "Heritage Couture": "#b45309", // amber-700
  "Minimalist":       "#0369a1", // sky-700
  "Romantic Feminine": "#be185d", // pink-700
  "Street & Youth":   "#15803d", // green-700
  "":                 "#475569", // slate-600 (unknown)
};

const DAYPART_BADGE: Record<string, string> = {
  "Overnight":  "bg-slate-700 text-slate-200",
  "Daytime":    "bg-sky-100   text-sky-800",
  "Prime Time": "bg-amber-100 text-amber-800",
  "Late Night": "bg-purple-900 text-purple-200",
};

// ── Types ─────────────────────────────────────────────────────────────────────

interface EpgSlot {
  time:              string;
  ends_at:           string;
  title:             string;
  tribe:             string;
  block_duration_min: number;
  start_offset_min:  number;
  daypart:           string;
  event_type:        string;
  block_label:       string;
  status:            "past" | "live" | "future";
  isLive:            boolean;
  is_filler:         boolean;
  runtime_min:       number;
  interstitial_min:  number;
}

interface FullDayResponse {
  day:             string;
  theme:           string;
  tribe:           string;
  total_minutes:   number;
  slot_count:      number;
  filler_count:    number;
  live_index:      number | null;
  now:             string;
  now_offset_min:  number | null;
  slots:           EpgSlot[];
  _engine:         string;
  controller_status?: string;
  active_gatekeeper?: string;
}

// ── Component ─────────────────────────────────────────────────────────────────

interface TvGuideProps {
  day?:     string;   // "Monday"–"Sunday", defaults to today
  apiBase?: string;   // e.g. "http://localhost:8081"
  pxPerMin?: number;  // override pixels-per-minute scale
}

export default function TvGuide({
  day,
  apiBase = "http://localhost:8081",
  pxPerMin = PX_PER_MIN,
}: TvGuideProps) {
  const [data, setData]       = useState<FullDayResponse | null>(null);
  const [error, setError]     = useState<string>("");
  const [loading, setLoading] = useState(true);
  const nowLineRef            = useRef<HTMLDivElement>(null);
  const scrollRef             = useRef<HTMLDivElement>(null);

  const totalHeight = 1440 * pxPerMin; // always 24h

  // ── Fetch ──────────────────────────────────────────────────────────────────

  const fetchGuide = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      if (day) params.set("day", day);
      params.set("current_time", new Date().toISOString());
      const res = await fetch(`${apiBase}/api/epg/full-day?${params}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json: FullDayResponse = await res.json();
      setData(json);
      setError("");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to load EPG");
    } finally {
      setLoading(false);
    }
  }, [day, apiBase]);

  useEffect(() => { fetchGuide(); }, [fetchGuide]);

  // ── Scroll "NOW" into view ─────────────────────────────────────────────────

  useEffect(() => {
    if (!data || data.now_offset_min == null) return;
    const container = scrollRef.current;
    if (!container) return;
    const nowPx = data.now_offset_min * pxPerMin;
    // Centre the "NOW" line in the viewport with a 120px lookahead
    container.scrollTop = Math.max(0, nowPx - container.clientHeight / 2 + 120);
  }, [data, pxPerMin]);

  // ── Helpers ────────────────────────────────────────────────────────────────

  function slotTop(slot: EpgSlot)    { return slot.start_offset_min * pxPerMin; }
  function slotHeight(slot: EpgSlot) { return slot.block_duration_min * pxPerMin; }
  function tribeColor(tribe: string) { return TRIBE_COLORS[tribe] ?? TRIBE_COLORS[""]; }

  // ── Loading / error states ─────────────────────────────────────────────────

  if (loading) return (
    <div className="flex items-center justify-center h-64 text-slate-400 text-sm">
      Loading Couture One schedule…
    </div>
  );

  if (error) return (
    <div className="flex items-center justify-center h-64 text-red-400 text-sm">
      {error}
    </div>
  );

  if (!data) return null;

  const nowOffsetPx = data.now_offset_min != null ? data.now_offset_min * pxPerMin : null;

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col h-full bg-slate-950 text-white font-sans select-none">

      {/* ── Header ── */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800 shrink-0">
        <div>
          <h2 className="text-lg font-semibold tracking-tight">
            {data.day}
            <span className="ml-2 text-xs font-normal text-slate-400 uppercase tracking-widest">
              {data.theme}
            </span>
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            {data.slot_count - data.filler_count} titles · {data.filler_count} off-air blocks
          </p>
        </div>

        {/* Bypass badge */}
        {data.controller_status === "MANUAL_BYPASS" && (
          <div className="flex items-center gap-1.5 rounded px-2 py-1 bg-red-900/60 border border-red-700 text-red-300 text-xs font-mono">
            <span className="w-1.5 h-1.5 rounded-full bg-red-400 animate-pulse" />
            MANUAL OVERRIDE · {data.active_gatekeeper}
          </div>
        )}

        <button
          onClick={fetchGuide}
          className="text-xs text-slate-400 hover:text-white border border-slate-700 rounded px-2 py-1 transition-colors"
        >
          Refresh
        </button>
      </div>

      {/* ── Guide body: hour rail + slots ── */}
      <div className="flex flex-1 overflow-hidden">

        {/* Hour rail */}
        <div className="relative shrink-0 w-14 bg-slate-900 border-r border-slate-800 overflow-hidden">
          <div style={{ height: totalHeight, position: "relative" }}>
            {HOUR_LABELS.map(h => (
              <div
                key={h}
                className="absolute left-0 right-0 flex items-start justify-end pr-2"
                style={{ top: h * 60 * pxPerMin }}
              >
                <span className="text-[10px] text-slate-500 leading-none mt-0.5">
                  {h === 24 ? "00" : String(h).padStart(2, "0")}:00
                </span>
                {/* Hour tick line — extends into the slot column */}
                <div
                  className="absolute left-0 right-0 border-t border-slate-800"
                  style={{ top: 0, width: "100vw" }}
                />
              </div>
            ))}
          </div>
        </div>

        {/* Scrollable slot column */}
        <div
          ref={scrollRef}
          className="relative flex-1 overflow-y-scroll overflow-x-hidden"
          style={{ height: "100%" }}
        >
          <div
            className="relative"
            style={{ height: totalHeight }}
          >

            {/* Hour grid lines */}
            {HOUR_LABELS.map(h => (
              <div
                key={h}
                className="absolute left-0 right-0 border-t border-slate-800/60 pointer-events-none"
                style={{ top: h * 60 * pxPerMin }}
              />
            ))}

            {/* NOW indicator */}
            {nowOffsetPx != null && (
              <div
                ref={nowLineRef}
                className="absolute left-0 right-0 z-20 pointer-events-none"
                style={{ top: nowOffsetPx }}
              >
                <div className="flex items-center gap-1">
                  <div className="w-2 h-2 rounded-full bg-red-500 shrink-0" />
                  <div className="flex-1 h-px bg-red-500" />
                  <span className="text-[10px] text-red-400 font-mono pr-2 shrink-0">
                    {new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                  </span>
                </div>
              </div>
            )}

            {/* Slots */}
            {data.slots.map((slot, i) => {
              const top    = slotTop(slot);
              const height = slotHeight(slot);
              const color  = tribeColor(slot.tribe);
              const isFiller = slot.is_filler;

              if (isFiller) {
                // Off-Air filler — minimal, muted
                return (
                  <div
                    key={`filler-${i}`}
                    className="absolute left-1 right-1 rounded-sm overflow-hidden"
                    style={{ top, height: Math.max(height, 2) }}
                  >
                    <div
                      className="w-full h-full flex items-center px-2 opacity-30"
                      style={{ background: "repeating-linear-gradient(45deg, #1e293b 0, #1e293b 4px, #0f172a 4px, #0f172a 8px)" }}
                    >
                      {height >= 20 && (
                        <span className="text-[9px] text-slate-500 uppercase tracking-widest truncate">
                          {slot.title}
                        </span>
                      )}
                    </div>
                  </div>
                );
              }

              // Real slot
              const isLive = slot.isLive;
              const isPast = slot.status === "past";

              return (
                <div
                  key={`slot-${i}`}
                  tabIndex={0}
                  role="button"
                  aria-label={`${slot.time} – ${slot.title}`}
                  className={[
                    "absolute left-1 right-1 rounded overflow-hidden transition-all",
                    "focus:outline-none focus:ring-2 focus:ring-white/40",
                    isLive  ? "ring-2 ring-white shadow-lg shadow-white/10 z-10" : "",
                    isPast  ? "opacity-50" : "",
                  ].join(" ")}
                  style={{ top, height: Math.max(height, 4) }}
                >
                  {/* Colour accent bar */}
                  <div
                    className="absolute left-0 top-0 bottom-0 w-1"
                    style={{ background: color }}
                  />

                  {/* Card body */}
                  <div
                    className="w-full h-full pl-2.5 pr-1.5 pt-1 pb-1 flex flex-col justify-between"
                    style={{ background: `${color}18` }}
                  >
                    {height >= 16 && (
                      <div className="flex items-start justify-between gap-1">
                        <span
                          className="text-[11px] font-medium leading-tight truncate"
                          style={{ color: isLive ? "#fff" : "#cbd5e1" }}
                        >
                          {slot.title}
                        </span>
                        {isLive && (
                          <span className="shrink-0 text-[9px] font-bold uppercase bg-red-600 text-white px-1 rounded leading-tight">
                            LIVE
                          </span>
                        )}
                      </div>
                    )}

                    {height >= 36 && (
                      <div className="flex items-center gap-1.5 flex-wrap mt-auto">
                        <span className="text-[9px] text-slate-400 font-mono">
                          {slot.time}–{slot.ends_at}
                        </span>
                        {slot.tribe && (
                          <span
                            className="text-[9px] px-1 rounded-sm"
                            style={{ background: `${color}30`, color }}
                          >
                            {slot.tribe}
                          </span>
                        )}
                        {slot.daypart && (
                          <span className={`text-[9px] px-1 rounded-sm ${DAYPART_BADGE[slot.daypart] ?? ""}`}>
                            {slot.daypart}
                          </span>
                        )}
                        {slot.interstitial_min > 0 && (
                          <span className="text-[9px] text-amber-500/70">
                            +{slot.interstitial_min}m interstitial
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* ── Footer ── */}
      <div className="shrink-0 px-4 py-2 border-t border-slate-800 flex items-center justify-between">
        <span className="text-[10px] text-slate-600 font-mono">
          [ Engine: {data._engine} | Scale: {pxPerMin}px/min | Total: {totalHeight}px ]
        </span>
        <span className="text-[10px] text-slate-600">
          Couture One · 24-hour Broadcast Cycle
        </span>
      </div>
    </div>
  );
}
