/**
 * PersonaToggle.tsx
 *
 * Pill-shaped segmented control for switching between the two chat personas.
 * Drop this into your header and wire `selectedPersona` into your chat payload.
 *
 * Integration steps:
 *   1. Import and render <PersonaToggle> in your header component.
 *   2. Pass `onPersonaChange` down to wherever you call the /generate endpoint.
 *   3. In your fetch/submit handler, spread `{ persona: selectedPersona }` into the body.
 *
 * CSS variables consumed (map to your existing design tokens):
 *   --color-primary          : your Couture Orchid hex (e.g. #c084fc)
 *   --color-rose-gold        : Rose Gold accent         (e.g. #f4a67a)
 *   --glass-bg               : frosted glass fill       (e.g. rgba(255,255,255,0.06))
 *   --glass-border           : frosted glass border     (e.g. rgba(255,255,255,0.10))
 *
 * If your project uses Tailwind CSS variables, swap the inline style references
 * with your existing utility classes (e.g. `glass-strong`).
 */

import React, { useState } from "react";

// ── Types ──────────────────────────────────────────────────────────────────────

export type Persona = "executive" | "analyst";

interface PersonaToggleProps {
  value: Persona;
  onChange: (persona: Persona) => void;
  className?: string;
}

// ── Config ─────────────────────────────────────────────────────────────────────

const OPTIONS: { value: Persona; label: string }[] = [
  { value: "executive", label: "The Executive Brief" },
  { value: "analyst",   label: "The Analyst Workspace" },
];

// ── Component ──────────────────────────────────────────────────────────────────

export function PersonaToggle({ value, onChange, className = "" }: PersonaToggleProps) {
  return (
    <div
      className={className}
      style={{
        position: "relative",
        display: "inline-flex",
        alignItems: "center",
        borderRadius: "9999px",
        padding: "3px",
        background: "var(--glass-bg, rgba(255,255,255,0.06))",
        border: "1px solid var(--glass-border, rgba(255,255,255,0.10))",
        backdropFilter: "blur(12px)",
        WebkitBackdropFilter: "blur(12px)",
        boxShadow: "0 2px 20px rgba(192, 132, 252, 0.08), inset 0 1px 0 rgba(255,255,255,0.06)",
        gap: "2px",
      }}
      role="group"
      aria-label="Persona selector"
    >
      {OPTIONS.map((opt) => {
        const isActive = value === opt.value;
        return (
          <button
            key={opt.value}
            onClick={() => onChange(opt.value)}
            style={{
              position: "relative",
              zIndex: 1,
              display: "inline-flex",
              alignItems: "center",
              justifyContent: "center",
              borderRadius: "9999px",
              paddingTop: "6px",
              paddingBottom: "6px",
              paddingLeft: "16px",
              paddingRight: "16px",
              fontSize: "11px",
              fontWeight: isActive ? 600 : 400,
              letterSpacing: "0.06em",
              textTransform: "uppercase",
              whiteSpace: "nowrap",
              border: "none",
              cursor: "pointer",
              transition: "color 0.22s ease, background 0.22s ease, box-shadow 0.22s ease",
              background: isActive
                ? "linear-gradient(135deg, rgba(192,132,252,0.22) 0%, rgba(244,166,122,0.14) 100%)"
                : "transparent",
              color: isActive
                ? "var(--color-primary, #c084fc)"
                : "rgba(255,255,255,0.38)",
              boxShadow: isActive
                ? "0 0 12px rgba(192,132,252,0.20), 0 1px 0 rgba(255,255,255,0.06), inset 0 0 0 1px rgba(192,132,252,0.25)"
                : "none",
            }}
            aria-pressed={isActive}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}

// ── Standalone hook for use in a parent chat component ─────────────────────────

export function usePersona(initial: Persona = "executive") {
  const [persona, setPersona] = useState<Persona>(initial);
  return { persona, setPersona };
}
