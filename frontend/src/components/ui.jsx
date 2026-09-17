// components/ui.jsx — Reusable UI primitives for DeepTrace

import { useState, useEffect } from "react";

export function ScoreRing({ score, label, color, size = 90 }) {
  const [animated, setAnimated] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setAnimated(true), 200);
    return () => clearTimeout(t);
  }, []);
  const r = (size - 12) / 2;
  const circ = 2 * Math.PI * r;
  const dash = animated ? (score / 100) * circ : 0;

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8 }}>
      <div style={{ position: "relative", width: size, height: size }}>
        <svg width={size} height={size} style={{ transform: "rotate(-90deg)", position: "absolute", top: 0, left: 0 }}>
          <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--dt-border-subtle)" strokeWidth={5} />
          <circle
            cx={size / 2}
            cy={size / 2}
            r={r}
            fill="none"
            stroke={color}
            strokeWidth={5}
            strokeDasharray={`${dash} ${circ}`}
            strokeLinecap="round"
            style={{ transition: "stroke-dasharray 1s cubic-bezier(.4,0,.2,1)" }}
          />
        </svg>
        <div
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            flexDirection: "column",
          }}
        >
          <span
            style={{
              fontSize: size * 0.2,
              fontWeight: 600,
              color,
              fontFamily: "var(--font-mono)",
              lineHeight: 1,
            }}
          >
            {score}
          </span>
          <span style={{ fontSize: size * 0.09, color: "var(--dt-muted)", fontFamily: "var(--font-mono)" }}>%</span>
        </div>
      </div>
      <span
        style={{
          fontSize: 10,
          color: "var(--dt-muted)",
          textTransform: "uppercase",
          letterSpacing: "0.08em",
          fontFamily: "var(--font-mono)",
          textAlign: "center",
        }}
      >
        {label}
      </span>
    </div>
  );
}

export function AnalysisBar({ label, score, findings, details, delay = 0 }) {
  const [animated, setAnimated] = useState(false);
  const [open, setOpen] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setAnimated(true), delay);
    return () => clearTimeout(t);
  }, [delay]);
  const color = score > 70 ? "var(--dt-danger)" : score > 40 ? "var(--dt-warn)" : "var(--dt-safe)";

  return (
    <div style={{ marginBottom: 16 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 6,
          cursor: "pointer",
          userSelect: "none",
        }}
        onClick={() => setOpen(!open)}
      >
        <span
          style={{
            fontSize: 12,
            color: "var(--dt-text)",
            fontFamily: "var(--font-mono)",
            textTransform: "uppercase",
            letterSpacing: "0.06em",
          }}
        >
          {label}
        </span>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 12, fontWeight: 600, color, fontFamily: "var(--font-mono)" }}>{score}/100</span>
          <span
            style={{
              fontSize: 10,
              color: "var(--dt-muted)",
              transition: "transform 0.2s",
              display: "inline-block",
              transform: open ? "rotate(180deg)" : "rotate(0deg)",
            }}
          >
            ▼
          </span>
        </div>
      </div>
      <div style={{ height: 4, background: "var(--dt-surface-2)", borderRadius: 2, overflow: "hidden" }}>
        <div
          style={{
            height: "100%",
            borderRadius: 2,
            width: animated ? `${score}%` : "0%",
            background: color,
            transition: "width 0.9s cubic-bezier(.4,0,.2,1)",
          }}
        />
      </div>
      {open && (
        <div
          style={{
            marginTop: 10,
            padding: "12px 14px",
            background: "var(--dt-surface-2)",
            borderRadius: 4,
            borderLeft: `3px solid ${color}`,
          }}
        >
          {details && (
            <p style={{ fontSize: 13, color: "var(--dt-muted)", margin: "0 0 10px", lineHeight: 1.65 }}>{details}</p>
          )}
          {findings?.map((f, i) => (
            <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: 8, marginTop: 6 }}>
              <span style={{ color, fontSize: 10, marginTop: 3, flexShrink: 0 }}>—</span>
              <span style={{ fontSize: 13, color: "var(--dt-text)", lineHeight: 1.5 }}>{f}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function ScanOverlay() {
  return (
    <div style={{ position: "absolute", inset: 0, pointerEvents: "none", zIndex: 10, overflow: "hidden" }}>
      <div
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          height: 1,
          background: "var(--dt-accent)",
          opacity: 0.35,
          animation: "scanLine 2.2s ease-in-out infinite",
        }}
      />
      <style>{`
        @keyframes scanLine {
          0% { top: 0; opacity: 0.2; }
          50% { opacity: 0.5; }
          100% { top: 100%; opacity: 0.2; }
        }
      `}</style>
    </div>
  );
}

export function Chip({ children, color }) {
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        padding: "4px 10px",
        background: color ? `${color}14` : "var(--dt-surface-2)",
        border: `1px solid ${color ? `${color}40` : "var(--dt-border-subtle)"}`,
        borderRadius: 4,
        fontSize: 12,
        color: color || "var(--dt-muted)",
        fontFamily: "var(--font-mono)",
      }}
    >
      {children}
    </span>
  );
}

export function SeverityBadge({ level }) {
  const map = {
    HIGH: "var(--dt-danger)",
    MEDIUM: "var(--dt-warn)",
    LOW: "var(--dt-safe)",
    CRITICAL: "var(--dt-danger)",
  };
  const color = map[level] || "var(--dt-muted)";
  return (
    <span
      style={{
        fontSize: 10,
        fontFamily: "var(--font-mono)",
        fontWeight: 600,
        padding: "3px 8px",
        borderRadius: 4,
        background: `${color}18`,
        color,
        border: `1px solid ${color}44`,
      }}
    >
      {level}
    </span>
  );
}
