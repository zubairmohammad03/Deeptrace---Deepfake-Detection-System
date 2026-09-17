// ScanHistory.jsx — localStorage-backed scan history
import { useState, useEffect, useCallback } from "react";
import { HISTORY_KEY } from "../constants.js";

const VERDICT_COLOR  = { DEEPFAKE: "#ff4d6d", AUTHENTIC: "#00f5a0", SUSPICIOUS: "#ffd166" };
const FILTER_OPTS    = ["all", "deepfake", "authentic", "suspicious"];

function filterColor(f) {
  if (f === "deepfake")  return { bg: "rgba(255,77,109,0.1)",   border: "rgba(255,77,109,0.35)",   text: "#ff4d6d" };
  if (f === "authentic") return { bg: "rgba(0,245,160,0.1)",    border: "rgba(0,245,160,0.35)",    text: "#00f5a0" };
  if (f === "suspicious")return { bg: "rgba(255,209,102,0.1)",  border: "rgba(255,209,102,0.35)",  text: "#ffd166" };
  return { bg: "rgba(255,255,255,0.07)", border: "rgba(255,255,255,0.18)", text: "#fff" };
}

function formatTimestamp(iso) {
  const d = new Date(iso);
  const now = new Date();
  const diffDays = Math.floor((now - d) / 86400000);
  const time = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  if (diffDays === 0) return { date: "Today", time };
  if (diffDays === 1) return { date: "Yesterday", time };
  return { date: d.toLocaleDateString([], { month: "short", day: "numeric" }), time };
}

function formatSize(bytes) {
  if (!bytes) return "—";
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function loadHistory() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]"); } catch { return []; }
}

export default function ScanHistory({ onViewResult }) {
  const [history, setHistory] = useState([]);
  const [filter, setFilter]   = useState("all");
  const [search, setSearch]   = useState("");

  const refresh = useCallback(() => setHistory(loadHistory()), []);

  useEffect(() => {
    refresh();
    window.addEventListener("storage", refresh);
    return () => window.removeEventListener("storage", refresh);
  }, [refresh]);

  const clearHistory = () => {
    localStorage.removeItem(HISTORY_KEY);
    setHistory([]);
  };

  const exportCSV = () => {
    if (!history.length) return;
    const header = "timestamp,filename,size,verdict,confidence,fake_score,risk_level";
    const rows = history.map(h =>
      `"${h.timestamp}","${h.filename}","${formatSize(h.filesize)}","${h.verdict}",${h.confidence},${h.overall_score},"${h.risk_level}"`
    );
    const blob = new Blob([[header, ...rows].join("\n")], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `deeptrace-history-${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  // Filter + search
  const filtered = history.filter(h => {
    const matchFilter = filter === "all" || h.verdict?.toLowerCase() === filter;
    const matchSearch = !search || h.filename?.toLowerCase().includes(search.toLowerCase());
    return matchFilter && matchSearch;
  });

  // Group by relative date
  const groups = {};
  filtered.forEach(h => {
    const { date } = formatTimestamp(h.timestamp);
    if (!groups[date]) groups[date] = [];
    groups[date].push(h);
  });

  const stats = [
    { label: "Total Scans", value: history.length,                                           color: "rgba(255,255,255,0.75)" },
    { label: "Deepfakes",   value: history.filter(h => h.verdict === "DEEPFAKE").length,     color: "#ff4d6d" },
    { label: "Authentic",   value: history.filter(h => h.verdict === "AUTHENTIC").length,    color: "#00f5a0" },
    { label: "Suspicious",  value: history.filter(h => h.verdict === "SUSPICIOUS").length,   color: "#ffd166" },
  ];

  return (
    <div style={{ maxWidth: 900, margin: "0 auto" }}>

      {/* Header */}
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: 24, gap: 16, flexWrap: "wrap" }}>
        <div>
          <h1 style={{ fontSize: 22, fontWeight: 700, margin: "0 0 5px", color: "#fff", letterSpacing: -0.4 }}>Scan History</h1>
          <p style={{ fontSize: 13, color: "rgba(255,255,255,0.38)", margin: 0 }}>
            {history.length ? `${history.length} scan${history.length !== 1 ? "s" : ""} stored locally` : "No scans recorded yet"}
          </p>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button
            onClick={refresh}
            style={{
              padding: "8px 16px", borderRadius: 8,
              background: "rgba(255,255,255,0.04)",
              border: "1px solid rgba(255,255,255,0.1)",
              color: "rgba(255,255,255,0.45)", fontSize: 12,
              cursor: "pointer", fontFamily: "'IBM Plex Sans', system-ui, sans-serif",
              transition: "all 0.15s",
            }}
            onMouseEnter={e => { e.currentTarget.style.background = "rgba(255,255,255,0.08)"; e.currentTarget.style.color = "#fff"; }}
            onMouseLeave={e => { e.currentTarget.style.background = "rgba(255,255,255,0.04)"; e.currentTarget.style.color = "rgba(255,255,255,0.45)"; }}
          >↺ Refresh</button>
          <button
            onClick={exportCSV}
            disabled={!history.length}
            style={{
              padding: "8px 16px", borderRadius: 8,
              background: "rgba(255,255,255,0.05)",
              border: "1px solid rgba(255,255,255,0.1)",
              color: history.length ? "rgba(255,255,255,0.55)" : "rgba(255,255,255,0.2)", fontSize: 12,
              cursor: history.length ? "pointer" : "not-allowed",
              fontFamily: "'IBM Plex Sans', system-ui, sans-serif",
            }}
          >↓ Export CSV</button>
          {history.length > 0 && (
            <button
              onClick={clearHistory}
              style={{
                padding: "8px 16px", borderRadius: 8,
                background: "rgba(255,77,109,0.07)",
                border: "1px solid rgba(255,77,109,0.2)",
                color: "#ff4d6d", fontSize: 12,
                cursor: "pointer", fontFamily: "'IBM Plex Sans', system-ui, sans-serif",
                transition: "all 0.15s",
              }}
              onMouseEnter={e => { e.currentTarget.style.background = "rgba(255,77,109,0.14)"; }}
              onMouseLeave={e => { e.currentTarget.style.background = "rgba(255,77,109,0.07)"; }}
            >Clear All</button>
          )}
        </div>
      </div>

      {/* Stats */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12, marginBottom: 24 }}>
        {stats.map(s => (
          <div key={s.label} style={{
            padding: "16px 18px",
            background: "rgba(255,255,255,0.03)",
            border: "1px solid rgba(255,255,255,0.08)",
            borderRadius: 12, backdropFilter: "blur(10px)",
          }}>
            <div style={{ fontSize: 28, fontWeight: 700, color: s.color, fontFamily: "'Space Mono', monospace", marginBottom: 5, lineHeight: 1 }}>
              {s.value}
            </div>
            <div style={{ fontSize: 10, color: "rgba(255,255,255,0.28)", textTransform: "uppercase", letterSpacing: 1.2, fontFamily: "'Space Mono', monospace" }}>
              {s.label}
            </div>
          </div>
        ))}
      </div>

      {/* Search + filter row */}
      <div style={{ display: "flex", gap: 12, marginBottom: 24, flexWrap: "wrap", alignItems: "center" }}>
        {/* Search */}
        <div style={{ position: "relative", flex: "1 1 200px", minWidth: 160 }}>
          <span style={{
            position: "absolute", left: 11, top: "50%", transform: "translateY(-50%)",
            fontSize: 13, color: "rgba(255,255,255,0.25)", pointerEvents: "none",
          }}>🔍</span>
          <input
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search by filename…"
            style={{
              width: "100%", padding: "8px 12px 8px 32px",
              background: "rgba(255,255,255,0.05)",
              border: "1px solid rgba(255,255,255,0.1)",
              borderRadius: 8, color: "#fff", fontSize: 13,
              fontFamily: "'IBM Plex Sans', system-ui, sans-serif",
              outline: "none", transition: "border-color 0.18s",
            }}
            onFocus={e => { e.currentTarget.style.borderColor = "rgba(0,245,160,0.4)"; }}
            onBlur={e => { e.currentTarget.style.borderColor = "rgba(255,255,255,0.1)"; }}
          />
        </div>

        {/* Filter tabs */}
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {FILTER_OPTS.map(f => {
            const active = filter === f;
            const fc = filterColor(f);
            return (
              <button key={f} onClick={() => setFilter(f)} style={{
                padding: "6px 14px", borderRadius: 100,
                border: `1px solid ${active ? fc.border : "rgba(255,255,255,0.08)"}`,
                background: active ? fc.bg : "transparent",
                color: active ? fc.text : "rgba(255,255,255,0.38)",
                cursor: "pointer", fontSize: 12, fontWeight: active ? 600 : 400,
                textTransform: "capitalize",
                fontFamily: "'IBM Plex Sans', system-ui, sans-serif",
                transition: "all 0.15s",
              }}>{f}</button>
            );
          })}
        </div>
      </div>

      {/* Empty state */}
      {history.length === 0 && (
        <div style={{ textAlign: "center", padding: "72px 0", color: "rgba(255,255,255,0.2)" }}>
          <div style={{ fontSize: 40, marginBottom: 14, opacity: 0.35 }}>🕐</div>
          <div style={{ fontSize: 15, fontWeight: 500, marginBottom: 8, color: "rgba(255,255,255,0.3)" }}>No scans yet</div>
          <div style={{ fontSize: 13, color: "rgba(255,255,255,0.18)" }}>
            Run an analysis in Single Analysis — results are saved automatically
          </div>
        </div>
      )}

      {/* Results empty */}
      {history.length > 0 && filtered.length === 0 && (
        <div style={{ textAlign: "center", padding: "48px 0", color: "rgba(255,255,255,0.25)", fontSize: 14 }}>
          No results match this filter.
        </div>
      )}

      {/* Timeline grouped by date */}
      {Object.entries(groups).map(([dateLabel, items]) => (
        <div key={dateLabel} style={{ marginBottom: 30 }}>
          {/* Date divider */}
          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
            <span style={{
              fontSize: 10, fontFamily: "'Space Mono', monospace",
              color: "rgba(255,255,255,0.3)", textTransform: "uppercase", letterSpacing: 1.8, flexShrink: 0,
            }}>{dateLabel}</span>
            <div style={{ flex: 1, height: 1, background: "rgba(255,255,255,0.06)" }} />
            <span style={{ fontSize: 10, fontFamily: "'Space Mono', monospace", color: "rgba(255,255,255,0.2)", flexShrink: 0 }}>
              {items.length} scan{items.length !== 1 ? "s" : ""}
            </span>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
            {items.map((h, i) => {
              const vc = VERDICT_COLOR[h.verdict] || "#fff";
              const { time } = formatTimestamp(h.timestamp);
              return (
                <div
                  key={h.id}
                  style={{
                    display: "flex", gap: 12, alignItems: "center",
                    padding: "12px 16px",
                    background: "rgba(255,255,255,0.025)",
                    border: "1px solid rgba(255,255,255,0.07)",
                    borderRadius: 10,
                    animation: `dt-fade-up 0.38s ease ${i * 50}ms both`,
                    transition: "border-color 0.18s, background 0.18s",
                    cursor: "default",
                  }}
                  onMouseEnter={e => { e.currentTarget.style.borderColor = "rgba(255,255,255,0.13)"; e.currentTarget.style.background = "rgba(255,255,255,0.035)"; }}
                  onMouseLeave={e => { e.currentTarget.style.borderColor = "rgba(255,255,255,0.07)"; e.currentTarget.style.background = "rgba(255,255,255,0.025)"; }}
                >
                  {/* Verdict bar */}
                  <div style={{
                    width: 3, alignSelf: "stretch", borderRadius: 2, flexShrink: 0,
                    background: vc, boxShadow: `0 0 8px ${vc}55`,
                  }} />

                  {/* Thumbnail */}
                  {h.thumbnail ? (
                    <div style={{
                      width: 40, height: 40, borderRadius: 7, flexShrink: 0,
                      overflow: "hidden", border: "1px solid rgba(255,255,255,0.1)",
                    }}>
                      <img src={h.thumbnail} alt="" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
                    </div>
                  ) : (
                    <div style={{
                      width: 40, height: 40, borderRadius: 7, flexShrink: 0,
                      background: "rgba(255,255,255,0.05)",
                      border: "1px solid rgba(255,255,255,0.08)",
                      display: "flex", alignItems: "center", justifyContent: "center",
                      fontSize: 18,
                    }}>🖼</div>
                  )}

                  {/* Timestamp */}
                  <div style={{ width: 44, flexShrink: 0 }}>
                    <div style={{ fontSize: 11, fontFamily: "'Space Mono', monospace", color: "rgba(255,255,255,0.3)" }}>
                      {time}
                    </div>
                  </div>

                  {/* File info */}
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{
                      fontSize: 13, fontWeight: 600, color: "#fff",
                      whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", marginBottom: 2,
                    }}>
                      {h.filename}
                    </div>
                    <div style={{ fontSize: 11, color: "rgba(255,255,255,0.22)" }}>{formatSize(h.filesize)}</div>
                  </div>

                  {/* Scores */}
                  <div style={{ textAlign: "center", flexShrink: 0, width: 44 }}>
                    <div style={{ fontSize: 18, fontWeight: 700, color: vc, fontFamily: "'Space Mono', monospace", lineHeight: 1 }}>
                      {h.overall_score ?? "—"}
                    </div>
                    <div style={{ fontSize: 8, color: "rgba(255,255,255,0.22)", fontFamily: "'Space Mono', monospace", textTransform: "uppercase", letterSpacing: 0.5, marginTop: 2 }}>
                      score
                    </div>
                  </div>
                  <div style={{ textAlign: "center", flexShrink: 0, width: 52 }}>
                    <div style={{ fontSize: 13, fontWeight: 600, color: "rgba(255,255,255,0.7)", fontFamily: "'Space Mono', monospace" }}>
                      {h.confidence != null ? `${h.confidence}%` : "—"}
                    </div>
                    <div style={{ fontSize: 8, color: "rgba(255,255,255,0.22)", fontFamily: "'Space Mono', monospace", textTransform: "uppercase", letterSpacing: 0.5, marginTop: 2 }}>
                      conf
                    </div>
                  </div>

                  {/* Verdict badge */}
                  <span style={{
                    padding: "3px 9px", borderRadius: 6, flexShrink: 0,
                    background: `${vc}14`, border: `1px solid ${vc}30`,
                    fontSize: 10, fontWeight: 700, color: vc,
                    fontFamily: "'Space Mono', monospace", letterSpacing: 0.5,
                    whiteSpace: "nowrap",
                  }}>
                    {h.verdict || "—"}
                  </span>

                  {/* View button */}
                  {h.result && (
                    <button
                      onClick={() => onViewResult(h)}
                      style={{
                        padding: "6px 13px", borderRadius: 6, flexShrink: 0,
                        background: "rgba(0,245,160,0.07)",
                        border: "1px solid rgba(0,245,160,0.18)",
                        color: "#00f5a0", fontSize: 11, fontWeight: 600,
                        cursor: "pointer",
                        fontFamily: "'IBM Plex Sans', system-ui, sans-serif",
                        whiteSpace: "nowrap",
                        transition: "all 0.15s",
                      }}
                      onMouseEnter={e => { e.currentTarget.style.background = "rgba(0,245,160,0.15)"; }}
                      onMouseLeave={e => { e.currentTarget.style.background = "rgba(0,245,160,0.07)"; }}
                    >
                      View →
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}
