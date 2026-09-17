// BatchAnalysis.jsx — Fully functional multi-file batch analysis
import { useState, useRef, useCallback } from "react";
import { analyzeMedia } from "../services/claudeApi.js";

const VERDICT_COLOR = { DEEPFAKE: "#ff4d6d", AUTHENTIC: "#00f5a0", SUSPICIOUS: "#ffd166" };

function formatSize(bytes) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

async function makeThumbnail(file) {
  if (file.type.startsWith("video/")) return null;
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        const MAX = 96;
        const ratio = Math.min(MAX / img.width, MAX / img.height, 1);
        const canvas = document.createElement("canvas");
        canvas.width  = Math.round(img.width  * ratio);
        canvas.height = Math.round(img.height * ratio);
        canvas.getContext("2d").drawImage(img, 0, 0, canvas.width, canvas.height);
        resolve(canvas.toDataURL("image/jpeg", 0.75));
      };
      img.onerror = () => resolve(null);
      img.src = e.target.result;
    };
    reader.onerror = () => resolve(null);
    reader.readAsDataURL(file);
  });
}

function StatCard({ label, value, color }) {
  return (
    <div style={{
      padding: "16px 18px",
      background: "rgba(255,255,255,0.03)",
      border: "1px solid rgba(255,255,255,0.08)",
      borderRadius: 12,
      backdropFilter: "blur(10px)",
    }}>
      <div style={{ fontSize: 28, fontWeight: 700, color, fontFamily: "'Space Mono', monospace", marginBottom: 5, lineHeight: 1 }}>
        {value}
      </div>
      <div style={{ fontSize: 10, color: "rgba(255,255,255,0.28)", textTransform: "uppercase", letterSpacing: 1.3, fontFamily: "'Space Mono', monospace" }}>
        {label}
      </div>
    </div>
  );
}

function ProgressBar({ value, color = "linear-gradient(90deg,#00f5a0,#00d9f5)" }) {
  return (
    <div style={{ height: 3, background: "rgba(255,255,255,0.07)", borderRadius: 2, overflow: "hidden", marginTop: 8 }}>
      <div style={{
        height: "100%", width: `${value}%`,
        background: color, borderRadius: 2,
        transition: "width 0.45s ease",
        boxShadow: "0 0 8px rgba(0,245,160,0.4)",
      }} />
    </div>
  );
}

export default function BatchAnalysis() {
  const [items, setItems]       = useState([]);
  const [isDragging, setDrag]   = useState(false);
  const [isRunning, setRunning] = useState(false);
  const fileInputRef            = useRef();
  const runningRef              = useRef(false);
  const itemsRef                = useRef([]);

  // Keep ref in sync so runAll can snapshot without stale closure
  // eslint-disable-next-line react-hooks/exhaustive-deps
  itemsRef.current = items;

  const addFiles = useCallback(async (fileList) => {
    const accepted = Array.from(fileList).filter(f =>
      /^(image\/(jpeg|png|webp)|video\/(mp4|quicktime|avi|x-msvideo))$/.test(f.type)
    );
    if (!accepted.length) return;
    const newItems = await Promise.all(accepted.map(async (file) => ({
      id: `${Date.now()}-${Math.random()}`,
      file,
      name: file.name,
      size: formatSize(file.size),
      status: "queued",
      verdict: null, confidence: null, overall_score: null,
      thumbnail: await makeThumbnail(file),
      progress: 0,
      error: null,
    })));
    setItems(prev => [...prev, ...newItems]);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault(); setDrag(false);
    if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files);
  }, [addFiles]);

  const runAll = useCallback(async () => {
    if (runningRef.current) return;
    runningRef.current = true;
    setRunning(true);

    // Snapshot queued items from ref (always fresh)
    const queued = itemsRef.current.filter(i => i.status === "queued");

    for (const { id, file: currentFile } of queued) {
      // Mark processing
      setItems(prev => prev.map(i => i.id === id ? { ...i, status: "processing", progress: 8 } : i));

      // Animate progress while backend works
      const ticker = setInterval(() => {
        setItems(prev => prev.map(i =>
          i.id === id && i.status === "processing"
            ? { ...i, progress: Math.min(i.progress + Math.random() * 12 + 4, 88) }
            : i
        ));
      }, 550);

      try {
        const result = await analyzeMedia(null, currentFile.type, currentFile);
        clearInterval(ticker);
        setItems(prev => prev.map(i => i.id === id ? {
          ...i, status: "done", progress: 100,
          verdict: result.verdict,
          confidence: result.confidence,
          overall_score: result.overall_score,
        } : i));
      } catch (err) {
        clearInterval(ticker);
        setItems(prev => prev.map(i => i.id === id ? {
          ...i, status: "error", progress: 0,
          error: err.message || "Analysis failed",
        } : i));
      }
    }

    runningRef.current = false;
    setRunning(false);
  }, []);

  const clearAll  = () => { if (!isRunning) setItems([]); };
  const clearDone = () => { if (!isRunning) setItems(prev => prev.filter(i => i.status === "queued")); };

  const exportResults = () => {
    const done = items.filter(i => i.status === "done");
    if (!done.length) return;

    // CSV
    const header = "filename,size,verdict,confidence,fake_score";
    const rows = done.map(i =>
      `"${i.name}","${i.size}","${i.verdict}",${i.confidence},${i.overall_score}`
    );
    const csv = [header, ...rows].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `deeptrace-batch-${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const stats = [
    { label: "Total",      value: items.length,                                    color: "rgba(255,255,255,0.7)" },
    { label: "Queued",     value: items.filter(i => i.status === "queued").length,  color: "rgba(255,255,255,0.35)" },
    { label: "Processing", value: items.filter(i => i.status === "processing").length, color: "#00d9f5" },
    { label: "Completed",  value: items.filter(i => i.status === "done").length,    color: "#00f5a0" },
  ];

  return (
    <div style={{ maxWidth: 960, margin: "0 auto" }}>

      {/* Header */}
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: 24, gap: 16, flexWrap: "wrap" }}>
        <div>
          <h1 style={{ fontSize: 22, fontWeight: 700, margin: "0 0 5px", color: "#fff", letterSpacing: -0.4 }}>Batch Analysis</h1>
          <p style={{ fontSize: 13, color: "rgba(255,255,255,0.38)", margin: 0 }}>Process multiple images and videos through the forensic engine</p>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button
            onClick={() => fileInputRef.current?.click()}
            style={{
              padding: "9px 18px", borderRadius: 8, flexShrink: 0,
              background: "rgba(0,245,160,0.08)",
              border: "1px solid rgba(0,245,160,0.22)",
              color: "#00f5a0", fontWeight: 600,
              cursor: "pointer", fontSize: 13,
              fontFamily: "'IBM Plex Sans', system-ui, sans-serif",
              transition: "all 0.18s",
            }}
            onMouseEnter={e => { e.currentTarget.style.background = "rgba(0,245,160,0.15)"; }}
            onMouseLeave={e => { e.currentTarget.style.background = "rgba(0,245,160,0.08)"; }}
          >
            + Add Files
          </button>
          <button
            onClick={exportResults}
            disabled={!items.some(i => i.status === "done")}
            style={{
              padding: "9px 18px", borderRadius: 8, flexShrink: 0,
              background: items.some(i => i.status === "done") ? "linear-gradient(135deg,#00f5a0,#00d9f5)" : "rgba(255,255,255,0.06)",
              border: "none",
              color: items.some(i => i.status === "done") ? "#080b12" : "rgba(255,255,255,0.25)",
              fontWeight: 700, cursor: items.some(i => i.status === "done") ? "pointer" : "not-allowed",
              fontSize: 13,
              fontFamily: "'IBM Plex Sans', system-ui, sans-serif",
              boxShadow: items.some(i => i.status === "done") ? "0 4px 18px rgba(0,245,160,0.25)" : "none",
              transition: "all 0.2s",
            }}
          >
            ↓ Export CSV
          </button>
        </div>
      </div>

      {/* Stats */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12, marginBottom: 24 }}>
        {stats.map(s => <StatCard key={s.label} {...s} />)}
      </div>

      {/* Drop zone */}
      <div
        onDragOver={e => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        style={{
          border: `2px dashed ${isDragging ? "#00f5a0" : "rgba(255,255,255,0.1)"}`,
          borderRadius: 14, padding: "36px 24px",
          textAlign: "center", cursor: "pointer",
          background: isDragging ? "rgba(0,245,160,0.06)" : "rgba(255,255,255,0.018)",
          transition: "all 0.22s cubic-bezier(0.4,0,0.2,1)",
          marginBottom: 24,
          boxShadow: isDragging ? "0 0 30px rgba(0,245,160,0.15), inset 0 0 20px rgba(0,245,160,0.04)" : "none",
        }}
      >
        <div style={{
          width: 52, height: 52, borderRadius: 14, margin: "0 auto 14px",
          background: isDragging ? "rgba(0,245,160,0.14)" : "rgba(255,255,255,0.05)",
          border: `1px solid ${isDragging ? "rgba(0,245,160,0.4)" : "rgba(255,255,255,0.1)"}`,
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 24, transition: "all 0.22s",
          transform: isDragging ? "scale(1.1) translateY(-2px)" : "scale(1)",
          boxShadow: isDragging ? "0 8px 24px rgba(0,245,160,0.2)" : "none",
        }}>📂</div>
        <div style={{ fontSize: 15, fontWeight: 600, color: isDragging ? "#00f5a0" : "#fff", marginBottom: 6, transition: "color 0.2s" }}>
          {isDragging ? "Release to add files" : "Drop files here or click to browse"}
        </div>
        <div style={{ fontSize: 12, color: "rgba(255,255,255,0.28)" }}>
          JPEG · PNG · WebP · MP4 · MOV · AVI &nbsp;·&nbsp; Up to 20 files
        </div>
      </div>

      <input
        ref={fileInputRef} type="file" multiple
        accept="image/jpeg,image/png,image/webp,video/mp4,video/quicktime,video/avi,video/x-msvideo"
        style={{ display: "none" }}
        onChange={e => { addFiles(e.target.files); e.target.value = ""; }}
      />

      {/* Queue header + controls */}
      {items.length > 0 && (
        <>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
            <h3 style={{
              fontSize: 10, fontFamily: "'Space Mono', monospace",
              textTransform: "uppercase", letterSpacing: 1.8,
              color: "rgba(255,255,255,0.28)", margin: 0,
            }}>
              Files Queue ({items.length})
            </h3>
            <div style={{ display: "flex", gap: 8 }}>
              <button onClick={clearAll} disabled={isRunning} style={{
                padding: "6px 14px", borderRadius: 7,
                background: "transparent", border: "1px solid rgba(255,255,255,0.1)",
                color: "rgba(255,255,255,0.4)", fontSize: 12,
                cursor: isRunning ? "not-allowed" : "pointer",
                fontFamily: "'IBM Plex Sans', system-ui, sans-serif",
                opacity: isRunning ? 0.5 : 1,
              }}>Clear All</button>
              <button
                onClick={runAll}
                disabled={isRunning || !items.some(i => i.status === "queued")}
                style={{
                  padding: "6px 16px", borderRadius: 7,
                  background: isRunning ? "rgba(0,217,245,0.08)" : "rgba(0,245,160,0.1)",
                  border: `1px solid ${isRunning ? "rgba(0,217,245,0.25)" : "rgba(0,245,160,0.25)"}`,
                  color: isRunning ? "#00d9f5" : "#00f5a0",
                  fontSize: 12, fontWeight: 600,
                  cursor: (isRunning || !items.some(i => i.status === "queued")) ? "not-allowed" : "pointer",
                  fontFamily: "'IBM Plex Sans', system-ui, sans-serif",
                  opacity: (!isRunning && !items.some(i => i.status === "queued")) ? 0.4 : 1,
                  transition: "all 0.2s",
                }}>
                {isRunning ? "⟳ Running…" : "▶ Run All"}
              </button>
            </div>
          </div>

          {/* File cards */}
          <div style={{ display: "flex", flexDirection: "column", gap: 8, marginBottom: 24 }}>
            {items.map((item, idx) => {
              const vc = item.verdict ? VERDICT_COLOR[item.verdict] : null;
              const isProc = item.status === "processing";
              const isDone = item.status === "done";
              const isErr  = item.status === "error";
              return (
                <div
                  key={item.id}
                  style={{
                    padding: "14px 16px",
                    background: isProc
                      ? "rgba(0,217,245,0.04)"
                      : isDone
                      ? "rgba(255,255,255,0.025)"
                      : "rgba(255,255,255,0.02)",
                    border: `1px solid ${isProc ? "rgba(0,217,245,0.22)" : isDone && vc ? `${vc}22` : "rgba(255,255,255,0.07)"}`,
                    borderRadius: 10,
                    display: "flex", gap: 12, alignItems: "center",
                    animation: `dt-fade-up 0.35s ease ${idx * 40}ms both`,
                    transition: "all 0.22s",
                  }}
                >
                  {/* Thumbnail or icon */}
                  <div style={{
                    width: 44, height: 44, borderRadius: 8, flexShrink: 0,
                    background: item.thumbnail ? "transparent" : "rgba(255,255,255,0.05)",
                    border: "1px solid rgba(255,255,255,0.1)",
                    overflow: "hidden",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    fontSize: 20,
                  }}>
                    {item.thumbnail
                      ? <img src={item.thumbnail} alt="" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
                      : (item.name.match(/\.(mp4|mov|avi)$/i) ? "🎬" : "🖼")}
                  </div>

                  {/* File info + progress */}
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{
                      fontSize: 13, fontWeight: 600, color: "#fff",
                      whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", marginBottom: 2,
                    }}>
                      {item.name}
                    </div>
                    <div style={{ fontSize: 11, color: "rgba(255,255,255,0.25)" }}>{item.size}</div>
                    {isProc && <ProgressBar value={item.progress} />}
                    {isErr && (
                      <div style={{ fontSize: 11, color: "#ff4d6d", marginTop: 4 }}>
                        {item.error?.slice(0, 60)}
                      </div>
                    )}
                  </div>

                  {/* Verdict / status */}
                  <div style={{ flexShrink: 0, textAlign: "right", minWidth: 80 }}>
                    {isDone && vc ? (
                      <>
                        <div style={{
                          display: "inline-block", padding: "2px 9px", borderRadius: 5, marginBottom: 3,
                          background: `${vc}14`, border: `1px solid ${vc}35`,
                          fontSize: 10, fontWeight: 700, color: vc,
                          fontFamily: "'Space Mono', monospace",
                        }}>
                          {item.verdict}
                        </div>
                        <div style={{ fontSize: 10, color: "rgba(255,255,255,0.3)", fontFamily: "'Space Mono', monospace" }}>
                          {item.confidence}% · {item.overall_score}/100
                        </div>
                      </>
                    ) : isProc ? (
                      <span style={{
                        fontSize: 10, color: "#00d9f5",
                        fontFamily: "'Space Mono', monospace",
                        animation: "dt-pulse 1.2s ease-in-out infinite",
                      }}>ANALYZING…</span>
                    ) : isErr ? (
                      <span style={{ fontSize: 10, color: "#ff4d6d", fontFamily: "'Space Mono', monospace" }}>ERROR</span>
                    ) : (
                      <span style={{ fontSize: 10, color: "rgba(255,255,255,0.25)", fontFamily: "'Space Mono', monospace" }}>QUEUED</span>
                    )}
                  </div>

                  {/* Remove button */}
                  {!isRunning && (
                    <button
                      onClick={() => setItems(prev => prev.filter(i => i.id !== item.id))}
                      style={{
                        width: 24, height: 24, borderRadius: 6, flexShrink: 0,
                        background: "rgba(255,255,255,0.04)",
                        border: "1px solid rgba(255,255,255,0.08)",
                        color: "rgba(255,255,255,0.3)",
                        cursor: "pointer", fontSize: 12, lineHeight: 1,
                        display: "flex", alignItems: "center", justifyContent: "center",
                        transition: "all 0.15s",
                      }}
                      onMouseEnter={e => { e.currentTarget.style.background = "rgba(255,77,109,0.12)"; e.currentTarget.style.color = "#ff4d6d"; }}
                      onMouseLeave={e => { e.currentTarget.style.background = "rgba(255,255,255,0.04)"; e.currentTarget.style.color = "rgba(255,255,255,0.3)"; }}
                    >✕</button>
                  )}
                </div>
              );
            })}
          </div>
        </>
      )}

      {/* Empty state */}
      {items.length === 0 && (
        <div style={{ textAlign: "center", padding: "48px 0", color: "rgba(255,255,255,0.2)", fontSize: 14 }}>
          <div style={{ fontSize: 32, marginBottom: 12, opacity: 0.4 }}>📁</div>
          Add files above to begin batch analysis
        </div>
      )}
    </div>
  );
}
