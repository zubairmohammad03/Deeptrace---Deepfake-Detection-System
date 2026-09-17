// App.jsx — DeepTrace AI v4
import { useState, useCallback, useEffect } from "react";
import UploadZone       from "./components/UploadZone.jsx";
import ScanningView     from "./components/ScanningView.jsx";
import ResultsDashboard from "./components/ResultsDashboard.jsx";
import Sidebar          from "./components/Sidebar.jsx";
import BatchAnalysis    from "./components/BatchAnalysis.jsx";
import ScanHistory      from "./components/ScanHistory.jsx";
import LiveDetection    from "./components/LiveDetection.jsx";
import ApiIntegration   from "./components/ApiIntegration.jsx";
import { analyzeMedia, extractVideoFrame, fileToBase64, checkBackendHealth } from "./services/claudeApi.js";
import { HISTORY_KEY } from "./constants.js";

const SECTION_TITLES = {
  analysis: "Single Analysis",
  batch:    "Batch Analysis",
  history:  "Scan History",
  live:     "Live Detection",
  api:      "API Integration",
};


function generateThumbnail(src) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const MAX = 200;
      const ratio = Math.min(MAX / img.width, MAX / img.height, 1);
      const canvas = document.createElement("canvas");
      canvas.width  = Math.round(img.width  * ratio);
      canvas.height = Math.round(img.height * ratio);
      canvas.getContext("2d").drawImage(img, 0, 0, canvas.width, canvas.height);
      resolve(canvas.toDataURL("image/jpeg", 0.8));
    };
    img.onerror = () => resolve(null);
    img.src = src;
  });
}

function saveScanToHistory(file, result, thumbnail) {
  try {
    const prev = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
    const entry = {
      id: Date.now(),
      timestamp: new Date().toISOString(),
      filename: file.name,
      filesize: file.size,
      verdict: result.verdict,
      confidence: result.confidence,
      overall_score: result.overall_score,
      risk_level: result.risk_level,
      thumbnail,
      result,
    };
    localStorage.setItem(HISTORY_KEY, JSON.stringify([entry, ...prev].slice(0, 50)));
  } catch { /* storage full or unavailable */ }
}

export default function App() {
  const [activeSection, setActiveSection] = useState("analysis");

  // ── Existing detection state (untouched) ──────────────────────────────────
  const [stage, setStage]               = useState("idle");
  const [file, setFile]                 = useState(null);
  const [preview, setPreview]           = useState(null);
  const [result, setResult]             = useState(null);
  const [error, setError]               = useState(null);
  const [scanProgress, setScanProgress] = useState(0);
  const [scanStep, setScanStep]         = useState(0);
  const [backendHealth, setBackendHealth] = useState(null);
  // ──────────────────────────────────────────────────────────────────────────

  useEffect(() => {
    checkBackendHealth().then(h => setBackendHealth(h || false));
  }, []);

  const handleFile = useCallback(async (f) => {
    setFile(f);
    setError(null);
    setResult(null);
    const objectUrl = URL.createObjectURL(f);
    setPreview(objectUrl);
    setStage("scanning");
    setScanProgress(0);
    setScanStep(0);

    const timer = setInterval(() => {
      setScanProgress(p => { if (p >= 90) { clearInterval(timer); return 90; } return p + Math.random() * 7 + 3; });
      setScanStep(s => Math.min(s + 1, 9));
    }, 750);

    try {
      let base64, mType;
      if (f.type.startsWith("video/")) {
        base64 = await extractVideoFrame(f);
        mType  = "image/jpeg";
        // Swap video blob URL for the extracted frame so <img> can render it
        URL.revokeObjectURL(objectUrl);
        setPreview(`data:image/jpeg;base64,${base64}`);
      } else {
        base64 = await fileToBase64(f);
        mType  = f.type;
      }

      const analysis = await analyzeMedia(base64, mType, f);

      clearInterval(timer);
      setScanProgress(100);
      setScanStep(9);

      // For video use the extracted frame URL; for image use the blob URL
      const thumbSrc = f.type.startsWith("video/")
        ? `data:image/jpeg;base64,${base64}`
        : objectUrl;
      const thumb = await generateThumbnail(thumbSrc);
      saveScanToHistory(f, analysis, thumb);

      setTimeout(() => { setResult(analysis); setStage("results"); }, 600);
    } catch (err) {
      clearInterval(timer);
      setError(err.message || "Analysis failed. Please try again.");
      setStage("idle");
    }
  }, []);

  const reset = useCallback(() => {
    if (preview) URL.revokeObjectURL(preview);
    setStage("idle"); setFile(null); setPreview(null);
    setResult(null);  setError(null); setScanProgress(0); setScanStep(0);
  }, [preview]);

  // Re-view a past result from ScanHistory
  const onViewResult = useCallback((entry) => {
    if (preview) URL.revokeObjectURL(preview);
    setResult(entry.result);
    setPreview(entry.thumbnail);
    setFile(null);
    setStage("results");
    setActiveSection("analysis");
  }, [preview]);
  // ──────────────────────────────────────────────────────────────────────────

  const engineOnline = backendHealth !== null && backendHealth !== false;
  const engineLabel  = backendHealth === null  ? "Connecting…"
    : backendHealth === false                  ? "Backend Offline"
    : "Backend Online";
  const engineSub    = engineOnline
    ? (backendHealth?.model_loaded ? "EfficientNet-B4" : `OpenRouter`)
    : null;
  const engineColor  = backendHealth === false ? "#ff4d6d"
    : backendHealth === null ? "#ffd166" : "#00f5a0";

  return (
    <div style={{
      display: "flex",
      minHeight: "100vh",
      background: "#080b12",
      backgroundImage: `
        radial-gradient(ellipse at 18% 14%, #0d1f3c 0%, transparent 52%),
        radial-gradient(ellipse at 82% 86%, #180a28 0%, transparent 52%)
      `,
      color: "#fff",
      fontFamily: "'IBM Plex Sans', system-ui, sans-serif",
      overflowX: "hidden",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');
        *, *::before, *::after { box-sizing: border-box; }
        @keyframes fadeUp      { from{opacity:0;transform:translateY(18px)} to{opacity:1;transform:translateY(0)} }
        @keyframes sectionEnter{ from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
        @keyframes float       { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-9px)} }
        @keyframes spin        { to{transform:rotate(360deg)} }
        @keyframes scanLine    { 0%{top:-2px} 100%{top:102%} }
        @keyframes gridPulse   { 0%,100%{opacity:0.022} 50%{opacity:0.055} }
        @keyframes pulse       { 0%,100%{opacity:0.45} 50%{opacity:1} }
        @keyframes dt-fade-up  { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
        @keyframes dt-slide-in { from{opacity:0;transform:translateX(-10px)} to{opacity:1;transform:translateX(0)} }
        @keyframes dt-pulse    { 0%,100%{opacity:0.5} 50%{opacity:1} }
        @keyframes dt-spin     { to{transform:rotate(360deg)} }
        @keyframes glow-pulse  { 0%,100%{box-shadow:0 0 10px rgba(0,245,160,0.3)} 50%{box-shadow:0 0 24px rgba(0,245,160,0.6)} }
        @keyframes shimmer     { 0%{background-position:-200% 0} 100%{background-position:200% 0} }
      `}</style>

      {/* Fixed grid background */}
      <div style={{
        position: "fixed", inset: 0, pointerEvents: "none", zIndex: 0,
        backgroundImage: `
          linear-gradient(rgba(0,245,160,0.028) 1px, transparent 1px),
          linear-gradient(90deg, rgba(0,245,160,0.028) 1px, transparent 1px)
        `,
        backgroundSize: "44px 44px",
        animation: "gridPulse 5s ease-in-out infinite",
      }} />

      {/* Sidebar */}
      <Sidebar
        activeSection={activeSection}
        onSection={setActiveSection}
        engineLabel={engineLabel}
        engineSub={engineSub}
        engineColor={engineColor}
      />

      {/* Main column */}
      <div style={{
        flex: 1, display: "flex", flexDirection: "column",
        minWidth: 0, position: "relative", zIndex: 1,
      }}>

        {/* Top bar */}
        <header style={{
          height: 58,
          display: "flex", alignItems: "center", justifyContent: "space-between",
          padding: "0 36px",
          borderBottom: "1px solid rgba(255,255,255,0.06)",
          background: "rgba(8,11,18,0.88)",
          backdropFilter: "blur(18px)",
          position: "sticky", top: 0, zIndex: 10,
          gap: 16,
        }}>
          <h2 style={{ fontSize: 15, fontWeight: 600, margin: 0, color: "#fff", flexShrink: 0, letterSpacing: -0.2 }}>
            {SECTION_TITLES[activeSection]}
          </h2>
          <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
            {["GRAD-CAM XAI", "6 Forensic Dimensions", "Forensic Reports"].map(t => (
              <span key={t} style={{
                display: "inline-flex", alignItems: "center", gap: 5,
                padding: "4px 11px",
                background: "rgba(255,255,255,0.04)",
                border: "1px solid rgba(255,255,255,0.08)",
                borderRadius: 100, fontSize: 11,
                color: "rgba(255,255,255,0.35)",
                fontFamily: "'Space Mono', monospace",
                whiteSpace: "nowrap",
                letterSpacing: 0.3,
              }}>✦ {t}</span>
            ))}
          </div>
        </header>

        {/* Scrollable content */}
        <main style={{ flex: 1, overflowY: "auto", padding: "32px 40px 80px" }}>

          {/* Section enter animation keyed by section */}
          <div key={activeSection} style={{ animation: "sectionEnter 0.32s cubic-bezier(0.4,0,0.2,1)" }}>

            {/* ── Analysis section (detection flow) ── */}
            {activeSection === "analysis" && (
              <div style={{ maxWidth: 860, margin: "0 auto" }}>
                {stage === "idle" && (
                  <>
                    <div style={{
                      marginBottom: 28, padding: "20px 24px",
                      background: "rgba(255,255,255,0.025)",
                      border: "1px solid rgba(255,255,255,0.07)",
                      borderRadius: 14,
                      display: "flex", alignItems: "center", justifyContent: "space-between",
                      flexWrap: "wrap", gap: 16,
                    }}>
                      <p style={{ fontSize: 13, color: "rgba(255,255,255,0.42)", margin: 0, lineHeight: 1.7 }}>
                        Forensic-grade AI · EfficientNet-B4 with real GRAD-CAM heatmaps,<br />
                        6 analysis dimensions, and legally-defensible XAI explanations.
                      </p>
                      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                        {[{ icon: "🎯", label: "6 Dimensions" }, { icon: "🧠", label: "GRAD-CAM XAI" }, { icon: "📋", label: "PDF Reports" }].map(b => (
                          <div key={b.label} style={{
                            display: "flex", alignItems: "center", gap: 6,
                            padding: "6px 12px",
                            background: "rgba(0,245,160,0.06)",
                            border: "1px solid rgba(0,245,160,0.14)",
                            borderRadius: 8, fontSize: 12,
                            color: "rgba(255,255,255,0.55)",
                          }}>
                            <span>{b.icon}</span> {b.label}
                          </div>
                        ))}
                      </div>
                    </div>
                    <div style={{ animation: "fadeUp 0.6s ease 0.1s both" }}>
                      <UploadZone onFile={handleFile} error={error} />
                    </div>
                  </>
                )}
                {stage === "scanning" && (
                  <ScanningView preview={preview} progress={Math.min(scanProgress, 100)} stepIndex={scanStep} />
                )}
                {stage === "results" && result && (
                  <ResultsDashboard result={result} preview={preview} file={file} onReset={reset} />
                )}
              </div>
            )}

            {activeSection === "batch"   && <BatchAnalysis />}
            {activeSection === "history" && <ScanHistory onViewResult={onViewResult} />}
            {activeSection === "live"    && <LiveDetection />}
            {activeSection === "api"     && <ApiIntegration />}
          </div>
        </main>

        {/* Footer */}
        <footer style={{
          padding: "12px 36px",
          borderTop: "1px solid rgba(255,255,255,0.04)",
          background: "rgba(8,11,18,0.6)",
        }}>
          <p style={{
            fontSize: 10, margin: 0,
            color: "rgba(255,255,255,0.14)",
            fontFamily: "'Space Mono', monospace", letterSpacing: 1.2,
          }}>
            DEEPTRACE AI v4 · EFFICIENTNET-B4 + OPENROUTER · FORENSIC DEEPFAKE DETECTION
          </p>
        </footer>
      </div>
    </div>
  );
}
