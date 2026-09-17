// ApiIntegration.jsx — API reference & live Try-it console
import { useState, useRef } from "react";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";

const ENDPOINTS = [
  {
    method: "POST",
    path: "/api/detect",
    color: "#00f5a0",
    desc: "Analyze an image or video file for deepfake artifacts using the forensic AI engine.",
    params: [
      { name: "file", type: "File (multipart/form-data)", required: true, desc: "Image or video — JPEG, PNG, WebP, MP4, MOV, AVI. Max 50 MB." },
    ],
    response: `{\n  "result": {\n    "verdict": "DEEPFAKE",\n    "confidence": 94,\n    "overall_score": 89,\n    "risk_level": "HIGH",\n    "forensic_summary": "...",\n    "analysis": { ... },\n    "xai_highlights": [ ... ]\n  }\n}`,
  },
  {
    method: "GET",
    path: "/api/health",
    color: "#00d9f5",
    desc: "Check backend status and which detection engine is currently active.",
    params: [],
    response: `{\n  "status": "ok",\n  "model_loaded": true,\n  "openrouter_model": "...",\n  "version": "4.0"\n}`,
  },
];

const CODE_SAMPLES = {
  python: `import requests

url = "${BACKEND_URL}/api/detect"

with open("image.jpg", "rb") as f:
    response = requests.post(
        url,
        files={"file": ("image.jpg", f, "image/jpeg")}
    )

data = response.json()
result = data["result"]
print(f"Verdict:    {result['verdict']}")
print(f"Confidence: {result['confidence']}%")
print(f"Fake Score: {result['overall_score']} / 100")`,

  javascript: `const formData = new FormData();
formData.append("file", fileInput.files[0]);

const res = await fetch("${BACKEND_URL}/api/detect", {
  method: "POST",
  body: formData,
});

const { result } = await res.json();
console.log("Verdict:   ", result.verdict);
console.log("Confidence:", result.confidence + "%");
console.log("Fake Score:", result.overall_score + "/100");`,

  curl: `# Analyze a file
curl -X POST ${BACKEND_URL}/api/detect \\
  -F "file=@image.jpg" \\
  -H "Accept: application/json" | jq .result

# Check service health
curl ${BACKEND_URL}/api/health | jq .`,
};

const RATE_LIMITS = [
  { label: "Requests / minute",  value: "60" },
  { label: "Max file size",      value: "50 MB" },
  { label: "Timeout",            value: "60 seconds" },
  { label: "Supported formats",  value: "JPEG, PNG, WebP, MP4, MOV, AVI" },
];

const MODEL_INFO = [
  { label: "Primary model", value: "EfficientNet-B4" },
  { label: "Fallback",      value: "OpenRouter API" },
  { label: "XAI method",    value: "GRAD-CAM + SHAP" },
  { label: "Forensic axes", value: "6 dimensions" },
];

function SectionLabel({ children }) {
  return (
    <div style={{
      fontSize: 10, fontFamily: "'Space Mono', monospace",
      textTransform: "uppercase", letterSpacing: 1.8,
      color: "rgba(255,255,255,0.28)", marginBottom: 14,
    }}>{children}</div>
  );
}

export default function ApiIntegration() {
  const [codeTab, setCodeTab]       = useState("python");
  const [copied, setCopied]         = useState(false);

  // Try-it state
  const [tryMode, setTryMode]       = useState("file");   // "file" | "url"
  const [tryFile, setTryFile]       = useState(null);
  const [tryUrl, setTryUrl]         = useState("");
  const [tryLoading, setTryLoading] = useState(false);
  const [tryResult, setTryResult]   = useState(null);
  const [tryError, setTryError]     = useState(null);
  const [tryStep, setTryStep]       = useState(null);
  const fileInputRef                = useRef();

  const copyCode = () => {
    navigator.clipboard.writeText(CODE_SAMPLES[codeTab]).catch(() => {});
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  const runTryIt = async () => {
    setTryLoading(true);
    setTryResult(null);
    setTryError(null);

    try {
      let file;

      if (tryMode === "file") {
        if (!tryFile) throw new Error("Please select a file to analyze.");
        file = tryFile;
        setTryStep("Uploading file…");
      } else {
        if (!tryUrl.trim()) throw new Error("Please enter an image URL.");
        setTryStep("Fetching image from URL…");
        const res = await fetch(tryUrl, { signal: AbortSignal.timeout(15000) });
        if (!res.ok) throw new Error(`Could not fetch URL: HTTP ${res.status}`);
        const blob = await res.blob();
        const ext  = tryUrl.split("?")[0].split(".").pop()?.toLowerCase() || "jpg";
        const type = blob.type || (ext === "png" ? "image/png" : "image/jpeg");
        file = new File([blob], `image.${ext}`, { type });
      }

      setTryStep("Sending to /api/detect…");
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(`${BACKEND_URL}/api/detect`, {
        method: "POST",
        body: formData,
        signal: AbortSignal.timeout(60000),
      });

      setTryStep("Processing response…");
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data?.detail || `Server error ${res.status}`);
      }

      setTryResult(data);
    } catch (err) {
      setTryError(err.message || "Request failed.");
    } finally {
      setTryLoading(false);
      setTryStep(null);
    }
  };

  const verdictColor = tryResult?.result?.verdict
    ? (tryResult.result.verdict === "DEEPFAKE" ? "#ff4d6d" : tryResult.result.verdict === "AUTHENTIC" ? "#00f5a0" : "#ffd166")
    : null;

  return (
    <div style={{ maxWidth: 920, margin: "0 auto" }}>

      {/* Header */}
      <div style={{ marginBottom: 28 }}>
        <h1 style={{ fontSize: 22, fontWeight: 700, margin: "0 0 5px", color: "#fff", letterSpacing: -0.4 }}>API Integration</h1>
        <p style={{ fontSize: 13, color: "rgba(255,255,255,0.38)", margin: 0 }}>
          Connect DeepTrace to your applications via REST — or try the API right here
        </p>
      </div>

      {/* Base URL card */}
      <div style={{
        padding: "18px 22px",
        background: "rgba(255,255,255,0.03)",
        border: "1px solid rgba(255,255,255,0.08)",
        borderRadius: 12, marginBottom: 22,
        display: "flex", alignItems: "center", justifyContent: "space-between",
        flexWrap: "wrap", gap: 14,
      }}>
        <div>
          <div style={{ fontSize: 10, fontFamily: "'Space Mono', monospace", textTransform: "uppercase", letterSpacing: 1.5, color: "rgba(255,255,255,0.28)", marginBottom: 9 }}>
            Base URL
          </div>
          <code style={{
            fontSize: 13, fontFamily: "'Space Mono', monospace", color: "#00f5a0",
            background: "rgba(0,245,160,0.08)", padding: "6px 14px", borderRadius: 7,
            border: "1px solid rgba(0,245,160,0.2)",
          }}>
            {BACKEND_URL}
          </code>
        </div>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          <div style={{
            padding: "6px 14px", background: "rgba(0,245,160,0.08)",
            border: "1px solid rgba(0,245,160,0.22)", borderRadius: 8,
            display: "flex", alignItems: "center", gap: 8,
          }}>
            <span style={{ width: 7, height: 7, borderRadius: "50%", background: "#00f5a0", boxShadow: "0 0 8px #00f5a0", animation: "dt-pulse 2s ease-in-out infinite" }} />
            <span style={{ fontSize: 12, fontFamily: "'Space Mono', monospace", color: "#00f5a0" }}>REST API</span>
          </div>
          <div style={{
            padding: "6px 14px", background: "rgba(255,255,255,0.04)",
            border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8,
          }}>
            <span style={{ fontSize: 11, fontFamily: "'Space Mono', monospace", color: "rgba(255,255,255,0.4)" }}>JSON · multipart/form-data</span>
          </div>
        </div>
      </div>

      {/* Endpoints */}
      <div style={{ marginBottom: 24 }}>
        <SectionLabel>Endpoints</SectionLabel>
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {ENDPOINTS.map(ep => (
            <div key={ep.path} style={{
              padding: "18px 20px",
              background: "rgba(255,255,255,0.025)",
              border: "1px solid rgba(255,255,255,0.08)",
              borderRadius: 12,
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 10 }}>
                <span style={{
                  padding: "3px 10px", borderRadius: 5,
                  background: `${ep.color}14`, border: `1px solid ${ep.color}33`,
                  fontSize: 10, fontWeight: 700, color: ep.color,
                  fontFamily: "'Space Mono', monospace", letterSpacing: 1, flexShrink: 0,
                }}>{ep.method}</span>
                <code style={{ fontSize: 14, fontFamily: "'Space Mono', monospace", color: "#fff" }}>{ep.path}</code>
              </div>
              <p style={{ fontSize: 13, color: "rgba(255,255,255,0.45)", margin: "0 0 14px", lineHeight: 1.65 }}>{ep.desc}</p>
              {ep.params.length > 0 && (
                <div style={{ marginBottom: 14 }}>
                  <div style={{ fontSize: 10, color: "rgba(255,255,255,0.22)", fontFamily: "'Space Mono', monospace", letterSpacing: 1.2, textTransform: "uppercase", marginBottom: 8 }}>
                    Body Parameters
                  </div>
                  {ep.params.map(p => (
                    <div key={p.name} style={{
                      display: "flex", gap: 12, alignItems: "flex-start",
                      padding: "8px 12px", background: "rgba(255,255,255,0.03)",
                      borderRadius: 7, flexWrap: "wrap",
                    }}>
                      <code style={{ fontSize: 12, fontFamily: "'Space Mono', monospace", color: ep.color }}>{p.name}</code>
                      <span style={{ fontSize: 11, color: "rgba(255,255,255,0.22)", fontFamily: "'Space Mono', monospace" }}>{p.type}</span>
                      {p.required && (
                        <span style={{ fontSize: 10, color: "#ff4d6d", fontFamily: "'Space Mono', monospace", padding: "1px 6px", border: "1px solid rgba(255,77,109,0.3)", borderRadius: 4 }}>required</span>
                      )}
                      <span style={{ fontSize: 12, color: "rgba(255,255,255,0.38)", flex: 1 }}>{p.desc}</span>
                    </div>
                  ))}
                </div>
              )}
              <div>
                <div style={{ fontSize: 10, color: "rgba(255,255,255,0.22)", fontFamily: "'Space Mono', monospace", letterSpacing: 1.2, textTransform: "uppercase", marginBottom: 8 }}>
                  Response · 200 OK
                </div>
                <pre style={{
                  margin: 0, padding: "12px 14px", background: "rgba(0,0,0,0.28)",
                  borderRadius: 7, fontSize: 11.5, fontFamily: "'Space Mono', monospace",
                  color: "rgba(255,255,255,0.48)", border: "1px solid rgba(255,255,255,0.06)",
                  overflowX: "auto", lineHeight: 1.75,
                }}>
                  {ep.response}
                </pre>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Code samples */}
      <div style={{
        background: "rgba(255,255,255,0.025)",
        border: "1px solid rgba(255,255,255,0.08)",
        borderRadius: 12, overflow: "hidden", marginBottom: 22,
      }}>
        <div style={{
          display: "flex", alignItems: "center", justifyContent: "space-between",
          padding: "10px 18px", borderBottom: "1px solid rgba(255,255,255,0.07)",
          background: "rgba(255,255,255,0.02)",
        }}>
          <div style={{ display: "flex", gap: 4 }}>
            {Object.keys(CODE_SAMPLES).map(lang => (
              <button key={lang} onClick={() => setCodeTab(lang)} style={{
                padding: "5px 14px", borderRadius: 6, border: "none",
                background: codeTab === lang ? "rgba(0,245,160,0.1)" : "transparent",
                color: codeTab === lang ? "#00f5a0" : "rgba(255,255,255,0.38)",
                fontSize: 12, fontWeight: codeTab === lang ? 600 : 400,
                cursor: "pointer", fontFamily: "'IBM Plex Sans', system-ui, sans-serif",
                textTransform: "capitalize", transition: "all 0.14s",
              }}>{lang}</button>
            ))}
          </div>
          <button onClick={copyCode} style={{
            padding: "5px 14px", borderRadius: 6,
            background: copied ? "rgba(0,245,160,0.1)" : "rgba(255,255,255,0.04)",
            border: `1px solid ${copied ? "rgba(0,245,160,0.28)" : "rgba(255,255,255,0.1)"}`,
            color: copied ? "#00f5a0" : "rgba(255,255,255,0.42)",
            fontSize: 11, cursor: "pointer", fontFamily: "'Space Mono', monospace",
            transition: "all 0.15s",
          }}>
            {copied ? "✓ Copied" : "Copy"}
          </button>
        </div>
        <pre style={{
          margin: 0, padding: "20px 22px",
          fontSize: 12.5, fontFamily: "'Space Mono', monospace",
          color: "rgba(255,255,255,0.72)", lineHeight: 1.85, overflowX: "auto",
          background: "transparent",
        }}>
          {CODE_SAMPLES[codeTab]}
        </pre>
      </div>

      {/* ── Try-it Console ── */}
      <div style={{
        background: "rgba(0,245,160,0.03)",
        border: "1px solid rgba(0,245,160,0.14)",
        borderRadius: 14, overflow: "hidden", marginBottom: 22,
      }}>
        {/* Header */}
        <div style={{
          padding: "14px 20px", borderBottom: "1px solid rgba(0,245,160,0.1)",
          background: "rgba(0,245,160,0.05)",
          display: "flex", alignItems: "center", gap: 10,
        }}>
          <span style={{
            width: 8, height: 8, borderRadius: "50%", background: "#00f5a0",
            boxShadow: "0 0 8px #00f5a0", animation: "dt-pulse 2s ease-in-out infinite",
          }} />
          <span style={{ fontSize: 13, fontWeight: 600, color: "#00f5a0" }}>Try It Live</span>
          <span style={{ fontSize: 12, color: "rgba(255,255,255,0.3)", marginLeft: 4 }}>
            — Hit the real /api/detect endpoint from your browser
          </span>
        </div>

        <div style={{ padding: "20px 22px" }}>
          {/* Mode toggle */}
          <div style={{ display: "flex", gap: 6, marginBottom: 16 }}>
            {["file", "url"].map(mode => (
              <button key={mode} onClick={() => { setTryMode(mode); setTryResult(null); setTryError(null); }} style={{
                padding: "6px 16px", borderRadius: 8,
                background: tryMode === mode ? "rgba(0,245,160,0.1)" : "rgba(255,255,255,0.04)",
                border: `1px solid ${tryMode === mode ? "rgba(0,245,160,0.3)" : "rgba(255,255,255,0.1)"}`,
                color: tryMode === mode ? "#00f5a0" : "rgba(255,255,255,0.38)",
                fontSize: 12, fontWeight: tryMode === mode ? 600 : 400,
                cursor: "pointer", fontFamily: "'IBM Plex Sans', system-ui, sans-serif",
                textTransform: "capitalize", transition: "all 0.15s",
              }}>
                {mode === "file" ? "📁 Upload File" : "🔗 Image URL"}
              </button>
            ))}
          </div>

          {/* Input area */}
          {tryMode === "file" ? (
            <div
              onClick={() => fileInputRef.current?.click()}
              style={{
                padding: "20px", marginBottom: 14,
                border: `1.5px dashed ${tryFile ? "rgba(0,245,160,0.4)" : "rgba(255,255,255,0.12)"}`,
                borderRadius: 10, textAlign: "center", cursor: "pointer",
                background: tryFile ? "rgba(0,245,160,0.05)" : "rgba(255,255,255,0.02)",
                transition: "all 0.2s",
              }}
              onMouseEnter={e => { e.currentTarget.style.borderColor = "rgba(0,245,160,0.4)"; }}
              onMouseLeave={e => { e.currentTarget.style.borderColor = tryFile ? "rgba(0,245,160,0.4)" : "rgba(255,255,255,0.12)"; }}
            >
              <input
                ref={fileInputRef} type="file"
                accept="image/*,video/mp4,video/quicktime"
                style={{ display: "none" }}
                onChange={e => { setTryFile(e.target.files[0] || null); setTryResult(null); setTryError(null); e.target.value = ""; }}
              />
              {tryFile ? (
                <div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: "#00f5a0", marginBottom: 3 }}>{tryFile.name}</div>
                  <div style={{ fontSize: 11, color: "rgba(255,255,255,0.3)" }}>
                    {(tryFile.size / (1024 * 1024)).toFixed(2)} MB · Click to change
                  </div>
                </div>
              ) : (
                <div>
                  <div style={{ fontSize: 20, marginBottom: 8 }}>📤</div>
                  <div style={{ fontSize: 13, color: "rgba(255,255,255,0.45)" }}>Click to select an image or video</div>
                  <div style={{ fontSize: 11, color: "rgba(255,255,255,0.22)", marginTop: 4 }}>JPEG · PNG · WebP · MP4</div>
                </div>
              )}
            </div>
          ) : (
            <input
              value={tryUrl}
              onChange={e => { setTryUrl(e.target.value); setTryResult(null); setTryError(null); }}
              placeholder="https://example.com/image.jpg"
              style={{
                width: "100%", padding: "10px 14px", marginBottom: 14,
                background: "rgba(255,255,255,0.05)",
                border: "1px solid rgba(255,255,255,0.1)",
                borderRadius: 8, color: "#fff", fontSize: 13,
                fontFamily: "'Space Mono', monospace",
                outline: "none", transition: "border-color 0.18s",
              }}
              onFocus={e => { e.currentTarget.style.borderColor = "rgba(0,245,160,0.4)"; }}
              onBlur={e => { e.currentTarget.style.borderColor = "rgba(255,255,255,0.1)"; }}
            />
          )}

          {/* Analyze button */}
          <button
            onClick={runTryIt}
            disabled={tryLoading}
            style={{
              padding: "10px 24px", borderRadius: 8,
              background: tryLoading ? "rgba(255,255,255,0.07)" : "linear-gradient(135deg,#00f5a0,#00d9f5)",
              border: "none",
              color: tryLoading ? "rgba(255,255,255,0.4)" : "#080b12",
              fontWeight: 700, fontSize: 13,
              cursor: tryLoading ? "not-allowed" : "pointer",
              fontFamily: "'IBM Plex Sans', system-ui, sans-serif",
              boxShadow: tryLoading ? "none" : "0 4px 18px rgba(0,245,160,0.25)",
              transition: "all 0.2s",
              display: "flex", alignItems: "center", gap: 8,
            }}
          >
            {tryLoading ? (
              <>
                <span style={{
                  width: 14, height: 14, border: "2px solid rgba(255,255,255,0.2)",
                  borderTopColor: "#fff", borderRadius: "50%",
                  animation: "dt-spin 0.75s linear infinite", display: "inline-block",
                }} />
                {tryStep || "Analyzing…"}
              </>
            ) : "▶ Analyze"}
          </button>

          {/* Error */}
          {tryError && (
            <div style={{
              marginTop: 14, padding: "12px 14px",
              background: "rgba(255,77,109,0.08)", border: "1px solid rgba(255,77,109,0.25)",
              borderRadius: 8, color: "#ff4d6d", fontSize: 13,
            }}>
              {tryError}
            </div>
          )}

          {/* Result */}
          {tryResult && (
            <div style={{ marginTop: 16 }}>
              {/* Quick verdict */}
              {tryResult.result && (
                <div style={{
                  padding: "14px 18px", marginBottom: 12,
                  background: `${verdictColor}0a`,
                  border: `1px solid ${verdictColor}30`,
                  borderRadius: 10,
                  display: "flex", gap: 20, alignItems: "center", flexWrap: "wrap",
                }}>
                  <div>
                    <div style={{ fontSize: 10, color: "rgba(255,255,255,0.3)", fontFamily: "'Space Mono', monospace", textTransform: "uppercase", letterSpacing: 1.5, marginBottom: 4 }}>Verdict</div>
                    <div style={{ fontSize: 20, fontWeight: 800, color: verdictColor, fontFamily: "'Space Mono', monospace" }}>
                      {tryResult.result.verdict}
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: 10, color: "rgba(255,255,255,0.3)", fontFamily: "'Space Mono', monospace", textTransform: "uppercase", letterSpacing: 1.5, marginBottom: 4 }}>Confidence</div>
                    <div style={{ fontSize: 20, fontWeight: 700, color: "rgba(255,255,255,0.8)", fontFamily: "'Space Mono', monospace" }}>
                      {tryResult.result.confidence}%
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: 10, color: "rgba(255,255,255,0.3)", fontFamily: "'Space Mono', monospace", textTransform: "uppercase", letterSpacing: 1.5, marginBottom: 4 }}>Fake Score</div>
                    <div style={{ fontSize: 20, fontWeight: 700, color: "rgba(255,255,255,0.8)", fontFamily: "'Space Mono', monospace" }}>
                      {tryResult.result.overall_score}/100
                    </div>
                  </div>
                  <div style={{ flex: 1, fontSize: 12, color: "rgba(255,255,255,0.4)", lineHeight: 1.6 }}>
                    {tryResult.result.forensic_summary?.slice(0, 160)}…
                  </div>
                </div>
              )}

              {/* Raw JSON */}
              <div style={{ fontSize: 10, color: "rgba(255,255,255,0.22)", fontFamily: "'Space Mono', monospace", textTransform: "uppercase", letterSpacing: 1.2, marginBottom: 8 }}>
                Raw JSON Response
              </div>
              <pre style={{
                margin: 0, padding: "14px 16px",
                background: "rgba(0,0,0,0.35)", borderRadius: 8,
                fontSize: 11, fontFamily: "'Space Mono', monospace",
                color: "rgba(255,255,255,0.55)", lineHeight: 1.8,
                overflowX: "auto", overflowY: "auto", maxHeight: 320,
                border: "1px solid rgba(255,255,255,0.07)",
              }}>
                {JSON.stringify(tryResult, null, 2)}
              </pre>
            </div>
          )}
        </div>
      </div>

      {/* Info grid */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        <div style={{
          padding: "16px 20px", background: "rgba(255,255,255,0.025)",
          border: "1px solid rgba(255,255,255,0.08)", borderRadius: 12,
        }}>
          <SectionLabel>Rate Limits & Constraints</SectionLabel>
          {RATE_LIMITS.map((item, i, arr) => (
            <div key={item.label} style={{
              display: "flex", justifyContent: "space-between", alignItems: "flex-start",
              padding: "8px 0", borderBottom: i < arr.length - 1 ? "1px solid rgba(255,255,255,0.05)" : "none", gap: 12,
            }}>
              <span style={{ fontSize: 12, color: "rgba(255,255,255,0.35)", flexShrink: 0 }}>{item.label}</span>
              <span style={{ fontSize: 12, fontFamily: "'Space Mono', monospace", color: "rgba(255,255,255,0.7)", textAlign: "right" }}>{item.value}</span>
            </div>
          ))}
        </div>
        <div style={{
          padding: "16px 20px", background: "rgba(255,255,255,0.025)",
          border: "1px solid rgba(255,255,255,0.08)", borderRadius: 12,
        }}>
          <SectionLabel>Detection Engine</SectionLabel>
          {MODEL_INFO.map((item, i, arr) => (
            <div key={item.label} style={{
              display: "flex", justifyContent: "space-between", alignItems: "center",
              padding: "8px 0", borderBottom: i < arr.length - 1 ? "1px solid rgba(255,255,255,0.05)" : "none",
            }}>
              <span style={{ fontSize: 12, color: "rgba(255,255,255,0.35)" }}>{item.label}</span>
              <span style={{ fontSize: 12, fontFamily: "'Space Mono', monospace", color: "#00f5a0" }}>{item.value}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
