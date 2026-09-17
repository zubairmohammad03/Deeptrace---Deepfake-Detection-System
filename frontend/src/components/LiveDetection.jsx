// LiveDetection.jsx — Real webcam live deepfake detection
import { useState, useRef, useEffect, useCallback } from "react";
import { analyzeMedia } from "../services/claudeApi.js";

const VERDICT_COLOR = { DEEPFAKE: "#ff4d6d", AUTHENTIC: "#00f5a0", SUSPICIOUS: "#ffd166" };

export default function LiveDetection() {
  const videoRef    = useRef(null);
  const canvasRef   = useRef(null);
  const streamRef   = useRef(null);
  const intervalRef = useRef(null);
  const analyzingRef = useRef(false);

  const [cameraOn, setCameraOn]     = useState(false);
  const [error, setError]           = useState(null);
  const [liveResult, setLiveResult] = useState(null);
  const [frameCount, setFrameCount] = useState(0);
  const [analyzing, setAnalyzing]   = useState(false);
  const [cameras, setCameras]       = useState([]);
  const [selectedCam, setSelectedCam] = useState("");
  const [threshold, setThreshold]   = useState(0.7);
  const [latency, setLatency]       = useState(null);
  const [fps, setFps]               = useState(null);

  // Enumerate cameras on mount
  useEffect(() => {
    navigator.mediaDevices?.enumerateDevices?.()
      .then(devices => {
        const cams = devices.filter(d => d.kind === "videoinput");
        setCameras(cams);
        if (cams.length) setSelectedCam(cams[0].deviceId || "");
      })
      .catch(() => {});
  }, []);

  const captureAndAnalyze = useCallback(async () => {
    if (analyzingRef.current) return;
    const video  = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2) return;

    analyzingRef.current = true;
    setAnalyzing(true);
    const t0 = Date.now();

    canvas.width  = video.videoWidth  || 640;
    canvas.height = video.videoHeight || 480;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);

    await new Promise(resolve => {
      canvas.toBlob(async (blob) => {
        if (!blob) { resolve(); return; }
        const file = new File([blob], "frame.jpg", { type: "image/jpeg" });
        try {
          const result = await analyzeMedia(null, "image/jpeg", file);
          setLiveResult(result);
          setFrameCount(c => c + 1);
          setLatency(Date.now() - t0);
        } catch {
          // Silent fail — keep detecting
        } finally {
          analyzingRef.current = false;
          setAnalyzing(false);
          resolve();
        }
      }, "image/jpeg", 0.85);
    });
  }, []);

  const startCamera = useCallback(async () => {
    setError(null);
    try {
      const constraints = {
        video: selectedCam
          ? { deviceId: { exact: selectedCam }, width: { ideal: 1280 }, height: { ideal: 720 } }
          : { width: { ideal: 1280 }, height: { ideal: 720 } },
      };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      // Re-enumerate to get real device labels (requires permission)
      const devices = await navigator.mediaDevices.enumerateDevices();
      const cams = devices.filter(d => d.kind === "videoinput");
      setCameras(cams);

      setCameraOn(true);
      setFrameCount(0);
      setLiveResult(null);

      // Estimate FPS from stream track settings
      const track = stream.getVideoTracks()[0];
      const settings = track?.getSettings?.();
      if (settings?.frameRate) setFps(Math.round(settings.frameRate));

      // Start analysis every 2.5s
      intervalRef.current = setInterval(captureAndAnalyze, 2500);
    } catch (err) {
      setError(
        err.name === "NotAllowedError"
          ? "Camera access denied. Please allow camera permissions in your browser."
          : err.name === "NotFoundError"
          ? "No camera found. Connect a camera and try again."
          : err.message || "Could not access camera."
      );
    }
  }, [selectedCam, captureAndAnalyze]);

  const stopCamera = useCallback(() => {
    clearInterval(intervalRef.current);
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraOn(false);
    setLiveResult(null);
    setFrameCount(0);
    setLatency(null);
    analyzingRef.current = false;
    setAnalyzing(false);
  }, []);

  // Cleanup on unmount
  useEffect(() => () => stopCamera(), [stopCamera]);

  const resultColor = liveResult ? (VERDICT_COLOR[liveResult.verdict] || "#fff") : "#00f5a0";
  const isFake = liveResult?.verdict === "DEEPFAKE";
  const isSusp = liveResult?.verdict === "SUSPICIOUS";

  const metrics = [
    { label: "Frames",    value: frameCount > 0 ? frameCount : "—",                    color: cameraOn ? "#00f5a0" : "rgba(255,255,255,0.25)" },
    { label: "Latency",   value: latency != null ? `${latency}ms` : "—",               color: cameraOn ? "rgba(255,255,255,0.65)" : "rgba(255,255,255,0.25)" },
    { label: "Camera FPS",value: fps != null ? `${fps}` : "—",                         color: cameraOn ? "rgba(255,255,255,0.65)" : "rgba(255,255,255,0.25)" },
    { label: "Confidence",value: liveResult ? `${liveResult.confidence}%` : "—",       color: liveResult ? resultColor : "rgba(255,255,255,0.25)" },
  ];

  return (
    <div style={{ maxWidth: 1020, margin: "0 auto" }}>

      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 22, fontWeight: 700, margin: "0 0 5px", color: "#fff", letterSpacing: -0.4 }}>Live Detection</h1>
        <p style={{ fontSize: 13, color: "rgba(255,255,255,0.38)", margin: 0 }}>
          Real-time deepfake detection via webcam — frame sampled every 2.5 seconds
        </p>
      </div>

      {/* Error banner */}
      {error && (
        <div style={{
          padding: "12px 16px", marginBottom: 20,
          background: "rgba(255,77,109,0.08)",
          border: "1px solid rgba(255,77,109,0.3)",
          borderRadius: 10, color: "#ff4d6d", fontSize: 13,
        }}>{error}</div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 288px", gap: 16, alignItems: "start" }}>

        {/* ── Camera feed ── */}
        <div style={{
          background: "rgba(255,255,255,0.025)",
          border: `1px solid ${cameraOn && liveResult ? `${resultColor}33` : "rgba(255,255,255,0.08)"}`,
          borderRadius: 14, overflow: "hidden",
          display: "flex", flexDirection: "column",
          transition: "border-color 0.4s",
        }}>
          {/* Header bar */}
          <div style={{
            padding: "11px 16px",
            display: "flex", alignItems: "center", justifyContent: "space-between",
            borderBottom: "1px solid rgba(255,255,255,0.06)",
            background: "rgba(255,255,255,0.02)",
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{
                width: 7, height: 7, borderRadius: "50%", flexShrink: 0,
                background: cameraOn ? "#ff4d6d" : "rgba(255,255,255,0.18)",
                boxShadow: cameraOn ? "0 0 10px #ff4d6d" : "none",
                animation: cameraOn ? "dt-pulse 1.1s ease-in-out infinite" : "none",
              }} />
              <span style={{
                fontSize: 11, fontFamily: "'Space Mono', monospace",
                color: cameraOn ? "#ff4d6d" : "rgba(255,255,255,0.28)",
                letterSpacing: 1.5,
              }}>
                {cameraOn ? "LIVE" : "STANDBY"}
              </span>
            </div>
            {cameraOn && analyzing && (
              <span style={{
                fontSize: 9, fontFamily: "'Space Mono', monospace",
                color: "#00d9f5", letterSpacing: 1,
                animation: "dt-pulse 0.8s ease-in-out infinite",
              }}>⟳ ANALYZING FRAME…</span>
            )}
            <span style={{ fontSize: 10, fontFamily: "'Space Mono', monospace", color: "rgba(255,255,255,0.22)" }}>
              {cameras.find(c => c.deviceId === selectedCam)?.label?.split("(")[0]?.trim() || "Camera"} · 1280×720
            </span>
          </div>

          {/* Video viewport */}
          <div style={{
            position: "relative", minHeight: 380,
            background: "#03050c",
            display: "flex", alignItems: "center", justifyContent: "center",
            overflow: "hidden",
          }}>
            {/* Grid overlay */}
            <div style={{
              position: "absolute", inset: 0, pointerEvents: "none",
              backgroundImage: `linear-gradient(rgba(0,245,160,0.018) 1px, transparent 1px), linear-gradient(90deg, rgba(0,245,160,0.018) 1px, transparent 1px)`,
              backgroundSize: "32px 32px",
            }} />

            {/* Actual video element */}
            <video
              ref={videoRef}
              muted playsInline
              style={{
                position: "absolute", inset: 0,
                width: "100%", height: "100%",
                objectFit: "cover",
                display: cameraOn ? "block" : "none",
              }}
            />

            {/* Hidden canvas for frame capture */}
            <canvas ref={canvasRef} style={{ display: "none" }} />

            {!cameraOn ? (
              /* Standby */
              <div style={{ textAlign: "center", zIndex: 1 }}>
                <div style={{
                  width: 68, height: 68, borderRadius: "50%",
                  border: "1.5px solid rgba(255,255,255,0.1)",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 28, margin: "0 auto 16px",
                  background: "rgba(255,255,255,0.04)",
                }}>📷</div>
                <div style={{ fontSize: 14, color: "rgba(255,255,255,0.38)", marginBottom: 6 }}>Camera not active</div>
                <div style={{ fontSize: 12, color: "rgba(255,255,255,0.2)" }}>
                  Click "Start Camera" to begin live detection
                </div>
              </div>
            ) : (
              /* Active overlays */
              <>
                {/* Gradient overlay */}
                <div style={{
                  position: "absolute", inset: 0, pointerEvents: "none",
                  background: "linear-gradient(180deg, rgba(0,0,0,0.3) 0%, rgba(0,0,0,0) 30%, rgba(0,0,0,0) 70%, rgba(0,0,0,0.4) 100%)",
                }} />

                {/* Scan line */}
                <div style={{
                  position: "absolute", left: 0, right: 0, height: 1.5,
                  background: "linear-gradient(90deg, transparent, #00f5a0 40%, #00d9f5 60%, transparent)",
                  opacity: 0.3, animation: "scanLine 3s linear infinite",
                  pointerEvents: "none",
                }} />

                {/* Live verdict overlay */}
                {liveResult && (
                  <div style={{
                    position: "absolute", bottom: 16, left: 16, right: 16,
                    display: "flex", alignItems: "center", justifyContent: "space-between",
                    gap: 10,
                    padding: "10px 14px",
                    background: "rgba(0,0,0,0.75)",
                    backdropFilter: "blur(8px)",
                    borderRadius: 10,
                    border: `1px solid ${resultColor}44`,
                    boxShadow: `0 0 20px ${resultColor}22`,
                    pointerEvents: "none",
                  }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                      <span style={{
                        width: 8, height: 8, borderRadius: "50%",
                        background: resultColor,
                        boxShadow: `0 0 10px ${resultColor}`,
                        flexShrink: 0,
                        animation: "dt-pulse 1.2s ease-in-out infinite",
                      }} />
                      <span style={{
                        fontFamily: "'Space Mono', monospace",
                        fontSize: 13, fontWeight: 700, color: resultColor,
                        letterSpacing: 1,
                      }}>
                        {liveResult.verdict}
                      </span>
                    </div>
                    <div style={{ display: "flex", gap: 16 }}>
                      <div style={{ textAlign: "center" }}>
                        <div style={{ fontSize: 15, fontWeight: 700, color: resultColor, fontFamily: "'Space Mono', monospace" }}>
                          {liveResult.confidence}%
                        </div>
                        <div style={{ fontSize: 8, color: "rgba(255,255,255,0.35)", fontFamily: "'Space Mono', monospace", textTransform: "uppercase" }}>conf</div>
                      </div>
                      <div style={{ textAlign: "center" }}>
                        <div style={{ fontSize: 15, fontWeight: 700,
                          color: liveResult.overall_score > 70 ? "#ff4d6d" : liveResult.overall_score > 40 ? "#ffd166" : "#00f5a0",
                          fontFamily: "'Space Mono', monospace" }}>
                          {liveResult.overall_score}%
                        </div>
                        <div style={{ fontSize: 8, color: "rgba(255,255,255,0.35)", fontFamily: "'Space Mono', monospace", textTransform: "uppercase" }}>fake</div>
                      </div>
                    </div>
                  </div>
                )}

                {/* REC indicator */}
                <div style={{
                  position: "absolute", top: 12, left: 14,
                  fontSize: 10, fontFamily: "'Space Mono', monospace", color: "#ff4d6d",
                  opacity: 0.85, letterSpacing: 1,
                  animation: "dt-pulse 1.1s ease-in-out infinite",
                }}>● REC</div>

                <div style={{
                  position: "absolute", top: 12, right: 14,
                  fontSize: 9, fontFamily: "'Space Mono', monospace",
                  color: "rgba(255,255,255,0.4)", letterSpacing: 1,
                }}>EfficientNet-B4 · Frame {frameCount}</div>
              </>
            )}
          </div>
        </div>

        {/* ── Controls panel ── */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>

          {/* Camera controls */}
          <div style={{
            padding: 18, borderRadius: 12,
            background: "rgba(255,255,255,0.025)",
            border: "1px solid rgba(255,255,255,0.08)",
          }}>
            <div style={{
              fontSize: 10, fontFamily: "'Space Mono', monospace",
              textTransform: "uppercase", letterSpacing: 1.8,
              color: "rgba(255,255,255,0.28)", marginBottom: 14,
            }}>Camera Controls</div>

            <button
              onClick={cameraOn ? stopCamera : startCamera}
              style={{
                width: "100%", padding: "11px", borderRadius: 8, marginBottom: 12,
                background: cameraOn
                  ? "rgba(255,77,109,0.1)"
                  : "linear-gradient(135deg,#00f5a0,#00d9f5)",
                border: cameraOn ? "1px solid rgba(255,77,109,0.3)" : "none",
                color: cameraOn ? "#ff4d6d" : "#080b12",
                fontWeight: 700, fontSize: 13, cursor: "pointer",
                fontFamily: "'IBM Plex Sans', system-ui, sans-serif",
                boxShadow: cameraOn ? "none" : "0 4px 18px rgba(0,245,160,0.28)",
                transition: "all 0.22s",
              }}
            >
              {cameraOn ? "⏹ Stop Camera" : "▶ Start Camera"}
            </button>

            <div style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 11, color: "rgba(255,255,255,0.28)", marginBottom: 6 }}>Camera Source</div>
              <select
                value={selectedCam}
                onChange={e => setSelectedCam(e.target.value)}
                disabled={cameraOn}
                style={{
                  width: "100%", padding: "8px 10px",
                  background: "rgba(255,255,255,0.05)",
                  border: "1px solid rgba(255,255,255,0.1)",
                  borderRadius: 7, color: "#fff", fontSize: 12,
                  fontFamily: "'IBM Plex Sans', system-ui, sans-serif",
                  cursor: cameraOn ? "not-allowed" : "pointer",
                  outline: "none", opacity: cameraOn ? 0.5 : 1,
                }}
              >
                {cameras.length > 0
                  ? cameras.map((c, i) => (
                    <option key={c.deviceId || i} value={c.deviceId}>
                      {c.label || `Camera ${i}`}
                    </option>
                  ))
                  : <option value="">Default Camera</option>
                }
              </select>
            </div>

            <div>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 7 }}>
                <span style={{ fontSize: 11, color: "rgba(255,255,255,0.28)" }}>Detection Threshold</span>
                <span style={{ fontSize: 11, fontFamily: "'Space Mono', monospace", color: "#00f5a0" }}>
                  {threshold.toFixed(2)}
                </span>
              </div>
              <input
                type="range" min="0" max="1" step="0.05"
                value={threshold}
                onChange={e => setThreshold(parseFloat(e.target.value))}
                style={{ width: "100%", accentColor: "#00f5a0", cursor: "pointer" }}
              />
              <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
                <span style={{ fontSize: 9, fontFamily: "'Space Mono', monospace", color: "rgba(255,255,255,0.18)" }}>Permissive</span>
                <span style={{ fontSize: 9, fontFamily: "'Space Mono', monospace", color: "rgba(255,255,255,0.18)" }}>Strict</span>
              </div>
            </div>
          </div>

          {/* Live metrics */}
          <div style={{
            padding: 18, borderRadius: 12,
            background: "rgba(255,255,255,0.025)",
            border: "1px solid rgba(255,255,255,0.08)",
          }}>
            <div style={{
              fontSize: 10, fontFamily: "'Space Mono', monospace",
              textTransform: "uppercase", letterSpacing: 1.8,
              color: "rgba(255,255,255,0.28)", marginBottom: 14,
            }}>Live Metrics</div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
              {metrics.map(m => (
                <div key={m.label} style={{
                  padding: "10px 12px",
                  background: "rgba(255,255,255,0.035)",
                  borderRadius: 8, border: "1px solid rgba(255,255,255,0.07)",
                }}>
                  <div style={{
                    fontSize: 17, fontWeight: 700, color: m.color,
                    fontFamily: "'Space Mono', monospace", marginBottom: 3, lineHeight: 1,
                    transition: "color 0.3s",
                  }}>{m.value}</div>
                  <div style={{ fontSize: 9, color: "rgba(255,255,255,0.25)", textTransform: "uppercase", letterSpacing: 0.8 }}>
                    {m.label}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Status card */}
          <div style={{
            padding: 16, borderRadius: 12,
            background: cameraOn ? `${resultColor}06` : "rgba(255,255,255,0.018)",
            border: `1px solid ${cameraOn ? `${resultColor}1a` : "rgba(255,255,255,0.07)"}`,
            transition: "all 0.4s",
          }}>
            <div style={{
              fontSize: 10, fontFamily: "'Space Mono', monospace",
              textTransform: "uppercase", letterSpacing: 1.5,
              color: cameraOn ? resultColor : "rgba(255,255,255,0.22)",
              marginBottom: 8, transition: "color 0.3s",
            }}>
              {!cameraOn ? "Detection Standby"
               : liveResult ? `Verdict: ${liveResult.verdict}`
               : "Initializing…"}
            </div>
            <div style={{ fontSize: 12, color: "rgba(255,255,255,0.4)", lineHeight: 1.65 }}>
              {!cameraOn
                ? "Start the camera to begin real-time deepfake detection using the forensic AI engine."
                : liveResult
                ? `Risk level: ${liveResult.risk_level || "—"}. Analyzing frames at ${Math.round(1000 / 2500)}fps sampling rate.`
                : "Waiting for the first frame analysis to complete…"}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
