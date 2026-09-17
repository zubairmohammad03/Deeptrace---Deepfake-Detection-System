// services/claudeApi.js
// All analysis goes through the FastAPI backend.
// Backend uses EfficientNet model if trained, otherwise OpenRouter API.

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";


// ── Main analysis function ────────────────────────────────────────────────────
export async function analyzeMedia(base64Data, mediaType, file = null) {
  if (!file) {
    throw new Error("No file provided for analysis.");
  }

  const formData = new FormData();
  formData.append("file", file);

  let response;
  try {
    response = await fetch(`${BACKEND_URL}/api/detect`, {
      method: "POST",
      body: formData,
      signal: AbortSignal.timeout(60000),
    });
  } catch (err) {
    if (err.name === "TimeoutError") {
      throw new Error("Request timed out after 60s. Is the backend still running?");
    }
    throw new Error(
      "Cannot connect to backend.\n\n" +
      "Start it with:\n" +
      "  cd backend\n" +
      "  uvicorn app.main:app --reload\n\n" +
      "Make sure backend/.env has your OPENROUTER_API_KEY."
    );
  }

  if (!response.ok) {
    let detail = `Server error ${response.status}`;
    try { detail = (await response.json())?.detail || detail; } catch { /* ignore */ }
    if (response.status === 413) throw new Error("File too large. Max 50MB.");
    if (response.status === 400) throw new Error(`Unsupported file type.`);
    if (response.status === 422) throw new Error(`Analysis error: ${detail}`);
    throw new Error(detail);
  }

  const data = await response.json();
  return data.result;
}


// ── Health check ──────────────────────────────────────────────────────────────
export async function checkBackendHealth() {
  try {
    const res = await fetch(`${BACKEND_URL}/api/health`, {
      signal: AbortSignal.timeout(3000),
    });
    return res.ok ? await res.json() : null;
  } catch {
    return null;
  }
}


// ── Video frame extraction (preview only — backend handles full analysis) ─────
export function extractVideoFrame(file) {
  return new Promise((resolve, reject) => {
    const video  = document.createElement("video");
    const canvas = document.createElement("canvas");
    video.preload = "metadata";
    video.muted   = true;
    video.src     = URL.createObjectURL(file);

    video.onloadeddata = () => { video.currentTime = Math.min(1.5, video.duration * 0.25); };
    video.onseeked = () => {
      canvas.width  = Math.min(video.videoWidth, 1024);
      canvas.height = Math.round(video.videoHeight * (canvas.width / video.videoWidth));
      canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
      URL.revokeObjectURL(video.src);
      resolve(canvas.toDataURL("image/jpeg", 0.9).split(",")[1]);
    };
    video.onerror = () => { URL.revokeObjectURL(video.src); reject(new Error("Could not load video.")); };
    setTimeout(() => reject(new Error("Video loading timed out.")), 15000);
  });
}


// ── File to base64 (preview only) ────────────────────────────────────────────
export function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload  = () => resolve(reader.result.split(",")[1]);
    reader.onerror = () => reject(new Error("Failed to read file."));
    reader.readAsDataURL(file);
  });
}
