// components/ScanningView.jsx
import { ScanOverlay } from "./ui.jsx";

const SCAN_STEPS = [
  "Initializing…",
  "Reading frame data…",
  "Texture and edge checks…",
  "Frequency cues…",
  "Lighting consistency…",
  "Metadata hints…",
  "Scoring signals…",
  "Building summary…",
  "Almost done…",
  "Finalizing…",
];

const MODULES = ["Faces & regions", "Texture", "Frequency", "Edges", "Lighting", "Model fusion"];

export default function ScanningView({ preview, progress, stepIndex }) {
  return (
    <div
      style={{
        borderRadius: 8,
        overflow: "hidden",
        border: "1px solid var(--dt-border-subtle)",
        background: "var(--dt-surface)",
        animation: "dt-fade-up 0.4s ease",
      }}
    >
      {preview && (
        <div style={{ position: "relative", maxHeight: 320, overflow: "hidden" }}>
          <img
            src={preview}
            alt=""
            style={{
              width: "100%",
              maxHeight: 320,
              objectFit: "cover",
              display: "block",
              filter: "brightness(0.45) contrast(1.05)",
            }}
          />
          <ScanOverlay />
          <div
            style={{
              position: "absolute",
              inset: 0,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              gap: 12,
            }}
          >
            <div
              style={{
                width: 44,
                height: 44,
                border: "2px solid var(--dt-border-subtle)",
                borderTopColor: "var(--dt-accent)",
                borderRadius: "50%",
                animation: "dt-spin 0.85s linear infinite",
              }}
            />
            <span
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                fontWeight: 500,
                color: "var(--dt-accent)",
                letterSpacing: "0.14em",
                textTransform: "uppercase",
              }}
            >
              Working
            </span>
          </div>
        </div>
      )}

      <div style={{ padding: "22px 24px" }}>
        <div style={{ marginBottom: 18 }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8, gap: 12 }}>
            <span
              style={{
                fontSize: 13,
                color: "var(--dt-muted)",
                lineHeight: 1.4,
              }}
            >
              {SCAN_STEPS[Math.min(stepIndex, SCAN_STEPS.length - 1)]}
            </span>
            <span
              style={{
                fontSize: 13,
                fontFamily: "var(--font-mono)",
                fontWeight: 600,
                color: "var(--dt-accent)",
                flexShrink: 0,
              }}
            >
              {Math.round(progress)}%
            </span>
          </div>
          <div
            style={{
              height: 3,
              background: "var(--dt-surface-2)",
              borderRadius: 2,
              overflow: "hidden",
            }}
          >
            <div
              style={{
                height: "100%",
                borderRadius: 2,
                width: `${progress}%`,
                background: "var(--dt-accent)",
                transition: "width 0.45s ease",
              }}
            />
          </div>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))",
            gap: 8,
          }}
        >
          {MODULES.map((mod, i) => {
            const active = progress > i * 14;
            return (
              <div
                key={mod}
                style={{
                  padding: "8px 10px",
                  borderRadius: 4,
                  background: active ? "var(--dt-accent-soft)" : "var(--dt-surface-2)",
                  border: `1px solid ${active ? "var(--dt-border)" : "var(--dt-border-subtle)"}`,
                  transition: "background 0.35s ease, border-color 0.35s ease",
                }}
              >
                <span
                  style={{
                    fontSize: 11,
                    color: active ? "var(--dt-text)" : "var(--dt-muted)",
                  }}
                >
                  {mod}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
