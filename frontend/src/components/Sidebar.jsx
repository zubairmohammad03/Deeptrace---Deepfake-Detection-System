// Sidebar.jsx — DeepTrace AI navigation panel
import { useState } from "react";

const IconAnalysis = () => (
  <svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round">
    <circle cx="8" cy="8" r="6.5" />
    <circle cx="8" cy="8" r="2.5" />
    <line x1="8" y1="1.5" x2="8" y2="0.2" />
    <line x1="8" y1="14.5" x2="8" y2="15.8" />
    <line x1="14.5" y1="8" x2="15.8" y2="8" />
    <line x1="1.5" y1="8" x2="0.2" y2="8" />
  </svg>
);

const IconBatch = () => (
  <svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round">
    <rect x="1" y="1" width="6" height="6" rx="1.5" />
    <rect x="9" y="1" width="6" height="6" rx="1.5" />
    <rect x="1" y="9" width="6" height="6" rx="1.5" />
    <rect x="9" y="9" width="6" height="6" rx="1.5" />
  </svg>
);

const IconHistory = () => (
  <svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="8" cy="8" r="6.5" />
    <polyline points="8,4 8,8.5 10.8,10.8" />
  </svg>
);

const IconLive = () => (
  <svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
    <rect x="1" y="4" width="10" height="8" rx="2" />
    <polyline points="11,6.5 15,4 15,12 11,9.5" />
  </svg>
);

const IconApi = () => (
  <svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="5,4.5 1.5,8 5,11.5" />
    <polyline points="11,4.5 14.5,8 11,11.5" />
    <line x1="9.5" y1="3" x2="6.5" y2="13" />
  </svg>
);

const NAV = [
  { id: "analysis", label: "Single Analysis", icon: <IconAnalysis /> },
  { id: "batch",    label: "Batch Analysis",  icon: <IconBatch /> },
  { id: "history",  label: "Scan History",    icon: <IconHistory /> },
  { id: "live",     label: "Live Detection",  icon: <IconLive /> },
  { id: "api",      label: "API Integration", icon: <IconApi /> },
];

export default function Sidebar({ activeSection, onSection, engineLabel, engineSub, engineColor }) {
  const [hovered, setHovered] = useState(null);

  return (
    <aside style={{
      width: 224,
      flexShrink: 0,
      height: "100vh",
      position: "sticky",
      top: 0,
      display: "flex",
      flexDirection: "column",
      background: "rgba(5,8,14,0.97)",
      borderRight: "1px solid rgba(255,255,255,0.055)",
      backdropFilter: "blur(20px)",
      zIndex: 20,
    }}>

      {/* Brand */}
      <div style={{
        padding: "20px 18px 18px",
        borderBottom: "1px solid rgba(255,255,255,0.055)",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{
            width: 36, height: 36, borderRadius: 10, flexShrink: 0,
            background: "linear-gradient(135deg, #00f5a0, #00d9f5)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 17,
            boxShadow: "0 0 22px rgba(0,245,160,0.35), 0 0 40px rgba(0,245,160,0.12)",
            animation: "glow-pulse 3s ease-in-out infinite",
          }}>🔍</div>
          <div>
            <div style={{
              fontFamily: "'Space Mono', monospace",
              fontSize: 13, fontWeight: 700,
              color: "#fff", letterSpacing: -0.3,
            }}>DeepTrace</div>
            <div style={{
              fontFamily: "'Space Mono', monospace",
              fontSize: 9, color: "rgba(255,255,255,0.25)",
              letterSpacing: 1.8, textTransform: "uppercase",
            }}>Forensic AI · v4</div>
          </div>
        </div>
      </div>

      {/* Section label */}
      <div style={{ padding: "16px 18px 6px" }}>
        <span style={{
          fontSize: 9,
          fontFamily: "'Space Mono', monospace",
          letterSpacing: 2.2,
          color: "rgba(255,255,255,0.18)",
          textTransform: "uppercase",
        }}>Modules</span>
      </div>

      {/* Nav */}
      <nav style={{ flex: 1, padding: "4px 10px", overflowY: "auto" }}>
        {NAV.map(item => {
          const active  = activeSection === item.id;
          const isHover = hovered === item.id && !active;
          return (
            <button
              key={item.id}
              onClick={() => onSection(item.id)}
              onMouseEnter={() => setHovered(item.id)}
              onMouseLeave={() => setHovered(null)}
              style={{
                width: "100%",
                display: "flex",
                alignItems: "center",
                gap: 10,
                padding: "9px 12px",
                borderRadius: 8,
                border: "none",
                marginBottom: 2,
                background: active
                  ? "rgba(0,245,160,0.1)"
                  : isHover
                  ? "rgba(255,255,255,0.055)"
                  : "transparent",
                color: active
                  ? "#00f5a0"
                  : isHover
                  ? "rgba(255,255,255,0.8)"
                  : "rgba(255,255,255,0.4)",
                cursor: "pointer",
                textAlign: "left",
                fontSize: 13,
                fontWeight: active ? 600 : 400,
                fontFamily: "'IBM Plex Sans', system-ui, sans-serif",
                transition: "all 0.17s cubic-bezier(0.4,0,0.2,1)",
                outline: "none",
                position: "relative",
                transform: isHover ? "translateX(3px)" : "translateX(0)",
                boxShadow: active ? "inset 0 0 0 1px rgba(0,245,160,0.14)" : "none",
              }}
            >
              {/* Active left bar */}
              {active && (
                <span style={{
                  position: "absolute",
                  left: 0,
                  top: "16%",
                  bottom: "16%",
                  width: 3,
                  borderRadius: 2,
                  background: "linear-gradient(180deg, #00f5a0, #00d9f5)",
                  boxShadow: "0 0 10px #00f5a0, 0 0 20px rgba(0,245,160,0.4)",
                }} />
              )}

              {/* Icon */}
              <span style={{
                color: active ? "#00f5a0" : isHover ? "rgba(255,255,255,0.7)" : "inherit",
                lineHeight: 0,
                transition: "color 0.17s, transform 0.17s",
                transform: isHover ? "scale(1.12)" : "scale(1)",
                flexShrink: 0,
              }}>
                {item.icon}
              </span>

              {item.label}

              {/* Active dot indicator */}
              {active && (
                <span style={{
                  marginLeft: "auto",
                  width: 5, height: 5, borderRadius: "50%",
                  background: "#00f5a0",
                  boxShadow: "0 0 6px #00f5a0",
                  animation: "dt-pulse 2s ease-in-out infinite",
                  flexShrink: 0,
                }} />
              )}
            </button>
          );
        })}
      </nav>

      {/* Footer — engine status */}
      <div style={{
        padding: "14px 18px",
        borderTop: "1px solid rgba(255,255,255,0.055)",
      }}>
        <div style={{
          fontSize: 9, fontFamily: "'Space Mono', monospace",
          color: "rgba(255,255,255,0.18)", letterSpacing: 1.5,
          textTransform: "uppercase", marginBottom: 7,
        }}>Engine Status</div>

        <div style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          padding: "8px 10px",
          background: `${engineColor}0f`,
          borderRadius: 8,
          border: `1px solid ${engineColor}28`,
          marginBottom: 10,
          transition: "all 0.4s ease",
        }}>
          <span style={{
            width: 7, height: 7, borderRadius: "50%", flexShrink: 0,
            background: engineColor,
            boxShadow: `0 0 10px ${engineColor}, 0 0 20px ${engineColor}55`,
            animation: "dt-pulse 2.2s ease-in-out infinite",
          }} />
          <div style={{ minWidth: 0 }}>
            <div style={{
              fontSize: 11,
              fontFamily: "'Space Mono', monospace",
              color: engineColor,
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
              fontWeight: 700,
            }}>
              {engineLabel}
            </div>
            {engineSub && (
              <div style={{
                fontSize: 9,
                fontFamily: "'Space Mono', monospace",
                color: "rgba(255,255,255,0.3)",
                marginTop: 1,
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
              }}>
                {engineSub}
              </div>
            )}
          </div>
        </div>

        <div style={{
          fontSize: 9,
          color: "rgba(255,255,255,0.1)",
          fontFamily: "'Space Mono', monospace",
          letterSpacing: 0.8,
          lineHeight: 1.6,
        }}>
          DEEPTRACE AI v4<br />FORENSIC DEEPFAKE DETECTION
        </div>
      </div>
    </aside>
  );
}
