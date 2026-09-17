// components/UploadZone.jsx
import { useRef, useCallback } from "react";

const ACCEPTED = ["image/jpeg", "image/png", "image/webp", "video/mp4", "video/quicktime", "video/avi", "video/x-msvideo"];

export default function UploadZone({ onFile, error }) {
  const inputRef = useRef();

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      const f = e.dataTransfer.files[0];
      if (f && ACCEPTED.includes(f.type)) onFile(f);
    },
    [onFile]
  );

  const handleChange = (e) => {
    const f = e.target.files[0];
    if (f) onFile(f);
    e.target.value = "";
  };

  return (
    <div>
      <div
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
        onClick={() => inputRef.current.click()}
        style={{
          border: "1px dashed var(--dt-border)",
          borderRadius: 8,
          padding: "56px 36px",
          textAlign: "center",
          cursor: "pointer",
          background: "var(--dt-surface)",
          transition: "border-color 0.2s, background 0.2s",
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.borderColor = "var(--dt-accent)";
          e.currentTarget.style.background = "var(--dt-surface-2)";
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.border = "1px dashed var(--dt-border)";
          e.currentTarget.style.background = "var(--dt-surface)";
        }}
      >
        <h2
          style={{
            fontSize: 18,
            fontWeight: 600,
            marginBottom: 8,
            color: "var(--dt-text)",
          }}
        >
          Drop a file here
        </h2>
        <p
          style={{
            color: "var(--dt-muted)",
            fontSize: 14,
            marginBottom: 24,
            lineHeight: 1.55,
          }}
        >
          JPEG, PNG, WebP, or MP4 / MOV / AVI
          <br />
          <span style={{ fontSize: 13 }}>Up to ~50 MB recommended</span>
        </p>
        <button
          type="button"
          style={{
            display: "inline-block",
            padding: "12px 28px",
            borderRadius: 6,
            background: "var(--dt-accent)",
            color: "#030303",
            fontWeight: 600,
            fontSize: 14,
            border: "none",
            cursor: "pointer",
            fontFamily: "var(--font-sans)",
          }}
        >
          Choose file
        </button>
        <p style={{ color: "var(--dt-muted)", fontSize: 13, marginTop: 16 }}>or drag and drop</p>
      </div>

      <input ref={inputRef} type="file" accept="image/*,video/*" style={{ display: "none" }} onChange={handleChange} />

      {error && (
        <div
          style={{
            marginTop: 16,
            padding: "14px 16px",
            background: "rgba(232, 93, 93, 0.08)",
            border: "1px solid rgba(232, 93, 93, 0.35)",
            borderRadius: 6,
            color: "var(--dt-danger)",
            fontSize: 13,
            lineHeight: 1.5,
          }}
        >
          {error}
        </div>
      )}
    </div>
  );
}
