import type { Config } from "tailwindcss";
import defaultTheme from "tailwindcss/defaultTheme";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", ...defaultTheme.fontFamily.sans],
      },
      colors: {
        bg: "var(--bg)",
        panel: "var(--panel)",
        "panel-elev": "var(--panel-elev)",
        border: "var(--border)",
        muted: "var(--muted)",
        text: "var(--text)",
        "text-dim": "var(--text-dim)",
        accent: "var(--accent)",
        "accent-weak": "var(--accent-weak)",
        danger: "var(--danger)",
        warning: "var(--warning)",
        ok: "var(--ok)",
        focus: "var(--focus)",
      },
      boxShadow: {
        panel: "0 18px 45px -28px rgba(0,0,0,0.6)",
      },
      borderRadius: {
        xl: "var(--radius)",
        "2xl": "calc(var(--radius) + 6px)",
      },
    },
  },
  plugins: [],
} satisfies Config;
