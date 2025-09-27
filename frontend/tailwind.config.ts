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
        surface: {
          100: "#111827",
          200: "#1f2937",
          300: "#273248",
        },
      },
      boxShadow: {
        glass: "0 20px 45px -20px rgba(0,0,0,0.45)",
      },
    },
  },
  plugins: [],
} satisfies Config;
