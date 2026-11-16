/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        background: '#0a0d14',
        card: '#111726',
        'card-hover': '#161e31',
        border: '#1e293b',
        'border-focus': '#3b82f6',
        accent: {
          blue: '#3b82f6',
          indigo: '#6366f1',
          violet: '#8b5cf6',
          cyan: '#06b6d4',
        },
        risk: {
          human: '#10b981',
          'human-bg': 'rgba(16, 185, 129, 0.12)',
          ai: '#ef4444',
          'ai-bg': 'rgba(239, 68, 68, 0.12)',
          uncertain: '#f59e0b',
          'uncertain-bg': 'rgba(245, 158, 11, 0.12)',
        }
      },
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'Courier New', 'monospace'],
      },
    },
  },
  plugins: [],
}
