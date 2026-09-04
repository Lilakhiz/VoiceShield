/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        void: '#12161C',
        panel: '#1B212B',
        panel2: '#212836',
        line: '#2A3340',
        ink: '#E7ECF2',
        dim: '#8B96A5',
        amber: '#E8A33D',
        cyan: '#4FD1C5',
        yellow: '#E8C34D',
        rust: '#E2574C',
      },
      fontFamily: {
        mono: ['"IBM Plex Mono"', 'ui-monospace', 'monospace'],
        sans: ['"IBM Plex Sans"', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
