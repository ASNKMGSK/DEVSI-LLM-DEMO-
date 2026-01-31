/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,jsx}',
    './components/**/*.{js,jsx}',
  ],
  theme: {
    extend: {
      colors: {
        // 데브시스터즈 쿠키런 브랜드 컬러 (부드러운 버전)
        devsisters: {
          yellow: '#EAC54F',
          orange: '#D97B4A',
          brown: '#5C4A3D',
          cream: '#FAF8F5',
          dark: '#3D3428',
        },
        cookie: {
          primary: '#EAC54F',
          secondary: '#D97B4A',
          accent: '#E06B6B',
          success: '#5CB97A',
          warning: '#EAC54F',
          error: '#D96B6B',
          info: '#6BA3D9',
          yellow: '#EAC54F',
          orange: '#D97B4A',
          brown: '#5C4A3D',
          cream: '#FAF8F5',
          beige: '#F0EDE8',
          light: '#FDF9F3',
        },
        // 등급별 컬러 (부드러운 버전)
        grade: {
          common: '#9CA3AF',
          rare: '#6BA3D9',
          superrare: '#A78BFA',
          epic: '#C084FC',
          legendary: '#EAC54F',
          ancient: '#D97B4A',
        },
      },
      fontFamily: {
        sans: ['Pretendard', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'Noto Sans KR', 'sans-serif'],
      },
      backgroundImage: {
        'cookie-gradient': 'linear-gradient(135deg, #EAC54F 0%, #D97B4A 100%)',
        'dark-gradient': 'linear-gradient(135deg, #3D3428 0%, #5C4A3D 100%)',
      },
      boxShadow: {
        'cookie': '0 4px 12px 0 rgba(234, 197, 79, 0.15)',
        'cookie-lg': '0 8px 24px -3px rgba(234, 197, 79, 0.2)',
        'soft': '0 2px 8px rgba(60, 52, 40, 0.05)',
        'soft-lg': '0 8px 24px rgba(60, 52, 40, 0.08)',
      },
      animation: {
        'bounce-slow': 'bounce 3s infinite',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'fade-in': 'fadeIn 0.3s ease-out',
        'slide-in': 'slideIn 0.25s ease-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0', transform: 'translateY(8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        slideIn: {
          '0%': { opacity: '0', transform: 'translateX(-10px)' },
          '100%': { opacity: '1', transform: 'translateX(0)' },
        },
      },
      transitionTimingFunction: {
        'smooth': 'cubic-bezier(0.4, 0, 0.2, 1)',
      },
    },
  },
  plugins: [require('@tailwindcss/typography')],
};
