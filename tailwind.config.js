/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./pages/**/*.{js,ts,jsx,tsx}","./components/**/*.{js,ts,jsx,tsx}"],
  theme: { extend: {
    colors: { 
      brand: {50:"#faf7f8",100:"#f5eef0",200:"#eddde1",300:"#dfc4ca",400:"#c2a2a8",500:"#C2A2A8",600:"#a8848c",700:"#8a6b73",800:"#6d535a",900:"#4d3c42"}, 
      cream:"#FAF9F6", 
      beige:"#F4EDE4"
    },
    fontFamily: {
      'friendly': [
        'YuGothic', 
        'Yu Gothic Medium', 
        'Yu Gothic',
        'Hiragino Kaku Gothic ProN', 
        'Hiragino Sans', 
        'BIZ UDPGothic',
        'Meiryo', 
        'system-ui',
        'sans-serif'
      ],
      'cute': [
        'Rounded Mplus 1c',
        'M PLUS Rounded 1c', 
        'YuGothic', 
        'Yu Gothic Medium',
        'Hiragino Kaku Gothic ProN',
        'BIZ UDPGothic', 
        'system-ui',
        'sans-serif'
      ],
      'sans': [
        'YuGothic', 
        'Yu Gothic Medium', 
        'Yu Gothic',
        'Hiragino Kaku Gothic ProN', 
        'Hiragino Sans', 
        'BIZ UDPGothic',
        'Meiryo', 
        'system-ui',
        'sans-serif'
      ]
    },
    boxShadow: {
      soft: "0 8px 30px rgba(0,0,0,0.06)"
    }, 
    borderRadius: {
      'xl2': '1.25rem'
    }
  }},
  plugins: [],
}
