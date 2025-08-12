/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./pages/**/*.{js,ts,jsx,tsx}","./components/**/*.{js,ts,jsx,tsx}"],
  theme: { extend: {
    colors: { brand: {50:"#fef7f7",100:"#fdeeee",200:"#fbdada",300:"#f6bdbd",400:"#ef9e9e",500:"#e27f7f",600:"#c55f5f",700:"#a14545",800:"#7e3535",900:"#582525"}, cream:"#FAF9F6", beige:"#F4EDE4"},
    boxShadow:{soft:"0 8px 30px rgba(0,0,0,0.06)"}, borderRadius:{'xl2':'1.25rem'}
  }},
  plugins: [],
}
