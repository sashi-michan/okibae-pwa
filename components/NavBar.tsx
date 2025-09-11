import Link from 'next/link';
export default function NavBar() {
  return (
    <header className="sticky top-0 z-10 bg-gradient-to-r from-pink-50/95 via-white/90 to-orange-50/95 backdrop-blur-md supports-[backdrop-filter]:bg-white/80 border-b border-pink-100/50 shadow-sm">
      <nav className="mx-auto max-w-3xl px-4 py-3 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2 font-bold transition-all duration-300 hover:scale-105">
          <img 
            src="/okibae-icon.svg" 
            alt="OKIBAE" 
            className="h-8 w-8 transition-transform duration-300 hover:scale-110" 
          />
        </Link>
        <div className="flex items-center gap-4 text-sm">
          <Link href="/about" className="hover:text-brand-600 transition-colors duration-200 relative group">
            このアプリについて
            <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-brand-500 group-hover:w-full transition-all duration-300"></span>
          </Link>
        </div>
      </nav>
    </header>
  )
}
