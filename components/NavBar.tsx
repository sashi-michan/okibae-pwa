import Link from 'next/link';
export default function NavBar() {
  return (
    <header className="sticky top-0 z-10 bg-gradient-to-r from-pink-50/95 via-white/90 to-orange-50/95 backdrop-blur-md supports-[backdrop-filter]:bg-white/80 border-b border-pink-100/50 shadow-sm">
      <nav className="mx-auto max-w-3xl px-4 py-3 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2 font-bold transition-all duration-300 hover:scale-105">
          <span className="inline-flex h-8 w-8 items-center justify-center rounded-xl bg-brand-500 text-white shadow-lg relative overflow-hidden group">
            <span className="relative z-10">Ki</span>
            <div className="absolute inset-0 bg-brand-400 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
          </span>
        </Link>
        <div className="flex items-center gap-4 text-sm">
          <Link href="/" className="hover:text-brand-600 transition-colors duration-200 relative group">
            Home
            <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-brand-500 group-hover:w-full transition-all duration-300"></span>
          </Link>
          <Link href="/about" className="hover:text-brand-600 transition-colors duration-200 relative group">
            About
            <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-brand-500 group-hover:w-full transition-all duration-300"></span>
          </Link>
        </div>
      </nav>
    </header>
  )
}
