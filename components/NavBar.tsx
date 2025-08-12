import Link from 'next/link';
export default function NavBar() {
  return (
    <header className="sticky top-0 z-10 bg-cream/80 backdrop-blur supports-[backdrop-filter]:bg-cream/60 border-b border-gray-200">
      <nav className="mx-auto max-w-3xl px-4 py-3 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2 font-bold">
          <span className="inline-flex h-8 w-8 items-center justify-center rounded-xl bg-brand-500 text-white">Ki</span>
          <span className="tracking-wide">OKIBAE</span>
        </Link>
        <div className="flex items-center gap-4 text-sm">
          <Link href="/" className="hover:underline">Home</Link>
          <Link href="/about" className="hover:underline">About</Link>
        </div>
      </nav>
    </header>
  )
}
