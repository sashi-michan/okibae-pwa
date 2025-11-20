import Link from 'next/link';
import { useEffect, useState } from 'react';
import { useAuth } from '../contexts/AuthContext';

export default function NavBar() {
  const { user, userData, signOut } = useAuth()
  const [mounted, setMounted] = useState(false)

  // クライアントサイドでのみレンダリング
  useEffect(() => {
    setMounted(true)
  }, [])

  // クレジット残高
  const balance = userData?.credits.balance ?? 0
  const isPro = userData?.subscription.status === 'pro'

  // 無料プランの上限
  const freeLimit = 20

  const handleSignOut = async () => {
    try {
      await signOut()
    } catch (error) {
      console.error('ログアウトエラー:', error)
    }
  }

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
          {mounted && userData && (
            <span className="px-2 py-1 rounded-full text-xs font-medium" style={{
              backgroundColor: !isPro && balance === 0 ? '#FEE2E2' :
                             !isPro && balance <= 5 ? '#FED7AA' :
                             '#C792A3',
              color: !isPro && balance === 0 ? '#A0616A' :
                     !isPro && balance <= 5 ? '#C4894D' :
                     'white'
            }}>
              {isPro ? `Pro ${balance}クレ` : `のこり ${balance}/${freeLimit}`}
            </span>
          )}
          <Link href="/about" className="hover:text-brand-600 transition-colors duration-200 relative group">
            使い方
            <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-brand-500 group-hover:w-full transition-all duration-300"></span>
          </Link>
          {user && (
            <button
              onClick={handleSignOut}
              className="hover:text-brand-600 transition-colors duration-200 relative group"
            >
              ログアウト
              <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-brand-500 group-hover:w-full transition-all duration-300"></span>
            </button>
          )}
        </div>
      </nav>
    </header>
  )
}
