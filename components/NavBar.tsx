import Link from 'next/link';
import { useEffect, useState, useCallback } from 'react';
import { useAuth } from '../contexts/AuthContext';

export default function NavBar() {
  const { user, userData, signOut } = useAuth()
  const [mounted, setMounted] = useState(false)
  const [isPurchasing, setIsPurchasing] = useState(false)

  // クライアントサイドでのみレンダリング
  useEffect(() => {
    setMounted(true)
  }, [])

  // クレジット残高
  const balance = userData?.credits.balance ?? 0
  const isPro = userData?.subscription.status === 'pro'

  const handlePurchase = useCallback(async () => {
    setIsPurchasing(true)
    try {
      // Checkout Session作成
      const response = await fetch('/api/stripe/create-checkout-session', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      })

      if (!response.ok) {
        throw new Error('Failed to create checkout session')
      }

      const { url } = await response.json()

      // Stripeチェックアウトページへリダイレクト
      if (url) {
        window.location.href = url
      }
    } catch (error) {
      console.error('Purchase error:', error)
      alert('購入処理でエラーが発生しました。もう一度お試しください。')
      setIsPurchasing(false)
    }
  }, [])

  const handleSignOut = useCallback(async () => {
    console.log('ログアウトボタンがクリックされました')
    try {
      await signOut()
      console.log('ログアウト成功')
      // リダイレクトはAuthContextが自動でやってくれる
    } catch (error) {
      console.error('ログアウトエラー:', error)
    }
  }, [signOut])

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
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full" style={{
              backgroundColor: !isPro && balance === 0 ? '#FEE2E2' :
                             !isPro && balance <= 5 ? '#FED7AA' :
                             '#C792A3',
              color: !isPro && balance === 0 ? '#A0616A' :
                     !isPro && balance <= 5 ? '#C4894D' :
                     'white'
            }}>
              <span className="text-xs font-medium">
                {isPro ? `Pro ${balance}回` : `のこり ${balance}回`}
              </span>
              <button
                onClick={handlePurchase}
                disabled={isPurchasing}
                className="px-2 py-0.5 rounded-md text-xs font-medium bg-white/20 hover:bg-white/30 border border-white/40 transition-all duration-200 flex items-center gap-1 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <span className="text-sm font-bold">+</span>
                {isPurchasing ? '処理中...' : '購入'}
              </button>
            </div>
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
