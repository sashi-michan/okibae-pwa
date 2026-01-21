import { useState, useEffect } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { useRouter } from 'next/router'
import Link from 'next/link'

export default function PurchasePage() {
  const { user, userData, authLoading } = useAuth()
  const router = useRouter()
  const [isPurchasing, setIsPurchasing] = useState(false)

  // 認証チェック（useEffect内で実行）
  useEffect(() => {
    if (!authLoading && !user) {
      router.push('/login')
    }
  }, [authLoading, user, router])

  const handlePurchase = async () => {
    setIsPurchasing(true)

    // Stripe遷移前のlocalStorage状態を確認
    console.log('[purchase] BEFORE Stripe redirect - localStorage sb- keys:',
      Object.keys(localStorage).filter(k => k.startsWith('sb-')))
    console.log('[purchase] BEFORE Stripe redirect - document.cookie:', document.cookie)

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
        console.log('[purchase] Redirecting to Stripe:', url)
        window.location.href = url
      }
    } catch (error) {
      console.error('Purchase error:', error)
      alert('購入処理でエラーが発生しました。もう一度お試しください。')
      setIsPurchasing(false)
    }
  }

  // 認証チェック中またはログアウト状態の場合は何も表示しない
  if (authLoading || !user) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-pink-50 via-cream-50 to-orange-50">
        <div className="text-gray-600">読み込み中...</div>
      </div>
    )
  }

  const balance = userData?.credits.balance ?? 0

  return (
    <div className="min-h-screen">
      <main className="mx-auto max-w-2xl px-4 py-12">
        <div className="mb-6">
          <button
            onClick={() => router.push('/')}
            className="mb-4 text-brand-600 hover:text-brand-700 flex items-center gap-2 transition-colors"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            戻る
          </button>
        </div>

        <div className="bg-white rounded-2xl shadow-lg border border-pink-100 p-8">
          <h1 className="text-2xl font-bold mb-6 text-center" style={{ color: '#666' }}>
            クレジット購入
          </h1>

          {/* 現在の残高表示 */}
          <div className="bg-pink-50 rounded-lg p-4 mb-6 text-center">
            <p className="text-sm mb-1" style={{ color: '#666' }}>現在の残高</p>
            <p className="text-3xl font-bold" style={{ color: '#C792A3' }}>
              {balance}回
            </p>
          </div>

          {/* クレジットの説明 */}
          <div className="mb-8">
            <h2 className="text-lg font-semibold mb-3" style={{ color: '#666' }}>クレジットとは？</h2>
            <div className="space-y-2" style={{ color: '#666' }}>
              <p>✓ 画像生成1回につき、1クレジットを消費します</p>
              <p>✓ クレジットは購入後すぐに使用できます</p>
              <p>✓ クレジットに有効期限はありません</p>
            </div>
          </div>

          {/* 購入プラン */}
          <div className="mb-8">
            <div className="rounded-xl p-6 transition-all duration-200" style={{
              border: '1px solid #C792A3',
              backgroundColor: '#FFF5F7'
            }}>
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h3 className="text-xl font-bold" style={{ color: '#666' }}>10クレジット</h3>
                  <p className="text-sm" style={{ color: '#666' }}>画像生成10回分</p>
                </div>
                <div className="text-right">
                  <p className="text-3xl font-bold" style={{ color: '#C792A3' }}>¥200</p>
                  <p className="text-xs" style={{ color: '#666' }}>税込</p>
                </div>
              </div>

              <button
                onClick={handlePurchase}
                disabled={isPurchasing}
                className="w-full py-3 rounded-lg font-semibold text-white transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                style={{
                  backgroundColor: '#C792A3',
                  boxShadow: '0 2px 8px rgba(199, 146, 163, 0.3)'
                }}
              >
                {isPurchasing ? '処理中...' : '購入する（決済画面へ）'}
              </button>
            </div>
          </div>

          {/* 特商法リンク */}
          <div className="text-center text-sm" style={{ color: '#666' }}>
            <p className="mb-2">お支払い方法：クレジットカード（Stripe決済）</p>
            <Link
              href="/tokushoho"
              className="underline hover:opacity-80 transition-opacity"
              style={{ color: '#666' }}
            >
              特定商取引法に基づく表記
            </Link>
          </div>
        </div>
      </main>
    </div>
  )
}
