import Link from 'next/link';
import { useEffect, useState, useCallback } from 'react';
import { useRouter } from 'next/router';
import { useAuth } from '../contexts/AuthContext';
import DeleteAccountModal from './DeleteAccountModal';

export default function NavBar() {
  const { user, userData, signOut } = useAuth()
  const router = useRouter()
  const [mounted, setMounted] = useState(false)
  const [isMenuOpen, setIsMenuOpen] = useState(false)
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false)

  // クライアントサイドでのみレンダリング
  useEffect(() => {
    setMounted(true)
  }, [])

  // メニュー外クリックで閉じる
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as HTMLElement
      if (isMenuOpen && !target.closest('.menu-container')) {
        setIsMenuOpen(false)
      }
    }

    if (isMenuOpen) {
      document.addEventListener('click', handleClickOutside)
    }

    return () => {
      document.removeEventListener('click', handleClickOutside)
    }
  }, [isMenuOpen])

  // クレジット残高
  const balance = userData?.credits.balance ?? 0
  const isPro = userData?.subscription.status === 'pro'


  const handlePurchase = useCallback(() => {
    router.push('/purchase')
  }, [router])

  const handleSignOut = useCallback(async () => {
    console.log('ログアウトボタンがクリックされました')
    setIsMenuOpen(false)

    try {
      const timeoutMs = 5000
      await Promise.race([
        signOut(),
        new Promise((_, rej) => setTimeout(() => rej(new Error('signOut timeout')), timeoutMs)),
      ])
    } catch (e) {
      console.error('ログアウト失敗/タイムアウト:', e)
    } finally {
      // "ログアウトが遅い/失敗した" でも必ずログインへ（戻るボタン対策で replace）
      router.replace('/login')
    }
  }, [signOut, router])

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen)
  }

  const closeMenu = () => {
    setIsMenuOpen(false)
  }

  const handleDeleteAccount = useCallback(async () => {
    try {
      const response = await fetch('/api/account/delete', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      })

      if (!response.ok) {
        throw new Error('退会処理に失敗しました')
      }

      // 退会成功後、セッションもクリア
      try {
        await signOut()
      } catch (signOutError) {
        console.error('サインアウトエラー（退会は完了済み）:', signOutError)
        // 退会は成功しているので、signOutエラーは無視して続行
      }

      // 退会成功後、モーダルを閉じてログインページへ遷移（戻るボタン対策で replace）
      setIsDeleteModalOpen(false)
      setIsMenuOpen(false)
      router.replace('/login')
    } catch (error) {
      console.error('退会エラー:', error)
      throw error
    }
  }, [router, signOut])

  const openDeleteModal = () => {
    closeMenu()
    setIsDeleteModalOpen(true)
  }

  return (
    <>
      <header className="sticky top-0 z-50 bg-white/40 backdrop-blur-md border-b border-white/50 shadow-sm">
        <nav className="mx-auto max-w-3xl px-4 py-3 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2 font-bold transition-all duration-300 hover:scale-105">
            <img
              src="/okibae-icon.svg"
              alt="OKIBAE"
              className="h-8 w-8 transition-transform duration-300 hover:scale-110"
            />
          </Link>
          <div className="flex items-center gap-4 text-sm menu-container relative">
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
                  className="px-2 py-0.5 rounded-md text-xs font-medium bg-white/20 hover:bg-white/30 border border-white/40 transition-all duration-200 flex items-center gap-1"
                >
                  <span className="text-sm font-bold">+</span>
                  購入
                </button>
              </div>
            )}

            {/* メニューボタン（ログイン状態で切り替え） */}
            <button
              onClick={toggleMenu}
              className="p-2 hover:bg-pink-100/50 rounded-lg transition-colors duration-200"
              aria-label="メニュー"
            >
              {mounted && user ? (
                // ログイン済み：プロフィール画像 or デフォルト画像
                <img
                  src={user.user_metadata?.avatar_url || '/icons/icon-user.png'}
                  alt="プロフィール"
                  className="w-8 h-8 rounded-full object-cover border-2"
                  style={{ borderColor: 'rgb(230, 205, 220)' }}
                />
              ) : (
                // 未ログイン：ハンバーガーメニュー
                <svg
                  className="w-6 h-6"
                  fill="none"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  viewBox="0 0 24 24"
                  stroke="#666"
                >
                  {isMenuOpen ? (
                    <path d="M6 18L18 6M6 6l12 12" />
                  ) : (
                    <path d="M4 6h16M4 12h16M4 18h16" />
                  )}
                </svg>
              )}
            </button>

            {/* ドロップダウンメニュー */}
            {isMenuOpen && (
              <div className="absolute top-full right-0 mt-2 w-56 bg-white rounded-lg shadow-lg border overflow-hidden z-20" style={{ borderColor: 'rgb(240, 220, 230)' }}>
                <div className="py-1">
                  {/* ログイン中のユーザー情報 */}
                  {user && (
                    <div className="px-4 py-3 border-b bg-pink-50/30" style={{ borderColor: 'rgb(240, 220, 230)' }}>
                      <div className="text-sm font-medium text-gray-800 truncate">
                        {user.email?.split('@')[0]}
                      </div>
                      <div className="text-xs text-gray-500 truncate">{user.email}</div>
                    </div>
                  )}

                  <Link
                    href="/about"
                    onClick={closeMenu}
                    className="block px-4 py-3 text-sm text-gray-700 hover:bg-pink-50 transition-colors duration-150"
                  >
                    使い方
                  </Link>
                  <Link
                    href="/terms"
                    onClick={closeMenu}
                    className="block px-4 py-3 text-sm text-gray-700 hover:bg-pink-50 transition-colors duration-150"
                  >
                    利用規約
                  </Link>
                  <Link
                    href="/privacy"
                    onClick={closeMenu}
                    className="block px-4 py-3 text-sm text-gray-700 hover:bg-pink-50 transition-colors duration-150"
                  >
                    プライバシーポリシー
                  </Link>
                  <Link
                    href="/tokushoho"
                    onClick={closeMenu}
                    className="block px-4 py-3 text-sm text-gray-700 hover:bg-pink-50 transition-colors duration-150"
                  >
                    特定商取引法に基づく表記
                  </Link>

                  {user && (
                    <>
                      <div className="border-t my-1" style={{ borderColor: 'rgb(240, 220, 230)' }}></div>
                      <button
                        onClick={handleSignOut}
                        className="w-full text-left px-4 py-3 text-sm text-gray-700 hover:bg-pink-50 transition-colors duration-150"
                      >
                        ログアウト
                      </button>
                      <button
                        onClick={openDeleteModal}
                        className="w-full text-left px-4 py-3 text-sm text-red-600 hover:bg-red-50 transition-colors duration-150"
                      >
                        退会
                      </button>
                    </>
                  )}
                </div>
              </div>
            )}
          </div>
        </nav>
      </header>

      {/* 退会確認モーダル */}
      <DeleteAccountModal
        isOpen={isDeleteModalOpen}
        onClose={() => setIsDeleteModalOpen(false)}
        onConfirm={handleDeleteAccount}
        balance={balance}
      />
    </>
  )
}
