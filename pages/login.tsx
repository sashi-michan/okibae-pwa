import { useEffect, useState } from 'react'
import { useRouter } from 'next/router'
import { useAuth } from '../contexts/AuthContext'
import Head from 'next/head'

export default function Login() {
  const { user, loading, signInWithGoogle } = useAuth()
  const router = useRouter()
  const [isSigningIn, setIsSigningIn] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  useEffect(() => {
    // すでにログイン済みならホームへリダイレクト
    if (!loading && user) {
      router.push('/')
    }

    // URLパラメータからエラーをチェック
    const error = router.query.error
    if (error === 'auth_failed') {
      setErrorMessage('ログイン処理に失敗しました。もう一度お試しください。')
    } else if (error === 'no_code') {
      setErrorMessage('認証コードが見つかりませんでした。')
    }
  }, [user, loading, router])

  const handleGoogleSignIn = async () => {
    try {
      setIsSigningIn(true)
      await signInWithGoogle()
    } catch (error) {
      console.error('ログインエラー:', error)
      alert('ログインに失敗しました。もう一度お試しください。')
      setIsSigningIn(false)
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-pink-50 via-cream-50 to-orange-50">
        <div className="text-gray-600">読み込み中...</div>
      </div>
    )
  }

  return (
    <>
      <Head>
        <title>ログイン - OKIBAE</title>
      </Head>
      <div className="min-h-screen flex items-start justify-center bg-gradient-to-br from-pink-50 via-cream-50 to-orange-50 px-4 pt-20">
        <div className="max-w-md w-full">
          {/* カード */}
          <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-lg border border-gray-100 p-8 md:p-10">
            {/* ロゴ・タイトル */}
            <div className="text-center mb-8">
              <h1 className="text-3xl font-bold text-gray-800 mb-2">
                OKIBAE
              </h1>
              <p className="text-sm text-gray-600">
                おしゃれな置き画を、かんたんに
              </p>
            </div>

            {/* 説明 */}
            <div className="mb-8 text-center">
              <p className="text-sm text-gray-700 leading-relaxed">
                手作り作家さん向けの商品撮影背景置き換えアプリ。
                <br />
                ログインして始めましょう
              </p>
            </div>

            {/* エラーメッセージ */}
            {errorMessage && (
              <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-xl">
                <div className="flex items-start gap-3">
                  <svg className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <p className="text-sm text-red-800">{errorMessage}</p>
                </div>
              </div>
            )}

            {/* Googleログインボタン */}
            <button
              onClick={handleGoogleSignIn}
              disabled={isSigningIn}
              className="w-full bg-white hover:bg-gray-50 text-gray-800 font-medium py-3 px-6 rounded-xl border-2 border-gray-200 transition-all duration-200 flex items-center justify-center gap-3 disabled:opacity-50 disabled:cursor-not-allowed shadow-sm hover:shadow-md"
            >
              <svg className="w-5 h-5" viewBox="0 0 24 24">
                <path
                  fill="#4285F4"
                  d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                />
                <path
                  fill="#34A853"
                  d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                />
                <path
                  fill="#FBBC05"
                  d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                />
                <path
                  fill="#EA4335"
                  d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                />
              </svg>
              {isSigningIn ? 'ログイン中...' : 'Googleでログイン'}
            </button>

            {/* 注意書き */}
            <p className="text-xs text-gray-500 text-center mt-6">
              ログインすることで、利用規約とプライバシーポリシーに同意したものとみなされます
            </p>
          </div>
        </div>
      </div>
    </>
  )
}
