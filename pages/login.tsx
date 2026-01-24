import { useEffect, useState } from 'react'
import { useRouter } from 'next/router'
import { useAuth } from '../contexts/AuthContext'
import Head from 'next/head'
import { LegalModal } from '../components/LegalModal'
import { ErrorModal, ErrorType } from '../components/ErrorModal'

export default function Login() {
  const { user, authLoading, signInWithGoogle } = useAuth()
  const router = useRouter()
  const [isSigningIn, setIsSigningIn] = useState(false)
  const [errorType, setErrorType] = useState<ErrorType | null>(null)
  const [showTermsModal, setShowTermsModal] = useState(false)
  const [showPrivacyModal, setShowPrivacyModal] = useState(false)

  useEffect(() => {
    if (!authLoading && user) {
      router.push('/')
      return
    }

    // OAuth コールバックからのエラー
    const error = router.query.error
    if (error === 'auth_failed') setErrorType('AUTH_OAUTH_FAILED')
    else if (error === 'no_code') setErrorType('AUTH_NO_CODE')

    // AuthContext からの認証エラー
    const reason = router.query.reason
    if (reason === 'restore_failed') {
      setErrorType('AUTH_RESTORE_FAILED')
    } else if (reason === 'fetch_failed') {
      setErrorType('AUTH_FETCH_FAILED')
    } else if (reason === 'signout_failed') {
      setErrorType('AUTH_SIGNOUT_FAILED')
    }
  }, [user, authLoading, router])

  const handleGoogleSignIn = async () => {
    try {
      if (process.env.NODE_ENV === 'development') {
        console.log('[login] ログインボタン押した')
      }
      setIsSigningIn(true)

      // 5秒後に保険でisSigningInをfalseに戻す（リダイレクト失敗時の保険）
      const timeoutId = setTimeout(() => {
        setIsSigningIn(false)
      }, 5000)

      await signInWithGoogle()
      // OAuth リダイレクトが開始されるため、通常ここには到達しない
      clearTimeout(timeoutId)
    } catch (error: any) {
      if (process.env.NODE_ENV === 'development') {
        console.log('[login] ログイン失敗:', error?.message || String(error))
      }
      setErrorType('AUTH_OAUTH_FAILED')
      setIsSigningIn(false)
    }
  }

  const handleRetry = () => {
    setErrorType(null)
    setIsSigningIn(false)
  }

  const handleReload = () => {
    window.location.reload()
  }

  if (authLoading || isSigningIn) {
    return (
      <div className="min-h-screen grid place-items-center bg-[#fdfcfb]">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-gray-400 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <div className="text-gray-400 font-light tracking-widest">ログイン中...</div>
        </div>
      </div>
    )
  }

  return (
    <>
      <Head>
        <title>ログイン - OKIBAE</title>
      </Head>

      {/* 全体を包む背景 */}
      <div className="w-full min-h-screen flex flex-col items-center justify-center p-4 relative overflow-hidden" style={{ background: 'linear-gradient(135deg, #fdfcfb 0%, #f7f4f1 50%, #fff5f0 100%)' }}>

        {/* 【ガラスのカード】
           py-12 md:py-20: 縦の余白をガッツリ増やしました！これで縦長でゆったりした印象になります。
           max-w-4xl: 幅はコンパクトなまま維持
        */}
        <div className="w-full max-w-4xl bg-white/40 backdrop-blur-3xl rounded-[2.5rem] border border-white/60 shadow-2xl shadow-[#b5a397]/10 flex flex-col md:flex-row items-center justify-center gap-10 md:gap-16 relative overflow-hidden py-12 px-8 md:py-20 md:px-12 z-10 mx-auto">
            
            {/* ガラスボード内の光沢エフェクト */}
            <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-br from-white/50 via-transparent to-transparent pointer-events-none"></div>

            {/* 左側：ビジュアル 
               h-72 md:h-[400px]: 画像エリアの高さを大幅にアップ！
               これでBeforeとAfterが上下に離れて、重なりが減ります。
            */}
            <div className="flex-1 w-full flex flex-col items-center md:items-end justify-center relative z-20">
               <div className="relative w-64 h-72 md:w-[360px] md:h-[400px]">
                  {/* Before画像（上側） */}
                  <div className="absolute top-0 right-0 md:right-0 w-40 h-48 md:w-56 md:h-64 bg-white/40 backdrop-blur-md p-2.5 rounded-2xl transform rotate-6 border border-white/50 shadow-lg float-element float-delay-1">
                    <div className="w-full h-full rounded-xl overflow-hidden relative">
                      <img
                        src="/samples/before.jpg"
                        alt="Before"
                        className="w-full h-full object-cover"
                      />
                      <div className="absolute bottom-3 right-3 bg-white/80 backdrop-blur-md px-3 py-1 rounded-full text-[10px] md:text-xs text-gray-600 font-medium tracking-wider shadow-sm border border-white/50">
                        Before
                      </div>
                    </div>
                  </div>

                  {/* After画像（下側）
                     bottom-0: エリアの一番下に配置。エリアが縦長になった分、Beforeから離れます。
                  */}
                  <div className="absolute bottom-0 left-0 md:left-2 w-44 h-56 md:w-56 md:h-72 bg-[#fcf9f7] p-2.5 rounded-2xl transform -rotate-3 z-20 shadow-2xl shadow-neutral-400/20 border border-white float-element float-delay-2">
                    <div className="w-full h-full rounded-xl overflow-hidden relative bg-gray-50">
                      <img
                        src="/samples/after.jpg"
                        alt="Sample"
                        className="w-full h-full object-cover"
                      />
                      <div className="absolute bottom-3 right-3 bg-white/80 backdrop-blur-md px-3 py-1 rounded-full text-[10px] md:text-xs text-gray-600 font-medium tracking-wider shadow-sm border border-white/50">
                        After
                      </div>
                    </div>
                  </div>
                </div>
            </div>

            {/* 右側：フォーム */}
            <div className="flex-1 w-full flex flex-col items-center md:items-start justify-center relative z-20">
              <div className="text-center md:text-left mb-8 md:mb-10">
                <div className="flex items-center justify-center md:justify-start gap-3 mb-4">
                  <img
                    src="/okibae-icon.svg"
                    alt="OKIBAE"
                    className="h-10 w-10 md:h-12 md:w-12"
                  />
                  <h1 className="text-3xl md:text-5xl font-thin tracking-[0.15em] font-sans" style={{ color: '#666' }}>
                    OKIBAE
                  </h1>
                </div>
                <p className="text-sm text-gray-600 font-light leading-loose tracking-wider">
                  おしゃれな置き画が簡単に作れるアプリ。<br />
                  あなたの商品写真を、もっと<span className="text-gray-800 font-normal border-b border-[#d4c4b7] pb-1">素敵</span>に。
                </p>
              </div>


              <div className="space-y-6">
                <button
                  onClick={handleGoogleSignIn}
                  disabled={isSigningIn}
                  className="w-full md:w-auto min-w-[240px] bg-white hover:bg-[#fafaf9] text-gray-600 font-medium py-3.5 px-8 rounded-full transition-all duration-500 flex items-center justify-center gap-4 shadow-xl shadow-[#b5a397]/10 hover:shadow-2xl hover:shadow-[#b5a397]/20 hover:-translate-y-1 group relative overflow-hidden tracking-wider border border-white/60 text-sm md:text-base"
                >
                  <img src="https://www.google.com/favicon.ico" alt="G" className="w-4 h-4 md:w-5 md:h-5 opacity-70 group-hover:opacity-100 transition-opacity" />
                  <span className="relative z-10">{isSigningIn ? '接続中...' : 'Googleではじめる'}</span>
                </button>
              </div>

              <div className="mt-10 text-center md:text-left">
                <p className="text-[10px] text-gray-500/60 font-light tracking-wide">
                  続行することで、
                  <a
                    href="#"
                    onClick={(e) => {
                      e.preventDefault()
                      setShowTermsModal(true)
                    }}
                    className="underline decoration-gray-300 hover:text-gray-600 transition-colors mx-1 cursor-pointer"
                  >
                    利用規約
                  </a>
                  ・
                  <a
                    href="#"
                    onClick={(e) => {
                      e.preventDefault()
                      setShowPrivacyModal(true)
                    }}
                    className="underline decoration-gray-300 hover:text-gray-600 transition-colors mx-1 cursor-pointer"
                  >
                    プライバシーポリシー
                  </a>
                  <br className="md:hidden"/>
                  に同意したものとみなされます。
                </p>
              </div>
            </div>

        </div>

        {/* フッター（著作権表記） */}
        <footer className="text-center text-xs text-gray-500 py-6 relative z-20">
          <span>© {new Date().getFullYear()} OKIBAE</span>
        </footer>
      </div>

      {/* モーダル */}
      <LegalModal
        isOpen={showTermsModal}
        onClose={() => setShowTermsModal(false)}
        type="terms"
      />
      <LegalModal
        isOpen={showPrivacyModal}
        onClose={() => setShowPrivacyModal(false)}
        type="privacy"
      />

      {/* エラーモーダル */}
      {errorType && (
        <ErrorModal
          isOpen={true}
          onClose={() => {
            setErrorType(null)
            // クエリパラメータをクリア
            router.replace('/login', undefined, { shallow: true })
          }}
          errorType={errorType}
          onRetry={handleRetry}
          onReload={handleReload}
        />
      )}
    </>
  )
}