import { useEffect, useRef, useState } from 'react'
import { useRouter } from 'next/router'
import clsx from 'clsx'
import { useAuth } from '../contexts/AuthContext'
import StepCard from '../components/StepCard'
import { useDeviceType } from '../hooks/useDeviceType'
import { usePWAInstall } from '../hooks/usePWAInstall'
import { ErrorModal, ErrorType } from '../components/ErrorModal'

type BgOption = 'white' | 'linen' | 'concrete' | 'wood' | 'white_wood'
type WeatherOption = 'sunny' | 'cloudy' | 'rainy'
type AspectRatioOption = 'square' | 'original'

type GenerationRequest = {
  imgUrl: string
  bg: BgOption
  weather: WeatherOption
  aspectRatio: AspectRatioOption
  originalSize: { width: number; height: number }
  selectedBgId: number
  startedAt: number  // タイムスタンプ（デバッグ用）
}

type AppState = {
  phase: 'IDLE' | 'FINAL_RENDERING' | 'FINAL_READY'
  status?: 'loading' | 'error'
  error?: string
  finalImageUrl?: string
  request?: GenerationRequest  // 生成リクエストのスナップショット
  jobId: number  // 生成ジョブID（Step2で使用）
}

export default function Home() {
  // 認証チェック
  const { user, userData, loading, authLoading, errorReason, refreshUserData } = useAuth()
  const router = useRouter()

  // デバイス判定とPWAインストール状態
  const { isIOS, isAndroid } = useDeviceType()
  const { isInstalled, canPrompt, promptInstall } = usePWAInstall()

  // すべてのstateとrefをフックルールに従って最上部に配置
  const [file, setFile] = useState<File | null>(null)
  const [imgUrl, setImgUrl] = useState<string>('')
  const [bg, setBg] = useState<BgOption>('white')            // デフォルト背景を白に設定
  const [weather, setWeather] = useState<WeatherOption>('sunny') // デフォルト天気を晴れに設定
  const [aspectRatio, setAspectRatio] = useState<AspectRatioOption>('square') // デフォルトは正方形
  const [originalSize, setOriginalSize] = useState<{width: number, height: number} | null>(null)
  const [modalImage, setModalImage] = useState<string | null>(null)
  const [appState, setAppState] = useState<AppState>({ phase: 'IDLE', jobId: 0 })
  const [imageKey, setImageKey] = useState('')               // 新しい画像で無効化
  const [showLineBrowserDialog, setShowLineBrowserDialog] = useState(false)
  const [canShare, setCanShare] = useState(false)           // Web Share API対応チェック

  // ErrorModal用のstate
  const [errorModal, setErrorModal] = useState<{
    isOpen: boolean
    errorType: ErrorType
    requestId?: string
    creditConsumed?: boolean
    customMessage?: string
  }>({
    isOpen: false,
    errorType: 'SERVER_ERROR'
  })

  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const imgRef = useRef<HTMLImageElement | null>(null)
  const outputSectionRef = useRef<HTMLDivElement | null>(null)

  // デバッグ用クエリパラメータ取得
  const debugStatus = router.query.debugStatus as string | undefined
  const debugDelay = router.query.debugDelay ? parseInt(router.query.debugDelay as string) : undefined
  const debugOkButNoImage = router.query.debugOkButNoImage === '1'

  // ★追加：デバッグ時だけ認証ガードを止める
  const bypassAuthRedirect = process.env.NODE_ENV === 'development' && (
    !!debugStatus || !!debugDelay || debugOkButNoImage
  )

  // 認証状態チェック
  useEffect(() => {
    if (!router.isReady) return
    if (bypassAuthRedirect) return

    // ログイン状態をチェック（authLoading=認証確認中、loading=データ取得中）
    // 両方が完了してからユーザーがいなければログインページへ
    if (!authLoading && !loading && !user) {
      router.push('/login')
    }
  }, [router.isReady, bypassAuthRedirect, user, authLoading, loading, router])

  // LINE ブラウザ検出は LineGuard コンポーネントで対応済み

  // Web Share API対応チェック（モバイルでのみ共有ボタンを表示）
  useEffect(() => {
    setCanShare(
      typeof navigator !== 'undefined' &&
      navigator.share !== undefined &&
      navigator.canShare !== undefined
    )
  }, [])

  useEffect(() => {
    if (!file) return
    const url = URL.createObjectURL(file)
    setImgUrl(url)
    setImageKey(String(Date.now()))

    // 元画像のサイズを取得
    const img = new Image()
    img.onload = () => {
      setOriginalSize({ width: img.naturalWidth, height: img.naturalHeight })
    }
    img.src = url

    // IDLE状態に設定（完全リセット）
    setAppState(prev => ({ ...prev, phase: 'IDLE', request: undefined, finalImageUrl: undefined }))

    return () => URL.revokeObjectURL(url)
  }, [file])

  // 司令塔useEffect - AI画像生成処理（Step2: jobId ベースイベント駆動）
  useEffect(() => {
    let cancelled = false

    const handleStateTransition = async () => {
      // Phase と request の存在チェック
      if (appState.phase !== 'FINAL_RENDERING') return
      if (!appState.request) return
      if (appState.status) return // 既に実行中なら早期リターン

      const { imgUrl: reqImgUrl, bg: reqBg, weather: reqWeather, aspectRatio: reqAspectRatio, originalSize: reqOriginalSize } = appState.request

      if (cancelled) return
      setAppState(prev => ({ ...prev, status: 'loading' }))

      // AI画像生成（nano banana）- requestスナップショットを使用
      const backgroundColor = getBackgroundColor(reqBg)
      // Convert imgUrl (blob) to base64 for nano banana
      const img = await loadImage(reqImgUrl)
      if (!img) {
        if (cancelled) return
        setAppState(prev => ({ ...prev, phase: 'IDLE', status: undefined, request: undefined }))
        setErrorModal({
          isOpen: true,
          errorType: 'INVALID_INPUT',
          creditConsumed: false,
          customMessage: '画像の読み込みに失敗しました。別の画像でもう一度試してみてください。'
        })
        return
      }
      const base64 = await toBase64Resized(img, 1536)
      const result = await generateStyledImage(base64, backgroundColor, reqWeather, reqAspectRatio, reqOriginalSize, {
        debugStatus,
        debugDelay,
        debugOkButNoImage
      })

      // キャンセルチェック（非同期処理後）
      if (cancelled) return

      // エラーハンドリング
      if (!result.success || !result.imageBase64) {
        setAppState(prev => ({ ...prev, phase: 'IDLE', status: undefined, request: undefined }))
        setErrorModal({
          isOpen: true,
          errorType: result.error?.type || 'SERVER_ERROR',
          requestId: result.error?.requestId,
          creditConsumed: result.error?.creditConsumed,
          customMessage: result.error?.message
        })
        return
      }

      // 生成成功 - FINAL_READY へ遷移
      setAppState(prev => ({ ...prev, phase: 'FINAL_READY', finalImageUrl: result.imageBase64, status: undefined }))

      // クレジット残高を再取得してNavBarを更新
      await refreshUserData()

      // 生成完了後、Step4（出力エリア）にスムーズスクロール
      setTimeout(() => {
        outputSectionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
      }, 100)
    }

    handleStateTransition()

    // クリーンアップ関数で cancelled フラグを立てる
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [appState.jobId])

  // Canvas描画useEffect - finalImageUrlをCanvasに描画
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !appState.finalImageUrl) return

    const img = new Image()
    img.onload = () => {
      canvas.width = img.width
      canvas.height = img.height
      const ctx = canvas.getContext('2d')
      if (ctx) {
        ctx.drawImage(img, 0, 0)
      }
    }
    img.src = appState.finalImageUrl
  }, [appState.finalImageUrl])

  // 初回認証チェック中は何も表示しない
  if (authLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-pink-50 via-cream-50 to-orange-50">
        <div className="text-gray-600">読み込み中...</div>
      </div>
    )
  }

  // 未ログインならnullを返す（ただしデバッグ時は表示してOK）
  if (!user && !bypassAuthRedirect) {
    return null
  }

  const onSelectFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (f) {
      // 生成中または完了状態で画像を選び直す場合は確認アラート
      if (appState.phase === 'FINAL_RENDERING' || appState.phase === 'FINAL_READY') {
        const confirmed = window.confirm('現在の変更が削除されます！')
        if (!confirmed) {
          // ファイル選択をリセット
          e.target.value = ''
          return
        }
      }

      setFile(f)
      // 画像選択時の状態クリアはuseEffectで自動処理される
    }
  }

  const getBackgroundColor = (bg: BgOption): string => {
    switch (bg) {
      case 'white': return '#FFFFFF'
      case 'linen': return '#F4EDE4'
      case 'concrete': return '#FAF9F6' // コンクリート背景の基調色
      case 'wood': return '#D2B48C' // 木目ナチュラル
      case 'white_wood': return '#F5F5DC' // 木目ホワイト
      default: return '#FFFFFF'
    }
  }


  const handleBgPreset = async (next: BgOption) => {
    // 最終画像生成済みの場合は確認ダイアログを表示
    if (appState.phase === 'FINAL_READY') {
      const confirmed = window.confirm('最終画像をクリアします！')
      if (!confirmed) {
        return // 変更をキャンセル
      }
      // IDLE状態に戻す
      setAppState(prev => ({ ...prev, phase: 'IDLE', request: undefined, finalImageUrl: undefined }))
    }

    setBg(next)
  }

  const handleWeatherPreset = async (next: WeatherOption) => {
    // 最終画像生成済みの場合は確認ダイアログを表示
    if (appState.phase === 'FINAL_READY') {
      const confirmed = window.confirm('最終画像をクリアします！')
      if (!confirmed) {
        return // 変更をキャンセル
      }
      // IDLE状態に戻す
      setAppState(prev => ({ ...prev, phase: 'IDLE', request: undefined, finalImageUrl: undefined }))
    }

    setWeather(next)
  }


  const handleGenerateFinal = async () => {
    if (!bg || !weather || !imgUrl || !originalSize) {
      return
    }

    // 背景IDの取得（bg から selectedBgId へのマッピング）
    const bgIdMap: Record<BgOption, number> = {
      white: 0,
      linen: 1,
      concrete: 2,
      wood: 3,
      white_wood: 4,
    }

    // 生成リクエストのスナップショットを作成
    const request: GenerationRequest = {
      imgUrl,
      bg,
      weather,
      aspectRatio,
      originalSize,
      selectedBgId: bgIdMap[bg],
      startedAt: Date.now(),
    }

    // 最終レンダリング開始 + jobIdをインクリメント
    setAppState(prev => ({
      ...prev,
      phase: 'FINAL_RENDERING',
      request,
      jobId: prev.jobId + 1,
    }))
  }

  const handleSave = async () => {
    const canvas = canvasRef.current
    if (!canvas) return

    // LINE ブラウザの場合は警告ダイアログを表示
    const userAgent = navigator.userAgent.toLowerCase()
    const isLineBrowser = userAgent.includes('line/') ||
                         userAgent.includes('line ') ||
                         userAgent.includes('linelite') ||
                         userAgent.includes('line_app')
    if (isLineBrowser) {
      setShowLineBrowserDialog(true)
      return
    }

    const url = canvas.toDataURL('image/png')
    const a = document.createElement('a')
    a.href = url; a.download = 'okibae.png'; a.click()
  }

  const handleShare = async () => {
    const canvas = canvasRef.current
    if (!canvas) return

    try {
      // Canvas を Blob に変換
      canvas.toBlob(async (blob) => {
        if (!blob) return

        // WebShare API が利用可能かチェック
        if (navigator.share && navigator.canShare) {
          const file = new File([blob], 'okibae.png', { type: 'image/png' })
          const shareData = {
            title: 'OKIBAE - 置き画',
            text: 'OKIBAEで作成した置き画です✨',
            files: [file]
          }

          // ファイル共有がサポートされているかチェック
          if (navigator.canShare(shareData)) {
            await navigator.share(shareData)
            return
          }
        }

        // フォールバック：ダウンロードに変更
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = 'okibae.png'
        a.click()
        URL.revokeObjectURL(url)
      }, 'image/png')
    } catch (error) {
      console.error('Share failed:', error)
    }
  }

  // PWAインストールバナーのハンドラー
  const handleInstallClick = async () => {
    if (isIOS) {
      // iOSの場合は説明ページに遷移
      router.push('/install-guide')
    } else if (isAndroid && canPrompt) {
      // Androidの場合はネイティブプロンプト表示
      await promptInstall()
    }
  }


  return (
    <div className="main-container">
      <div className="mb-6 relative">
        <div className="flex items-center justify-center gap-3 animate-fade-in">
          <img
            src="/okibae-icon.svg"
            alt="OKIBAE"
            className="h-10 w-10"
          />
          <h1 className="typography-main-title">OKIBAE</h1>
        </div>
        <p className="typography-subtitle mt-1 animate-slide-up text-center">おしゃれな置き画を、かんたんに</p>
      </div>

      {/* PWAインストールバナー */}
      {!isInstalled && (isIOS || (isAndroid && canPrompt)) && (
        <div className="max-w-2xl mx-auto px-8 mb-4">
          <button
            onClick={handleInstallClick}
            className="w-full bg-gradient-to-r from-brand-400 to-brand-500 text-white rounded-2xl p-4 shadow-soft hover:shadow-lg transition-all duration-300 flex items-center justify-between group"
          >
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-white/20 rounded-xl flex items-center justify-center">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
              </div>
              <div className="text-left">
                <div className="font-semibold">ホーム画面に追加</div>
                <div className="text-sm text-white/80">アプリのように使えます</div>
              </div>
            </div>
            <svg className="w-6 h-6 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </button>
        </div>
      )}

      <div className="max-w-2xl mx-auto px-8 py-8">
        <div className="space-y-6">

          <StepCard stepNumber={1} title="画像を選ぶ" className="animate-slide-up">
            {/* 撮影のコツ */}
            <div className="mb-4 p-3 bg-gray-100 border border-gray-200 rounded-lg">
              <ul className="text-xs text-gray-500 space-y-1">
                <li>• 商品がはっきりわかるように撮る</li>
                <li>• 平面で撮る</li>
                <li>• うまくいかないときは、撮る場所を変えてみる</li>
              </ul>
            </div>

            <label className={clsx(
              "btn btn-ghost cursor-pointer",
              appState.phase === 'FINAL_RENDERING' && "opacity-50 pointer-events-none cursor-not-allowed"
            )}>
              <input
                type="file"
                accept="image/*"
                className="hidden"
                onChange={onSelectFile}
                disabled={appState.phase === 'FINAL_RENDERING'}
              />
              画像を選ぶ
            </label>

            {imgUrl && (
              <div className="mt-4">
                <div className="aspect-square w-full overflow-hidden rounded-xl border bg-gray-50">
                  <div className="relative h-full w-full">
                    <img
                      ref={imgRef}
                      src={imgUrl}
                      alt="selected image"
                      className="h-full w-full object-contain"
                    />
                    {appState.status === 'loading' && appState.phase === 'IDLE' && (
                      <div className="absolute inset-0 bg-black/20 grid place-content-center">
                        <div className="bg-white/95 px-6 py-3 rounded-lg text-sm text-gray-700 flex items-center gap-2">
                          <div className="w-4 h-4 border-2 border-brand-500 border-t-transparent rounded-full animate-spin"></div>
                          背景除去中...
                        </div>
                      </div>
                    )}
                    {appState.status === 'error' && (
                      <div className="absolute inset-0 bg-red-500/20 grid place-content-center">
                        <div className="bg-white/95 px-6 py-3 rounded-lg text-sm text-red-700">
                          エラー: {appState.error}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </StepCard>

          <StepCard stepNumber={2} title="背景を選ぶ" className="animate-slide-up">
            {/* 説明 */}
            <div className="mb-4 p-3 bg-gray-100 border border-gray-200 rounded-lg">
              <ul className="text-xs text-gray-500 space-y-1">
                <li>• クリックまたはタップで拡大できます</li>
              </ul>
            </div>

            <div className="mb-4">
              <BackgroundCarousel current={bg} onChange={handleBgPreset} setModalImage={setModalImage} disabled={appState.phase === 'FINAL_RENDERING'} />
            </div>
          </StepCard>

          <StepCard stepNumber={3} title="天気を選ぶ" className="animate-slide-up">
            {/* 説明 */}
            <div className="mb-4 p-3 bg-gray-100 border border-gray-200 rounded-lg">
              <ul className="text-xs text-gray-500 space-y-1">
                <li>• 光の当たり方や空気感に影響します</li>
              </ul>
            </div>

            <div className="mb-4 flex flex-col sm:flex-row items-center sm:items-start sm:justify-start justify-center gap-3 sm:gap-2">
              <WeatherBadge current={weather} value="sunny" label="晴れ" color="sunny" onClick={handleWeatherPreset} disabled={appState.phase === 'FINAL_RENDERING'} />
              <WeatherBadge current={weather} value="cloudy" label="くもり" color="cloudy" onClick={handleWeatherPreset} disabled={appState.phase === 'FINAL_RENDERING'} />
              <WeatherBadge current={weather} value="rainy" label="雨" color="rainy" onClick={handleWeatherPreset} disabled={appState.phase === 'FINAL_RENDERING'} />
            </div>
          </StepCard>

          <StepCard ref={outputSectionRef} stepNumber={4} title="保存" className="animate-slide-up">
            {/* サイズ選択UI - Vertex AI制限により一時的に非表示
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">出力サイズ</label>
              <div className="flex items-center gap-2 mb-4">
                <AspectRatioBadge current={aspectRatio} value="square" label="正方形 (1:1)" onClick={setAspectRatio} />
                <AspectRatioBadge current={aspectRatio} value="original" label="元のサイズ" onClick={setAspectRatio} disabled={!originalSize} />
              </div>
            </div>
            */}
            <button
              className="btn btn-primary disabled:opacity-50 mb-4"
              onClick={handleGenerateFinal}
              disabled={!imgUrl || appState.phase === 'FINAL_RENDERING' || !userData || userData.credits.balance === 0}
            >
              {(!userData || userData.credits.balance === 0) ? 'クレジットが不足しています' :
               appState.phase === 'FINAL_RENDERING' ? '生成中...' :
               appState.phase === 'FINAL_READY' ? '再生成！' : '生成！'}
            </button>
            <div className="mb-4">
              {appState.phase === 'IDLE' ? (
                <div className="aspect-square w-full border-2 border-dashed border-gray-300 rounded-xl grid place-content-center text-gray-400 text-sm">
                  生成ボタンを押すと最終画像がここに表示されます
                </div>
              ) : appState.phase === 'FINAL_RENDERING' ? (
                <div className="aspect-square w-full border-2 border-dashed border-gray-300 rounded-xl grid place-content-center">
                  <div className="text-center">
                    <div className="w-8 h-8 border-2 border-brand-500 border-t-transparent rounded-full animate-spin mx-auto mb-2"></div>
                    <div className="text-sm text-gray-600">生成中...</div>
                  </div>
                </div>
              ) : (
                <div className="relative">
                  <canvas
                    ref={(el) => {
                      canvasRef.current = el
                      // Canvas要素がマウントされたら即座に描画
                      if (el && appState.finalImageUrl) {
                        const img = new Image()
                        img.onload = () => {
                          el.width = img.width
                          el.height = img.height
                          const ctx = el.getContext('2d')
                          if (ctx) {
                            ctx.drawImage(img, 0, 0)
                          }
                        }
                        img.src = appState.finalImageUrl
                      }
                    }}
                    className="max-w-full border rounded-xl"
                  />
                  <div className="text-sm text-green-600 mt-2 text-right">✓ 生成完了</div>
                </div>
              )}
            </div>

            {appState.phase === 'FINAL_READY' && (
              <div className="flex items-center gap-3">
                <button className="btn btn-ghost whitespace-nowrap" onClick={handleSave}>
                  ダウンロード
                </button>
                {canShare && (
                  <button
                    className="btn btn-ghost p-2 flex items-center gap-2 whitespace-nowrap"
                    onClick={handleShare}
                    title="共有"
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      viewBox="0 0 48 48"
                      className="w-5 h-5"
                    >
                      <defs>
                        <style>{`.cls-1,.cls-2{fill:none;}.cls-2{stroke:currentColor;stroke-linecap:round;stroke-linejoin:round;stroke-width:4px;}`}</style>
                      </defs>
                      <g>
                        <rect className="cls-1" width="48" height="48"/>
                      </g>
                      <g>
                        <polyline className="cls-2" points="6 34.83 6 41.83 42 41.83 42 34.83"/>
                        <line className="cls-2" x1="24" y1="32.82" x2="24" y2="18.82"/>
                        <line className="cls-2" x1="24" y1="9" x2="13" y2="20"/>
                        <line className="cls-2" x1="24" y1="9" x2="35" y2="20"/>
                      </g>
                    </svg>
                    共有
                  </button>
                )}
              </div>
            )}

            {/* フィードバックリンク */}
            <div className="mt-4 text-left">
              <a
                href="https://docs.google.com/forms/d/e/1FAIpQLSf_pkyMpQ0SQXJ--MhNItVSi9LRHW4OBNsUroergJYa396e6w/viewform?usp=header"
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-gray-500 hover:text-gray-700 underline decoration-gray-300 hover:decoration-gray-500 transition-colors duration-200"
              >
                ぜひ感想をお聞かせください！
              </a>
            </div>
          </StepCard>
        </div>
      </div>

      {/* モーダル表示 - 全画面表示のため最上位レベルに配置 */}
      {modalImage && (
        <div
          className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50"
          onClick={() => setModalImage(null)}
        >
          <div
            className="relative max-w-2xl max-h-[80vh] m-4"
            onClick={(e) => e.stopPropagation()}
          >
            <img
              src={modalImage}
              alt="背景プレビュー"
              className="w-full h-full object-contain rounded-lg max-w-full max-h-full"
            />
            <button
              onClick={() => setModalImage(null)}
              className="absolute top-2 right-2 bg-black bg-opacity-50 text-white rounded-full w-8 h-8 flex items-center justify-center hover:bg-opacity-75 transition-colors"
            >
              ✕
            </button>
          </div>
        </div>
      )}

      {/* fetch_failed エラーモーダル */}
      {errorReason === 'fetch_failed' && (
        <ErrorModal
          isOpen={true}
          onClose={() => {}}
          errorType="AUTH_FETCH_FAILED"
          onRetry={refreshUserData}
          onReload={() => window.location.reload()}
        />
      )}

      {/* エラーモーダル */}
      <ErrorModal
        isOpen={errorModal.isOpen}
        onClose={() => setErrorModal({ ...errorModal, isOpen: false })}
        errorType={errorModal.errorType}
        requestId={errorModal.requestId}
        creditConsumed={errorModal.creditConsumed}
        customMessage={errorModal.customMessage}
      />

    </div>
  )
}

// 参考サイトベースの真のカルーセル実装
function BackgroundCarousel({ current, onChange, setModalImage, disabled = false }: { current: BgOption, onChange: (value: BgOption) => void, setModalImage: (image: string | null) => void, disabled?: boolean }) {
  const [currentIndex, setCurrentIndex] = useState(0)
  const [touchStart, setTouchStart] = useState<number | null>(null)
  const [touchEnd, setTouchEnd] = useState<number | null>(null)

  const backgrounds: Array<{ value: BgOption; label: string; image: string }> = [
    { value: 'white', label: '白画用紙', image: '/samples/white.jpg' },
    { value: 'linen', label: '木綿', image: '/samples/cotton.jpg' },
    { value: 'concrete', label: 'コンクリート', image: '/samples/concrete.jpg' },
    { value: 'wood', label: '木目ナチュラル', image: '/samples/wood.jpg' },
    { value: 'white_wood', label: '木目ホワイト', image: '/samples/white_wood.jpg' }
  ]

  // current値からindexを初期化
  useEffect(() => {
    const index = backgrounds.findIndex(bg => bg.value === current)
    if (index !== -1) setCurrentIndex(index)
  }, [current])

  const handleSlideChange = (newIndex: number) => {
    if (disabled) return
    setCurrentIndex(newIndex)
    onChange(backgrounds[newIndex].value)
  }

  const handlePrev = () => {
    const newIndex = currentIndex > 0 ? currentIndex - 1 : backgrounds.length - 1
    handleSlideChange(newIndex)
  }

  const handleNext = () => {
    const newIndex = currentIndex < backgrounds.length - 1 ? currentIndex + 1 : 0
    handleSlideChange(newIndex)
  }

  // スワイプ処理
  const minSwipeDistance = 50

  const onTouchStart = (e: React.TouchEvent) => {
    setTouchEnd(null)
    setTouchStart(e.targetTouches[0].clientX)
  }

  const onTouchMove = (e: React.TouchEvent) => {
    setTouchEnd(e.targetTouches[0].clientX)
  }

  const onTouchEnd = () => {
    if (!touchStart || !touchEnd) return
    const distance = touchStart - touchEnd
    const isLeftSwipe = distance > minSwipeDistance
    const isRightSwipe = distance < -minSwipeDistance

    if (isLeftSwipe) {
      handleNext()
    } else if (isRightSwipe) {
      handlePrev()
    }
  }

  const currentBg = backgrounds[currentIndex]

  return (
    <div className={clsx("slick-container", disabled && "opacity-50 pointer-events-none")}>
      {/* メインカルーセルエリア */}
      <div
        className="slick-carousel"
        onTouchStart={disabled ? undefined : onTouchStart}
        onTouchMove={disabled ? undefined : onTouchMove}
        onTouchEnd={disabled ? undefined : onTouchEnd}
      >
        <div
          className="slick-track"
          style={{
            transform: `translateX(-${currentIndex * 33.333}%)`,
            width: '300%'
          }}
        >
          {backgrounds.map((bg, index) => (
            <div
              key={bg.value}
              className={`slick-slide ${index === currentIndex ? 'slick-center' : ''}`}
            >
              <div className="slick-slide-content">
                <img
                  src={bg.image}
                  alt={bg.label}
                  className="slick-slide-image"
                  onClick={() => {
                    if (index === currentIndex) {
                      // 現在選択中の画像をクリックした場合はモーダル表示
                      setModalImage(bg.image)
                    } else {
                      // 他の画像をクリックした場合は選択変更
                      handleSlideChange(index)
                    }
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ナビゲーション矢印 */}
      <button
        onClick={handlePrev}
        className="slick-arrow slick-prev"
        aria-label="前の背景"
        disabled={disabled}
      >
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
        </svg>
      </button>

      <button
        onClick={handleNext}
        className="slick-arrow slick-next"
        aria-label="次の背景"
        disabled={disabled}
      >
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
      </button>

      {/* ドットインジケーター */}
      <div className="slick-dots">
        {backgrounds.map((_, index) => (
          <button
            key={index}
            onClick={() => handleSlideChange(index)}
            className={`slick-dot ${index === currentIndex ? 'slick-active' : ''}`}
            aria-label={`スライド ${index + 1}`}
            disabled={disabled}
          />
        ))}
      </div>

      {/* ラベル表示 */}
      <div className="slick-label">
        <h3 className="typography-label text-center">{currentBg.label}</h3>
      </div>
    </div>
  )
}

function WeatherBadge({ current, value, label, color, onClick, disabled = false }:{
  current: WeatherOption | null,
  value: WeatherOption,
  label: string,
  color: WeatherOption,
  onClick: (v: WeatherOption)=>void | Promise<void>,
  disabled?: boolean
}) {
  const active = current === value

  const colorStyles = {
    sunny: active ? "text-amber-800" : "text-amber-700 hover:opacity-80",
    cloudy: active ? "text-purple-800" : "text-purple-700 hover:opacity-80",
    rainy: active ? "text-blue-800" : "text-blue-700 hover:opacity-80"
  }

  const backgroundStyles = {
    sunny: active ? { backgroundColor: '#EDBC9D' } : { backgroundColor: '#EDBC9D20', borderColor: '#EDBC9D60' },
    cloudy: active ? { backgroundColor: '#D6C5D5' } : { backgroundColor: '#D6C5D520', borderColor: '#D6C5D560' },
    rainy: active ? { backgroundColor: '#ACC3D6' } : { backgroundColor: '#ACC3D620', borderColor: '#ACC3D660' }
  }

  const renderIcon = () => {
    const iconClass = "w-4 h-4"
    switch (value) {
      case 'sunny':
        return (
          <svg className={iconClass} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeMiterlimit="10">
            <line x1="24" y1="2" x2="24" y2="5"/>
            <line x1="24" y1="43" x2="24" y2="46"/>
            <line x1="46" y1="24" x2="43" y2="24"/>
            <line x1="5" y1="24" x2="2" y2="24"/>
            <line x1="39.56" y1="39.56" x2="37.44" y2="37.44"/>
            <line x1="10.56" y1="10.56" x2="8.44" y2="8.44"/>
            <line x1="8.44" y1="39.56" x2="10.56" y2="37.44"/>
            <line x1="37.44" y1="10.56" x2="39.56" y2="8.44"/>
            <circle cx="24" cy="24" r="11"/>
          </svg>
        )
      case 'cloudy':
        return (
          <svg className={iconClass} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeMiterlimit="10">
            <path d="M36,18a10.19,10.19,0,0,0-2.08.22,11.49,11.49,0,1,0-22.61,3.9A8,8,0,1,0,10,38H36a10,10,0,0,0,0-20Z"/>
          </svg>
        )
      case 'rainy':
        return (
          <svg className={iconClass} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M40,33A16,16,0,0,0,8,33H34.67"/>
            <line x1="24" y1="33" x2="24" y2="42"/>
            <line x1="24" y1="13" x2="24" y2="17"/>
            <line x1="25.09" y1="2" x2="22.91" y2="8"/>
            <line x1="15.09" y1="2" x2="12.91" y2="8"/>
            <line x1="35.09" y1="2" x2="32.91" y2="8"/>
            <path d="M32,42a4,4,0,0,1-8,0"/>
          </svg>
        )
    }
  }

  return (
    <button
      onClick={() => onClick(value)}
      disabled={disabled}
      className={clsx(
        "inline-flex items-center justify-center gap-2 rounded-2xl px-6 py-3 font-medium shadow-soft transition-all duration-300 typography-button hover:shadow-lg border w-full sm:w-auto min-w-[100px]",
        colorStyles[color],
        disabled && "opacity-50 cursor-not-allowed pointer-events-none"
      )}
      style={backgroundStyles[color]}
    >
      {renderIcon()}
      <span>{label}</span>
    </button>
  )
}

function loadImage(url: string): Promise<HTMLImageElement | null> {
  return new Promise((res) => {
    if (!url) return res(null)
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => res(img)
    img.onerror = () => res(null)
    img.src = url
  })
}

async function toBase64Resized(imgEl: HTMLImageElement, maxSide=1536){

  const { naturalWidth:w, naturalHeight:h } = imgEl
  const scale = w>h ? maxSide/w : maxSide/h
  const rw = Math.round(w*scale), rh = Math.round(h*scale)
  const c = document.createElement('canvas'); c.width=rw; c.height=rh
  c.getContext('2d')!.drawImage(imgEl, 0,0,rw,rh)

  // blob URLの場合は元のファイル形式を保持する
  if (imgEl.src.startsWith('blob:')) {
    // blob URLから実際のファイルタイプを取得
    try {
      const response = await fetch(imgEl.src)
      const blob = await response.blob()
      const mimeType = blob.type


      let result
      if (mimeType === 'image/jpeg') {
        result = c.toDataURL('image/jpeg', 0.9)
      } else {
        result = c.toDataURL('image/png')
      }


      return result
    } catch (e) {
      console.warn('Failed to detect image type from blob, defaulting to PNG', e)
      const result = c.toDataURL('image/png')
      return result
    }
  } else {
    const result = c.toDataURL('image/png')
    return result
  }
}


// AI-powered styled image generation using nano banana (Gemini 2.5 Flash Image Preview)
async function generateStyledImage(
  cutoutBase64: string,
  backgroundColor: string,
  weather: WeatherOption,
  aspectRatio: AspectRatioOption,
  originalSize: {width: number, height: number} | null,
  debugOptions?: {
    debugStatus?: string
    debugDelay?: number
    debugOkButNoImage?: boolean
  }
): Promise<{
  success: boolean
  imageBase64?: string
  error?: {
    type: ErrorType
    requestId?: string
    creditConsumed?: boolean
    message?: string
  }
}> {

  try {
    // Convert backgroundColor to style mapping
    const styleMap: Record<string, string> = {
      '#FFFFFF': 'white',
      '#F4EDE4': 'linen',
      '#FAF9F6': 'concrete',
      '#D2B48C': 'wood',
      '#F5F5DC': 'white_wood'
    }
    const style = styleMap[backgroundColor] || 'white'


    // Convert base64 to blob for form data
    const base64Data = cutoutBase64.replace(/^data:image\/[^;]+;base64,/, '')
    const binaryData = atob(base64Data)
    const bytes = new Uint8Array(binaryData.length)

    for (let i = 0; i < binaryData.length; i++) {
      bytes[i] = binaryData.charCodeAt(i)
    }

    const blob = new Blob([bytes], { type: 'image/png' })

    // Create form data
    const formData = new FormData()
    formData.append('file', blob, 'cutout.png')
    formData.append('style', style)
    formData.append('weather', weather)
    formData.append('aspectRatio', aspectRatio)
    if (originalSize) {
      formData.append('originalWidth', originalSize.width.toString())
      formData.append('originalHeight', originalSize.height.toString())
    }

    // デバッグパラメータを追加
    if (debugOptions?.debugStatus) {
      formData.append('debugStatus', debugOptions.debugStatus)
    }
    if (debugOptions?.debugDelay) {
      formData.append('debugDelay', debugOptions.debugDelay.toString())
    }
    if (debugOptions?.debugOkButNoImage) {
      formData.append('debugOkButNoImage', '1')
    }

    // Call our AI styled image API with timeout
    const controller = new AbortController()
    // デバッグ用の遅延がある場合はタイムアウトを延長
    const timeoutMs = debugOptions?.debugDelay ? Math.max(debugOptions.debugDelay + 10000, 120000) : 120000
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs)

    const response = await fetch('/api/ai-shadows', {
      method: 'POST',
      body: formData,
      signal: controller.signal
    })

    clearTimeout(timeoutId)

    // Get requestId from header or body
    const requestIdHeader = response.headers.get('x-request-id')

    if (!response.ok) {
      let errorData
      try {
        errorData = await response.json()
      } catch {
        errorData = { error: 'Unknown error' }
      }

      const requestId = errorData.requestId || requestIdHeader || undefined
      const creditConsumed = errorData.creditConsumed

      // Map HTTP status to ErrorType
      let errorType: ErrorType = 'SERVER_ERROR'
      if (response.status === 401) {
        errorType = 'AUTH_REQUIRED'
      } else if (response.status === 403) {
        errorType = 'INSUFFICIENT_CREDIT'
      } else if (response.status === 400) {
        errorType = 'INVALID_INPUT'
      } else if (response.status === 504) {
        errorType = 'TIMEOUT'
      }

      return {
        success: false,
        error: {
          type: errorType,
          requestId,
          creditConsumed,
          message: errorData.error
        }
      }
    }

    const result = await response.json()

    if (!result.ok) {
      return {
        success: false,
        error: {
          type: 'GENERATION_FAILED',
          requestId: result.requestId || requestIdHeader || undefined,
          creditConsumed: result.creditConsumed,
          message: result.error
        }
      }
    }

    // Return the generated image base64
    return {
      success: true,
      imageBase64: result.imageBase64
    }

  } catch (error: any) {
    console.error('generateStyledImage error:', error.message, error)

    // Handle timeout
    if (error.name === 'AbortError') {
      return {
        success: false,
        error: {
          type: 'TIMEOUT',
          creditConsumed: undefined, // 不明
          message: 'Request timeout'
        }
      }
    }

    // Other errors
    return {
      success: false,
      error: {
        type: 'SERVER_ERROR',
        creditConsumed: false,
        message: error.message
      }
    }
  }
}
