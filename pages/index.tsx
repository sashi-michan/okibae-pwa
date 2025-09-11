import { useEffect, useRef, useState } from 'react'
import clsx from 'clsx'
import StepCard from '../components/StepCard'

type BgOption = 'white' | 'linen' | 'concrete'
type WeatherOption = 'sunny' | 'cloudy' | 'rainy'

type AppState = {
  phase: 'IDLE' | 'FINAL_RENDERING' | 'FINAL_READY'
  status?: 'loading' | 'error'
  error?: string
  finalImageUrl?: string
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null)
  const [imgUrl, setImgUrl] = useState<string>('')
  const [bg, setBg] = useState<BgOption>('white')            // デフォルト背景を白に設定
  const [weather, setWeather] = useState<WeatherOption>('sunny') // デフォルト天気を晴れに設定
  const [appState, setAppState] = useState<AppState>({ phase: 'IDLE' })
  const [imageKey, setImageKey] = useState('')               // 新しい画像で無効化

  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const imgRef = useRef<HTMLImageElement | null>(null)

  useEffect(() => {
    if (!file) return
    const url = URL.createObjectURL(file)
    setImgUrl(url)
    setImageKey(String(Date.now()))
    
    // IDLE状態に設定（完全リセット）
    setAppState({ phase: 'IDLE' })
    
    return () => URL.revokeObjectURL(url)
  }, [file])

  // 司令塔useEffect - 背景除去のみ自動実行
  useEffect(() => {
    const handleStateTransition = async () => {
      console.log('State transition:', { phase: appState.phase, imgUrl: !!imgUrl })
      
      switch (appState.phase) {
        case 'IDLE':
          // nano banana使用のため、背景除去は不要
          break
          
        case 'FINAL_RENDERING':
          // AI影生成 + drawComposite → FINAL_READY
          console.log('FINAL_RENDERING condition check:', {
            imgUrl: !!imgUrl,
            bg: !!bg,
            canvasRef: !!canvasRef.current,
            noStatus: !appState.status,
            currentStatus: appState.status
          })
          
          if (imgUrl && bg && !appState.status) {
            console.log('Starting final rendering with AI shadows')
            setAppState({ phase: 'FINAL_RENDERING', status: 'loading' })
            // TODO: AI影生成実装予定（nano banana）
            const backgroundColor = getBackgroundColor(bg)
            // Convert imgUrl (blob) to base64 for nano banana
            const img = await loadImage(imgUrl)
            if (!img) {
              setAppState({ phase: 'IDLE', status: 'error', error: '画像の読み込みに失敗しました' })
              return
            }
            const base64 = await toBase64Resized(img, 1536)
            const enhancedUrl = await enhanceWithAIShadows(base64, backgroundColor, weather)
            
            // nano banana結果をstateに保存してFINAL_READYで描画
            setAppState({ phase: 'FINAL_READY', finalImageUrl: enhancedUrl })
          }
          break
          
        case 'FINAL_READY':
          // nano banana結果をキャンバスに描画
          if (canvasRef.current && appState.finalImageUrl && !appState.status) {
            console.log('Drawing final image to canvas', { 
              finalImageUrl: appState.finalImageUrl.substring(0, 100) + '...',
              isDataUrl: appState.finalImageUrl.startsWith('data:')
            })
            const canvas = canvasRef.current
            const ctx = canvas.getContext('2d')!
            
            // キャンバスサイズ設定
            const outW = 1200, outH = 1200
            canvas.width = outW; canvas.height = outH
            
            // nano banana結果画像を読み込み
            const img = await loadImage(appState.finalImageUrl)
            if (img) {
              // アスペクト比を保持して中央に配置
              ctx.clearRect(0, 0, outW, outH)
              
              const imgAspect = img.naturalWidth / img.naturalHeight
              const canvasAspect = outW / outH
              
              let drawW, drawH, drawX, drawY
              
              if (imgAspect > canvasAspect) {
                // 横長の画像：幅に合わせる
                drawW = outW
                drawH = outW / imgAspect
                drawX = 0
                drawY = (outH - drawH) / 2
              } else {
                // 縦長の画像：高さに合わせる
                drawW = outH * imgAspect
                drawH = outH
                drawX = (outW - drawW) / 2
                drawY = 0
              }
              
              ctx.drawImage(img, drawX, drawY, drawW, drawH)
              console.log('Nano banana result drawn with aspect ratio preserved', { 
                original: { w: img.naturalWidth, h: img.naturalHeight },
                canvas: { w: outW, h: outH },
                draw: { x: drawX, y: drawY, w: drawW, h: drawH }
              })
            }
          }
          break
      }
    }
    
    handleStateTransition()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [appState.phase, imgUrl])


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
      default: return '#FFFFFF'
    }
  }

  const enhanceWithShadow = async (cutoutBase64: string, backgroundColor: string, updateState: boolean = false) => {
    console.log('enhanceWithShadow called', { backgroundColor, updateState, cutoutLength: cutoutBase64.length })
    try {
      const resp = await fetch('/api/enhance-shadow', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          cutoutImageBase64: cutoutBase64,
          backgroundColor 
        })
      })
      
      console.log('enhance-shadow response status:', resp.status)
      
      if (!resp.ok) {
        const errorText = await resp.text()
        console.error('enhance-shadow HTTP error:', resp.status, errorText)
        return cutoutBase64
      }
      
      const data = await resp.json()
      console.log('enhance-shadow response:', { ok: data.ok, hasImage: !!data.enhancedImageBase64 })
      
      if (!data.ok) {
        console.error('Shadow enhancement failed:', data.error)
        return cutoutBase64 // フォールバック：元の画像を返す
      }
      
      // 状態更新フラグが有効な場合の処理（nano banana使用のため不要）
      
      return data.enhancedImageBase64
    } catch (e: any) {
      console.error('Shadow enhancement error:', e.message, e)
      return cutoutBase64 // フォールバック：元の画像を返す
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
      setAppState({ phase: 'IDLE' })
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
      setAppState({ phase: 'IDLE' })
    }

    setWeather(next)
  }


  const handleGenerateFinal = async () => {
    console.log('handleGenerateFinal called', { phase: appState.phase, bg, weather, imgUrl: !!imgUrl })
    if (!bg || !weather || !imgUrl) {
      console.log('Missing bg, weather, or imgUrl, returning')
      return
    }
    
    // 最終レンダリング開始
    setAppState({ phase: 'FINAL_RENDERING' })
  }

  const handleSave = async () => {
    const canvas = canvasRef.current
    if (!canvas) return
    const url = canvas.toDataURL('image/png')
    const a = document.createElement('a')
    a.href = url; a.download = 'okibae.png'; a.click()
  }

  const drawComposite = async ({ bg, customCutoutUrl }:{ bg:BgOption | null; customCutoutUrl?: string }) => {
    console.log('drawComposite called', { bg, hasCanvas: !!canvasRef.current })
    if (!canvasRef.current || !bg) {
      console.log('Early return from drawComposite')
      return
    }
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')!

    const outW = 1200, outH = 1200
    canvas.width = outW; canvas.height = outH
    console.log('Canvas setup complete', { outW, outH })

    drawBackground(ctx, outW, outH, bg)
    console.log('Background drawn')

    const src = customCutoutUrl || imgUrl
    console.log('Image source determined', { src: src ? 'data:...' : 'null', imgUrl: imgUrl ? 'blob:...' : 'null' })
    if (!src) {
      console.log('No src, returning')
      return
    }
    const img = await loadImage(src)
    if (!img) {
      console.log('Failed to load image')
      return
    }
    console.log('Image loaded successfully', { naturalWidth: img.naturalWidth, naturalHeight: img.naturalHeight })

    // 配置サイズ（縦横どちらにも収まるようにフィット）
    const pad = Math.round(outW * 0.06)
    const maxW = outW - pad * 2
    const maxH = outH - pad * 2

    // 画像の自然サイズ
    const iw = img.naturalWidth
    const ih = img.naturalHeight

    // 縮小率は「入る方」に合わせる
    const scale = Math.min(maxW / iw, maxH / ih)

    const drawW = Math.round(iw * scale)
    const drawH = Math.round(ih * scale)
    const x = Math.round((outW - drawW) / 2)  // 中央寄せ
    const y = Math.round((outH - drawH) / 2)

    if (customCutoutUrl) {
      console.log('Drawing with cutout image (OpenCV shadows)')
      // OpenCV影統一なので自前影は削除済み
      ctx.filter = 'brightness(1.06) contrast(1.08)'
      ctx.drawImage(img, x, y, drawW, drawH)
      ctx.filter = 'none'
      const [r,g,b] = estimateBgAverage(ctx, outW, outH)
      ctx.save()
      ctx.globalCompositeOperation = 'soft-light'
      ctx.fillStyle = `rgba(${r},${g},${b},0.10)`; ctx.fillRect(0,0,outW,outH)
      ctx.globalCompositeOperation = 'multiply'
      ctx.fillStyle = `rgba(${r},${g},${b},0.05)`; ctx.fillRect(0,0,outW,outH)
      ctx.restore()
      console.log('Cutout drawing complete')
    } else {
      console.log('Drawing without cutout')
      ctx.filter = 'brightness(1.04) contrast(1.06)'
      ctx.drawImage(img, x, y, drawW, drawH)
      ctx.filter = 'none'
      console.log('Normal drawing complete')
    }
    console.log('drawComposite function completed')
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
      
      <div className="max-w-2xl mx-auto px-8 py-8">
        <div className="space-y-6">
          
          <StepCard stepNumber={1} title="画像を選ぶ" className="animate-slide-up">
            <label className="btn btn-ghost cursor-pointer">
              <input type="file" accept="image/*" className="hidden" onChange={onSelectFile} />
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
            <div className="mb-4">
              <BackgroundCarousel current={bg} onChange={handleBgPreset} />
            </div>
          </StepCard>
          
          <StepCard stepNumber={3} title="天気を選ぶ" className="animate-slide-up">
            <div className="mb-4 flex items-center gap-2">
              <WeatherBadge current={weather} value="sunny" label="晴れ" color="sunny" onClick={handleWeatherPreset} />
              <WeatherBadge current={weather} value="cloudy" label="くもり" color="cloudy" onClick={handleWeatherPreset} />
              <WeatherBadge current={weather} value="rainy" label="雨" color="rainy" onClick={handleWeatherPreset} />
            </div>
          </StepCard>
          
          <StepCard stepNumber={4} title="保存" className="animate-slide-up">
            {appState.phase !== 'FINAL_READY' ? (
              <button 
                className="btn btn-primary disabled:opacity-50 mb-4" 
                onClick={handleGenerateFinal} 
                disabled={!imgUrl || appState.phase === 'FINAL_RENDERING'}
              >
                {appState.phase === 'FINAL_RENDERING' ? '生成中...' : '生成！'}
              </button>
            ) : (
              <div className="text-sm text-green-600 mb-4">✓ 生成完了</div>
            )}
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
                <canvas 
                  ref={canvasRef} 
                  className="max-w-full border rounded-xl"
                />
              )}
            </div>
            
            {appState.phase === 'FINAL_READY' && (
              <button className="btn btn-ghost" onClick={handleSave}>
                画像をダウンロード
              </button>
            )}
          </StepCard>
        </div>
      </div>
    </div>
  )
}

// 参考サイトベースの真のカルーセル実装
function BackgroundCarousel({ current, onChange }: { current: BgOption, onChange: (value: BgOption) => void }) {
  const [currentIndex, setCurrentIndex] = useState(0)
  const [touchStart, setTouchStart] = useState<number | null>(null)
  const [touchEnd, setTouchEnd] = useState<number | null>(null)
  
  const backgrounds: Array<{ value: BgOption; label: string; image: string }> = [
    { value: 'white', label: '白画用紙', image: '/input_image/sample_white.jpeg' },
    { value: 'linen', label: 'リネン布', image: '/input_image/sample_linen.jpeg' },
    { value: 'concrete', label: 'コンクリート', image: '/input_image/sample_concrete.jpeg' }
  ]

  // current値からindexを初期化
  useEffect(() => {
    const index = backgrounds.findIndex(bg => bg.value === current)
    if (index !== -1) setCurrentIndex(index)
  }, [current])

  const handleSlideChange = (newIndex: number) => {
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
    <div className="slick-container">
      {/* メインカルーセルエリア */}
      <div 
        className="slick-carousel"
        onTouchStart={onTouchStart}
        onTouchMove={onTouchMove}
        onTouchEnd={onTouchEnd}
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
              onClick={() => handleSlideChange(index)}
            >
              <div className="slick-slide-content">
                <img 
                  src={bg.image}
                  alt={bg.label}
                  className="slick-slide-image"
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
      >
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
        </svg>
      </button>

      <button
        onClick={handleNext}
        className="slick-arrow slick-next"
        aria-label="次の背景"
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

function WeatherBadge({ current, value, label, color, onClick }:{ 
  current: WeatherOption | null, 
  value: WeatherOption, 
  label: string, 
  color: WeatherOption,
  onClick: (v: WeatherOption)=>void | Promise<void> 
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
      className={clsx(
        "inline-flex items-center justify-center gap-2 rounded-2xl px-4 py-2 font-medium shadow-soft transition-all duration-300 typography-button hover:shadow-lg border",
        colorStyles[color]
      )}
      style={backgroundStyles[color]}
    >
      {renderIcon()}
      <span>{label}</span>
    </button>
  )
}

function drawBackground(ctx: CanvasRenderingContext2D, w:number, h:number, bg: BgOption) {
  if (bg === 'white') {
    ctx.fillStyle = '#FFFFFF'; ctx.fillRect(0,0,w,h)
  } else if (bg === 'linen') {
    ctx.fillStyle = '#F4EDE4'; ctx.fillRect(0,0,w,h)
  } else {
    const tile = document.createElement('canvas')
    tile.width = 64; tile.height = 64
    const t = tile.getContext('2d')!
    t.fillStyle = '#FAF9F6'; t.fillRect(0,0,64,64)
    t.fillStyle = '#F2ECE4'; t.fillRect(0,0,64,32)
    const noise = t.createImageData(64,64)
    for (let i=0;i<noise.data.length;i+=4){
      const n = 248 + Math.floor(Math.random()*7)
      noise.data[i]=n; noise.data[i+1]=n; noise.data[i+2]=n; noise.data[i+3]=14
    }
    t.putImageData(noise,0,0)
    const pat = ctx.createPattern(tile, 'repeat')!
    ctx.fillStyle = pat; ctx.fillRect(0,0,w,h)
  }
}

function drawContactShadow(
  ctx: CanvasRenderingContext2D, img: HTMLImageElement,
  x:number, y:number, w:number, h:number, strength:number
){
  ctx.save()
  ctx.globalCompositeOperation = 'multiply'
  ctx.globalAlpha = Math.min(0.45, Math.max(0, strength))
  const off = document.createElement('canvas')
  off.width = w; off.height = h
  const o = off.getContext('2d')!
  o.drawImage(img, 0,0,w,h)
  const id = o.getImageData(0,0,w,h)
  for (let i=0;i<id.data.length;i+=4){ id.data[i]=0; id.data[i+1]=0; id.data[i+2]=0 }
  o.putImageData(id,0,0)
  o.filter = `blur(${Math.round(w*0.04)}px)`
  const sx = 1.02, sy = 0.88
  const dx = x + Math.round(w*0.02)
  const dy = y + Math.round(h*0.06)
  ctx.drawImage(off, 0,0,w,h, dx, dy, Math.round(w*sx), Math.round(h*sy))
  ctx.restore()
}

function estimateBgAverage(ctx: CanvasRenderingContext2D, w:number, h:number): [number,number,number] {
  const tmp = document.createElement('canvas')
  tmp.width = 64; tmp.height = 64
  const t = tmp.getContext('2d')!
  t.drawImage(ctx.canvas, 0,0,w,h, 0,0,64,64)
  const data = t.getImageData(0,0,64,64).data
  let r=0,g=0,b=0,c=0
  for (let i=0;i<data.length;i+=4) { r+=data[i]; g+=data[i+1]; b+=data[i+2]; c++ }
  return [Math.round(r/c), Math.round(g/c), Math.round(b/c)]
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
  console.log('toBase64Resized called:', {
    src: imgEl.src.substring(0, 50) + '...',
    naturalWidth: imgEl.naturalWidth,
    naturalHeight: imgEl.naturalHeight,
    complete: imgEl.complete
  })

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
      
      console.log('Detected MIME type from blob:', mimeType)
      
      let result
      if (mimeType === 'image/jpeg') {
        result = c.toDataURL('image/jpeg', 0.9)
      } else {
        result = c.toDataURL('image/png')
      }
      
      console.log('toBase64Resized result:', {
        resultLength: result.length,
        preview: result.substring(0, 50) + '...'
      })
      
      return result
    } catch (e) {
      console.warn('Failed to detect image type from blob, defaulting to PNG', e)
      const result = c.toDataURL('image/png')
      console.log('toBase64Resized fallback result:', result.length)
      return result
    }
  } else {
    const result = c.toDataURL('image/png')
    console.log('toBase64Resized non-blob result:', result.length)
    return result
  }
}

// New OpenCV-based shadow generation function
type ShadowOptions = {
  quality?: 'preview' | 'final'
  placement?: {
    scale?: number
    rotate?: number
    tx?: number
    ty?: number
  }
  directionDeg?: number
  distancePx?: number
  preset?: 'soft' | 'hard' | 'fabric' | 'none'
  blur?: number
  opacity?: number
}

async function generateShadowOpenCV(cutoutBase64: string, options: ShadowOptions = {}, signal?: AbortSignal): Promise<string | null> {
  try {
    console.log('generateShadowOpenCV called with options:', options)
    const response = await fetch('/api/generate-shadow', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        cutoutImageBase64: cutoutBase64,
        options
      }),
      signal
    })

    if (!response.ok) {
      const errorText = await response.text()
      console.error('generate-shadow HTTP error:', response.status, errorText)
      return null
    }

    const data = await response.json()
    
    if (!data.ok) {
      console.error('Shadow generation failed:', data.error)
      return null
    }
    
    // Debug logging
    if (data.debug) {
      console.log('Shadow generation debug:', data.debug)
    }
    if (data.lightSource) {
      console.log('Light source detected:', data.lightSource)
    }

    // Convert base64 to blob URL for consistent usage
    const shadowBase64 = data.shadowLayerBase64
    if (!shadowBase64) {
      console.error('No shadow layer returned')
      return null
    }

    // Convert to blob URL
    const base64Data = shadowBase64.replace(/^data:image\/[^;]+;base64,/, '')
    const binaryData = atob(base64Data)
    const bytes = new Uint8Array(binaryData.length)
    
    for (let i = 0; i < binaryData.length; i++) {
      bytes[i] = binaryData.charCodeAt(i)
    }
    
    const blob = new Blob([bytes], { type: 'image/png' })
    return URL.createObjectURL(blob)

  } catch (error: any) {
    if (signal?.aborted) {
      console.log('Shadow generation aborted')
      return null
    }
    console.error('Shadow generation error:', error.message, error)
    return null
  }
}

// AI-powered shadow generation using nano banana (Gemini 2.5 Flash Image Preview)
async function enhanceWithAIShadows(cutoutBase64: string, backgroundColor: string, weather: WeatherOption): Promise<string> {
  console.log('enhanceWithAIShadows called with nano banana', { backgroundColor, weather, cutoutLength: cutoutBase64.length })
  
  try {
    // Convert backgroundColor to style mapping
    const styleMap: Record<string, string> = {
      '#FFFFFF': 'white',
      '#F4EDE4': 'linen', 
      '#FAF9F6': 'concrete'
    }
    const style = styleMap[backgroundColor] || 'white'
    
    console.log(`Using AI shadow generation with style: ${style}, weather: ${weather}`)
    
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
    
    console.log('Calling nano banana API...')
    
    // Call our AI shadows API
    const response = await fetch('/api/ai-shadows', {
      method: 'POST',
      body: formData
    })
    
    if (!response.ok) {
      const errorText = await response.text()
      console.error('AI shadows API error:', response.status, errorText)
      throw new Error(`AI API failed: ${response.status}`)
    }
    
    const result = await response.json()
    
    if (!result.ok) {
      console.error('AI shadows generation failed:', result.error)
      throw new Error(result.error || 'AI generation failed')
    }
    
    console.log('AI shadows generation successful')
    
    // Return the generated image base64
    return result.imageBase64

  } catch (error: any) {
    console.error('enhanceWithAIShadows error:', error.message, error)
    // Fallback to original cutout if AI fails
    return cutoutBase64
  }
}
