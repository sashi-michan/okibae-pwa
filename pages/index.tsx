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
  // デバッグモード（開発時は制限なし）
  const DEBUG_MODE = process.env.NODE_ENV === 'development'
  
  const [file, setFile] = useState<File | null>(null)
  const [imgUrl, setImgUrl] = useState<string>('')
  const [bg, setBg] = useState<BgOption>('white')            // デフォルト背景を白に設定
  const [weather, setWeather] = useState<WeatherOption>('sunny') // デフォルト天気を晴れに設定
  const [modalImage, setModalImage] = useState<string | null>(null)
  const [appState, setAppState] = useState<AppState>({ phase: 'IDLE' })
  const [imageKey, setImageKey] = useState('')               // 新しい画像で無効化
  const [dailyUsage, setDailyUsage] = useState({ count: 0, date: '' })

  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const imgRef = useRef<HTMLImageElement | null>(null)

  // 日次使用制限の管理
  useEffect(() => {
    const initDailyUsage = () => {
      const today = new Date().toISOString().split('T')[0] // YYYY-MM-DD
      const savedDate = localStorage.getItem('okibae-date')
      const savedCount = localStorage.getItem('okibae-count')
      
      if (savedDate === today && savedCount) {
        // 今日のデータがある場合
        const count = parseInt(savedCount, 10)
        setDailyUsage({ count, date: today })
      } else {
        // 初回または日付が変わった場合はリセット
        localStorage.setItem('okibae-date', today)
        localStorage.setItem('okibae-count', '0')
        setDailyUsage({ count: 0, date: today })
      }
    }
    
    initDailyUsage()
  }, [])

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
      
      switch (appState.phase) {
        case 'IDLE':
          // nano banana使用のため、背景除去は不要
          break
          
        case 'FINAL_RENDERING':
          // AI画像生成 → FINAL_READY
          if (imgUrl && bg && !appState.status) {
            setAppState({ phase: 'FINAL_RENDERING', status: 'loading' })
            // AI画像生成（nano banana）
            const backgroundColor = getBackgroundColor(bg)
            // Convert imgUrl (blob) to base64 for nano banana
            const img = await loadImage(imgUrl)
            if (!img) {
              setAppState({ phase: 'IDLE', status: 'error', error: '画像の読み込みに失敗しました' })
              return
            }
            const base64 = await toBase64Resized(img, 1536)
            const enhancedUrl = await generateStyledImage(base64, backgroundColor, weather)
            
            // nano banana結果をstateに保存してFINAL_READYで描画
            setAppState({ phase: 'FINAL_READY', finalImageUrl: enhancedUrl })
            
            // 使用回数をカウントアップ
            const newCount = dailyUsage.count + 1
            setDailyUsage(prev => ({ ...prev, count: newCount }))
            localStorage.setItem('okibae-count', newCount.toString())
            
            // ナビバー更新のためのイベント発火
            window.dispatchEvent(new Event('okibae-usage-update'))
          }
          break
          
        case 'FINAL_READY':
          // nano banana結果をキャンバスに描画
          if (canvasRef.current && appState.finalImageUrl && !appState.status) {
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
    // 1日5枚制限チェック（デバッグモードは制限なし）
    if (!DEBUG_MODE && dailyUsage.count >= 5) {
      alert('申し訳ございません。1日の生成上限（5枚）に達しました。明日また挑戦してみてください！')
      return
    }
    
    if (!bg || !weather || !imgUrl) {
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
              <BackgroundCarousel current={bg} onChange={handleBgPreset} setModalImage={setModalImage} />
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
                disabled={!imgUrl || appState.phase === 'FINAL_RENDERING' || (!DEBUG_MODE && dailyUsage.count >= 5)}
              >
                {(!DEBUG_MODE && dailyUsage.count >= 5) ? '本日の上限に達しました' : 
                 appState.phase === 'FINAL_RENDERING' ? '生成中...' : '生成！'}
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
    </div>
  )
}

// 参考サイトベースの真のカルーセル実装
function BackgroundCarousel({ current, onChange, setModalImage }: { current: BgOption, onChange: (value: BgOption) => void, setModalImage: (image: string | null) => void }) {
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
async function generateStyledImage(cutoutBase64: string, backgroundColor: string, weather: WeatherOption): Promise<string> {
  
  try {
    // Convert backgroundColor to style mapping
    const styleMap: Record<string, string> = {
      '#FFFFFF': 'white',
      '#F4EDE4': 'linen', 
      '#FAF9F6': 'concrete'
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
    
    
    // Call our AI styled image API
    const response = await fetch('/api/ai-shadows', {
      method: 'POST',
      body: formData
    })
    
    if (!response.ok) {
      const errorText = await response.text()
      console.error('AI styled image API error:', response.status, errorText)
      throw new Error(`AI API failed: ${response.status}`)
    }
    
    const result = await response.json()
    
    if (!result.ok) {
      console.error('AI styled image generation failed:', result.error)
      throw new Error(result.error || 'AI generation failed')
    }
    
    
    // Return the generated image base64
    return result.imageBase64

  } catch (error: any) {
    console.error('generateStyledImage error:', error.message, error)
    // Fallback to original cutout if AI fails
    return cutoutBase64
  }
}
