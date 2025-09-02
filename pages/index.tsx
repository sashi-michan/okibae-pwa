import { useEffect, useRef, useState } from 'react'
import clsx from 'clsx'
import Stepper from '../components/Stepper'
import StepCard from '../components/StepCard'

type BgOption = 'white' | 'beige' | 'stripe'

type AppState = {
  phase: 'IDLE' | 'FINAL_RENDERING' | 'FINAL_READY'
  status?: 'loading' | 'error'
  error?: string
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null)
  const [imgUrl, setImgUrl] = useState<string>('')
  const [cutoutUrl, setCutoutUrl] = useState<string>('')     // 透過PNG（切り抜き）
  const [bg, setBg] = useState<BgOption>('white')            // デフォルト背景を白に設定
  const [appState, setAppState] = useState<AppState>({ phase: 'IDLE' })
  const [imageKey, setImageKey] = useState('')               // 新しい画像で無効化

  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const imgRef = useRef<HTMLImageElement | null>(null)

  useEffect(() => {
    if (!file) return
    const url = URL.createObjectURL(file)
    setImgUrl(url)
    setCutoutUrl('')
    setImageKey(String(Date.now()))
    
    // IDLE状態に設定（完全リセット）
    setAppState({ phase: 'IDLE' })
    
    return () => URL.revokeObjectURL(url)
  }, [file])

  // 司令塔useEffect - 背景除去のみ自動実行
  useEffect(() => {
    const handleStateTransition = async () => {
      console.log('State transition:', { phase: appState.phase, cutoutUrl: !!cutoutUrl })
      
      switch (appState.phase) {
        case 'IDLE':
          // 画像選択済み時に背景除去を自動実行
          if (imgUrl && !cutoutUrl && imgRef.current && !appState.status) {
            console.log('Starting background removal')
            setAppState({ phase: 'IDLE', status: 'loading' })
            await ensureCutout()
          }
          break
          
        case 'FINAL_RENDERING':
          // AI影生成 + drawComposite → FINAL_READY
          console.log('FINAL_RENDERING condition check:', {
            cutoutUrl: !!cutoutUrl,
            bg: !!bg,
            canvasRef: !!canvasRef.current,
            noStatus: !appState.status,
            currentStatus: appState.status
          })
          
          if (cutoutUrl && bg && canvasRef.current && !appState.status) {
            console.log('Starting final rendering with AI shadows')
            setAppState({ phase: 'FINAL_RENDERING', status: 'loading' })
            // TODO: AI影生成実装予定（nano banana）
            const backgroundColor = getBackgroundColor(bg)
            const enhancedUrl = await enhanceWithAIShadows(cutoutUrl, backgroundColor)
            await drawComposite({ bg, customCutoutUrl: enhancedUrl })
            setAppState({ phase: 'FINAL_READY' })
          }
          break
          
        case 'FINAL_READY':
          // 完了状態、追加処理なし
          break
      }
    }
    
    handleStateTransition()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [appState.phase, cutoutUrl, imgUrl])


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
      case 'beige': return '#F4EDE4'
      case 'stripe': return '#FAF9F6' // 布っぽい背景の基調色
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
      
      // 状態更新フラグが有効な場合、shadowUrlを更新
      if (updateState) {
        console.log('Updating shadowUrl state')
        setShadowUrl(data.enhancedImageBase64)
      }
      
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

  const ensureCutout = async () => {
    if (cutoutUrl || !imgRef.current) return
    const myKey = imageKey
    
    try {
      // 画像の読み込み完了を待つ
      const img = imgRef.current
      if (!img.complete || img.naturalWidth === 0) {
        console.log('Waiting for image to load...')
        await new Promise<void>((resolve, reject) => {
          const timeout = setTimeout(() => {
            reject(new Error('Image load timeout'))
          }, 10000) // 10秒タイムアウト
          
          img.onload = () => {
            clearTimeout(timeout)
            console.log('Image loaded:', { naturalWidth: img.naturalWidth, naturalHeight: img.naturalHeight })
            resolve()
          }
          img.onerror = () => {
            clearTimeout(timeout)
            reject(new Error('Image load failed'))
          }
          
          // 既に読み込まれている場合
          if (img.complete && img.naturalWidth > 0) {
            clearTimeout(timeout)
            resolve()
          }
        })
      }
      
      const base64 = await toBase64Resized(img, 1536)
      const resp = await fetch('/api/remove-bg', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ imageBase64: base64 })
      })
      const data = await resp.json()
      if (myKey !== imageKey) return
      if (!data.ok) {
        console.error('背景消し失敗:', data.error || 'unknown')
        setAppState({ phase: 'IDLE', status: 'error', error: data.error || '背景除去に失敗しました' })
        return
      }
      setCutoutUrl(data.pngBase64)
      setAppState({ phase: 'IDLE' }) // loading完了
    } catch (e:any) {
      console.error('remove-bg error:', e.message)
      setAppState({ phase: 'IDLE', status: 'error', error: '背景除去でエラーが発生しました' })
    }
  }

  const handleGenerateFinal = async () => {
    console.log('handleGenerateFinal called', { phase: appState.phase, bg, cutoutUrl })
    if (!bg || !cutoutUrl) {
      console.log('Missing bg or cutoutUrl, returning')
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

    const src = customCutoutUrl || cutoutUrl || imgUrl
    console.log('Image source determined', { src: src ? 'data:...' : 'null', cutoutUrl: cutoutUrl ? 'data:...' : 'null', imgUrl: imgUrl ? 'blob:...' : 'null' })
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

    if (cutoutUrl) {
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
    <div>
      <div className="mb-6">
        <h1 className="text-2xl md:text-3xl font-bold tracking-wide">OKIBAE — おしゃれな置き画を、かんたんに</h1>
      </div>
      
      <div className="grid lg:grid-cols-[260px_1fr] gap-6">
        <div className="lg:block hidden">
          <Stepper currentStep={1} />
        </div>
        
        <div className="space-y-6">
          <div className="lg:hidden">
            <Stepper currentStep={1} />
          </div>
          
          <StepCard stepNumber={1} title="画像を選ぶ">
            <p className="mb-4 text-sm text-gray-600">
              まずは画像を1枚えらんでね。
            </p>
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
          
          <StepCard stepNumber={2} title="背景を選んで生成">
            <p className="mb-4 text-sm text-gray-600">
              背景を選んで影付きの画像を生成します。
            </p>
            <div className="mb-4 flex items-center gap-2">
              <BgBadge current={bg} value="white" label="白" onClick={handleBgPreset} />
              <BgBadge current={bg} value="beige" label="ベージュ" onClick={handleBgPreset} />
              <BgBadge current={bg} value="stripe" label="布っぽい" onClick={handleBgPreset} />
            </div>
            
            {appState.phase !== 'FINAL_READY' ? (
              <button 
                className="btn btn-primary disabled:opacity-50" 
                onClick={handleGenerateFinal} 
                disabled={!cutoutUrl || appState.phase === 'FINAL_RENDERING'}
              >
                {appState.phase === 'FINAL_RENDERING' ? '生成中...' : 'AI影付き画像を生成！'}
              </button>
            ) : (
              <div className="text-sm text-green-600">✓ 生成完了</div>
            )}
          </StepCard>
          
          <StepCard stepNumber={3} title="保存">
            <div className="mb-4">
              {appState.phase === 'IDLE' ? (
                <div className="aspect-square w-full border-2 border-dashed border-gray-300 rounded-xl grid place-content-center text-gray-400 text-sm">
                  生成ボタンを押すと最終画像がここに表示されます
                </div>
              ) : appState.phase === 'FINAL_RENDERING' ? (
                <div className="aspect-square w-full border-2 border-dashed border-gray-300 rounded-xl grid place-content-center">
                  <div className="text-center">
                    <div className="w-8 h-8 border-2 border-brand-500 border-t-transparent rounded-full animate-spin mx-auto mb-2"></div>
                    <div className="text-sm text-gray-600">AI影生成中...</div>
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

function BgBadge({ current, value, label, onClick }:{ current: BgOption | null, value: BgOption, label: string, onClick: (v: BgOption)=>void | Promise<void> }) {
  const active = current === value
  return (
    <button
      onClick={() => onClick(value)}
      className={clsx("px-3 py-1 rounded-full text-sm border transition", active ? "bg-brand-500 text-white border-brand-500" : "bg-white hover:bg-gray-50 border-gray-200")}
    >
      {label}
    </button>
  )
}

function drawBackground(ctx: CanvasRenderingContext2D, w:number, h:number, bg: BgOption) {
  if (bg === 'white') {
    ctx.fillStyle = '#FFFFFF'; ctx.fillRect(0,0,w,h)
  } else if (bg === 'beige') {
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

// AI-powered shadow generation using nano banana or similar models
async function enhanceWithAIShadows(cutoutBase64: string, backgroundColor: string): Promise<string> {
  console.log('enhanceWithAIShadows called', { backgroundColor, cutoutLength: cutoutBase64.length })
  
  try {
    // TODO: Implement nano banana AI shadow generation
    // For now, return the original cutout as fallback
    console.log('AI shadow generation - TODO: implement nano banana')
    
    // Placeholder: In the future, this will call nano banana API
    // const aiResult = await fetch('/api/ai-shadows', {
    //   method: 'POST',
    //   headers: { 'Content-Type': 'application/json' },
    //   body: JSON.stringify({
    //     cutoutImageBase64: cutoutBase64,
    //     backgroundColor,
    //     model: 'nano-banana'
    //   })
    // })
    
    return cutoutBase64 // Temporary fallback

  } catch (error: any) {
    console.error('enhanceWithAIShadows error:', error.message, error)
    return cutoutBase64 // Fallback to original
  }
}
