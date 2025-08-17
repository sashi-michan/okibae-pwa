import { useEffect, useRef, useState } from 'react'
import clsx from 'clsx'
import Stepper from '../components/Stepper'
import StepCard from '../components/StepCard'

type BgOption = 'white' | 'beige' | 'stripe'

export default function Home() {
  const [file, setFile] = useState<File | null>(null)
  const [imgUrl, setImgUrl] = useState<string>('')
  const [cutoutUrl, setCutoutUrl] = useState<string>('')     // 透過PNG（切り抜き）
  const [bg, setBg] = useState<BgOption | null>(null)
  const [isBusy, setIsBusy] = useState(false)
  const [processing, setProcessing] = useState(false)        // 切り抜き中
  const [previewCompositing, setPreviewCompositing] = useState(false) // 仮合成中
  const [imageKey, setImageKey] = useState('')               // 新しい画像で無効化

  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const imgRef = useRef<HTMLImageElement | null>(null)

  useEffect(() => {
    if (!file) return
    const url = URL.createObjectURL(file)
    setImgUrl(url)
    setCutoutUrl('')
    setImageKey(String(Date.now()))
    return () => URL.revokeObjectURL(url)
  }, [file])

  // cutoutUrl が来たら、今の背景で本合成に描き直す
  useEffect(() => {
    if (cutoutUrl) {
      drawComposite({ useCutout: true, bg });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cutoutUrl]);

  // 背景を切り替えたら、手元にcutoutがあれば本合成／なければ仮合成
  useEffect(() => {
    if (!imgUrl) return;
    drawComposite({ useCutout: !!cutoutUrl, bg });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bg]);

  const onSelectFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (f) setFile(f)
  }

  const handleBgPreset = async (next: BgOption) => {
    setBg(next)
    setPreviewCompositing(true) // 仮合成中表示開始
    
    await drawComposite({ useCutout: false, bg: next }) // 仮合成実行
    setPreviewCompositing(false) // 仮合成完了
    
    // バックグラウンドで背景除去処理続行
    ensureCutout().then(() => drawComposite({ useCutout: true, bg: next }))
  }

  const ensureCutout = async () => {
    if (cutoutUrl || processing || !imgRef.current) return
    setProcessing(true)
    const myKey = imageKey
    try {
      const base64 = await toBase64Resized(imgRef.current, 1536)
      const resp = await fetch('/api/remove-bg', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ imageBase64: base64 })
      })
      const data = await resp.json()
      if (myKey !== imageKey) return
      if (!data.ok) {
        alert('背景消し失敗: ' + (data.error || 'unknown'))
        return
      }
      setCutoutUrl(data.pngBase64)
    } catch (e:any) {
      console.error('remove-bg error:', e.message)
    } finally {
      setProcessing(false)
    }
  }

  const handleSave = async () => {
    await drawComposite({ useCutout: !!cutoutUrl, bg })
    const canvas = canvasRef.current
    if (!canvas) return
    const url = canvas.toDataURL('image/png')
    const a = document.createElement('a')
    a.href = url; a.download = 'okibae.png'; a.click()
  }

  const drawComposite = async ({ useCutout, bg }:{ useCutout:boolean; bg:BgOption }) => {
    if (!canvasRef.current) return
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')!

    const outW = 1200, outH = 1200
    canvas.width = outW; canvas.height = outH

    drawBackground(ctx, outW, outH, bg)

    const src = (useCutout && cutoutUrl) ? cutoutUrl : imgUrl
    if (!src) return
    const img = await loadImage(src)
    if (!img) return

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

    if (useCutout && cutoutUrl) {
      drawContactShadow(ctx, img, x, y, drawW, drawH, 0.22)
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
    } else {
      ctx.filter = 'brightness(1.04) contrast(1.06)'
      ctx.drawImage(img, x, y, drawW, drawH)
      ctx.filter = 'none'
    }
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
          </StepCard>
          
          <StepCard stepNumber={2} title="背景を選んでプレビュー">
            <div className="mb-4 flex items-center gap-2">
              <BgBadge current={bg} value="white" label="白" onClick={handleBgPreset} />
              <BgBadge current={bg} value="beige" label="ベージュ" onClick={handleBgPreset} />
              <BgBadge current={bg} value="stripe" label="布っぽい" onClick={handleBgPreset} />
            </div>
            
            <div className={clsx("aspect-square w-full overflow-hidden rounded-xl border")}
              style={bg === 'stripe'
                ? { backgroundImage: 'repeating-linear-gradient(180deg,#FAF9F6 0px,#FAF9F6 8px,#F2ECE4 8px,#F2ECE4 16px)' }
                : bg === 'white' ? { backgroundColor: '#FFFFFF' } 
                : bg === 'beige' ? { backgroundColor: '#F4EDE4' }
                : { backgroundColor: '#F9F9F9' }
              }>
              {!imgUrl ? (
                <div className="h-full w-full grid place-content-center text-gray-400 text-sm">画像を選ぶとここに表示されます</div>
              ) : !bg ? (
                <div className="relative h-full w-full">
                  <img
                    ref={imgRef}
                    src={imgUrl}
                    alt="preview"
                    className="h-full w-full object-contain"
                  />
                  <div className="absolute inset-0 bg-black/10 grid place-content-center">
                    <div className="bg-white/90 px-4 py-2 rounded-lg text-sm text-gray-700">
                      背景を選ぶと加工されます
                    </div>
                  </div>
                </div>
              ) : previewCompositing ? (
                <div className="relative h-full w-full">
                  <img
                    ref={imgRef}
                    src={imgUrl}
                    alt="preview"
                    className="h-full w-full object-contain"
                  />
                  <div className="absolute inset-0 bg-black/20 grid place-content-center">
                    <div className="bg-white/95 px-6 py-3 rounded-lg text-sm text-gray-700 flex items-center gap-2">
                      <div className="w-4 h-4 border-2 border-brand-500 border-t-transparent rounded-full animate-spin"></div>
                      仮合成中...
                    </div>
                  </div>
                </div>
              ) : (
                <div className="relative h-full w-full">
                  <img
                    ref={imgRef}
                    src={cutoutUrl || imgUrl}
                    alt="preview"
                    className="h-full w-full object-contain"
                    onLoad={() => drawComposite({ useCutout: !!cutoutUrl, bg })}
                  />
                  <div className="absolute top-2 left-2 bg-green-500/90 text-white px-3 py-1 rounded-lg text-xs">
                    {cutoutUrl ? '完成プレビュー' : 'こんな感じになります'}
                  </div>
                  {processing && (
                    <div className="absolute inset-0 bg-black/20 grid place-content-center">
                      <div className="bg-white/95 px-6 py-3 rounded-lg text-sm text-gray-700 flex items-center gap-2">
                        <div className="w-4 h-4 border-2 border-brand-500 border-t-transparent rounded-full animate-spin"></div>
                        背景除去中...
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </StepCard>
          
          <StepCard stepNumber={3} title="保存">
            <div className="mb-4">
              <canvas ref={canvasRef} className="max-w-full border rounded-xl"></canvas>
            </div>
            <button className="btn btn-ghost disabled:opacity-50" onClick={handleSave} disabled={!imgUrl}>
              画像を保存
            </button>
          </StepCard>
        </div>
      </div>
    </div>
  )
}

function BgBadge({ current, value, label, onClick }:{ current: BgOption, value: BgOption, label: string, onClick: (v: BgOption)=>void | Promise<void> }) {
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
  const { naturalWidth:w, naturalHeight:h } = imgEl
  const scale = w>h ? maxSide/w : maxSide/h
  const rw = Math.round(w*scale), rh = Math.round(h*scale)
  const c = document.createElement('canvas'); c.width=rw; c.height=rh
  c.getContext('2d')!.drawImage(imgEl, 0,0,rw,rh)
  return c.toDataURL('image/png')
}
