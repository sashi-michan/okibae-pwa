// pages/api/remove-bg.ts
import type { NextApiRequest, NextApiResponse } from 'next'

type Res = { ok: true; pngBase64: string } | { ok: false; error: string }
export const config = { api: { bodyParser: { sizeLimit: '10mb' } } }

export default async function handler(req: NextApiRequest, res: NextApiResponse<Res>) {
  if (req.method !== 'POST') return res.status(405).json({ ok: false, error: 'Method not allowed' })

  try {
    const { imageBase64 } = req.body as { imageBase64: string }
    if (!imageBase64) return res.status(400).json({ ok: false, error: 'no image' })

    const provider = (process.env.BG_PROVIDER || 'replicate').toLowerCase()
    let pngBase64: string

    if (provider === 'removebg') {
      pngBase64 = await callRemoveBg(imageBase64)
    } else {
      pngBase64 = await callReplicate(imageBase64)
    }

    return res.status(200).json({ ok: true, pngBase64 })
  } catch (e: any) {
    return res.status(500).json({ ok: false, error: e?.message || 'unknown error' })
  }
}

// ---- providers ----

// Replicate（851-labs/background-remover の例）
async function callReplicate(imageBase64: string): Promise<string> {
  const token = process.env.REPLICATE_API_TOKEN
  if (!token) throw new Error('server token missing (REPLICATE_API_TOKEN)')

  const run = await fetch('https://api.replicate.com/v1/predictions', {
    method: 'POST',
    headers: { 'Authorization': `Token ${token}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({
      version: 'a029dff38972b5fda4ec5d75d7d1cd25aeff621d2cf4946a41055d7db66b80bc',
      input: { image: imageBase64, format: 'png', background_type: 'rgba' }
    })
  })
  const runJson = await run.json()
  if (!run.ok) throw new Error(runJson?.detail || 'replicate error')

  // poll
  let prediction = runJson
  while (prediction.status && prediction.status !== 'succeeded' && prediction.status !== 'failed') {
    await new Promise(r => setTimeout(r, 1200))
    const p = await fetch(`https://api.replicate.com/v1/predictions/${prediction.id}`, {
      headers: { 'Authorization': `Token ${token}` }
    })
    prediction = await p.json()
  }
  if (prediction.status !== 'succeeded') throw new Error('prediction failed')

  const outUrl = Array.isArray(prediction.output) ? prediction.output[0] : prediction.output
  const imgResp = await fetch(outUrl)
  const buf = await imgResp.arrayBuffer()
  return `data:image/png;base64,${Buffer.from(buf).toString('base64')}`
}

// remove.bg
async function callRemoveBg(imageBase64: string): Promise<string> {
  const key = process.env.REMOVE_BG_API_KEY
  if (!key) throw new Error('server token missing (REMOVE_BG_API_KEY)')

  // data URL → 生のbase64だけに
  const base64 = imageBase64.startsWith('data:')
    ? (imageBase64.split(',')[1] ?? '')
    : imageBase64

  // デバッグログ追加
  console.log('callRemoveBg input:', {
    isDataUrl: imageBase64.startsWith('data:'),
    inputLength: imageBase64.length,
    base64Length: base64.length,
    base64Preview: base64.substring(0, 50) + '...'
  })

  if (!base64 || base64.length === 0) {
    throw new Error('Empty base64 data')
  }

  const form = new FormData()
  form.append('image_file_b64', base64) // ← ここポイント！prefixは付けない
  form.append('format', 'png')          // 出力形式（入力の形式とは無関係）

  const resp = await fetch('https://api.remove.bg/v1.0/removebg', {
    method: 'POST',
    headers: { 'X-Api-Key': key },
    body: form,
  })

  if (!resp.ok) {
    const text = await resp.text()
    throw new Error(`remove.bg error: ${resp.status} ${text}`)
  }
  const buf = await resp.arrayBuffer()
  return `data:image/png;base64,${Buffer.from(buf).toString('base64')}`
}