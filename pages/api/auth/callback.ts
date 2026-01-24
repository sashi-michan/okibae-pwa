import { createServerClient } from '@supabase/ssr'
import { serialize } from 'cookie'
import type { NextApiRequest, NextApiResponse } from 'next'

const isDev = process.env.NODE_ENV === 'development'

type CookieToSet = {
  name: string
  value: string
  options: any
}

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  res.setHeader('Cache-Control', 'no-store')

  const codeParam = req.query.code
  const code = typeof codeParam === 'string' ? codeParam : null

  const rawNext = typeof req.query.next === 'string' ? req.query.next : '/'
  const next = rawNext.startsWith('/') ? rawNext : '/'

  if (!code) {
    return res.redirect('/login?error=no_code')
  }

  // ✅ setAll を「貯める」＆「呼ばれたら通知する」仕組み
  let buffered: CookieToSet[] = []
  let setAllCalled = false
  let resolveSetAll!: () => void
  const setAllPromise = new Promise<void>((r) => (resolveSetAll = r))

  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return Object.keys(req.cookies).map((name) => ({
            name,
            value: req.cookies[name] || '',
          }))
        },
        setAll(cookiesToSet) {
          // ここでは res に触らない（遅延で呼ばれても安全にする）
          buffered = cookiesToSet as CookieToSet[]
          setAllCalled = true
          resolveSetAll()
        },
      },
    }
  )

  const { error } = await supabase.auth.exchangeCodeForSession(code)

  if (error) {
    return res.redirect('/login?error=auth_failed')
  }

  // ✅ exchange が返ってきた後に setAll が遅延実行されるケースがあるので待つ
  //    （来なければすぐ進む）
  await Promise.race([
    setAllPromise,
    new Promise<void>((r) => setTimeout(r, 50)), // 50msだけ待つ（十分）
  ])

  // ✅ ここで初めて Set-Cookie を確実に付与してから redirect
  if (setAllCalled && buffered.length > 0 && !res.headersSent) {
    const existing = res.getHeader('Set-Cookie')
    const existingArr = Array.isArray(existing)
      ? existing
      : existing
        ? [String(existing)]
        : []

    const nextArr = buffered.map(({ name, value, options }) =>
      serialize(name, value, {
        path: '/',
        ...options,
        // localhost(http) 対策：dev は secure 落とす
        secure: isDev ? false : (options?.secure ?? true),
        sameSite: options?.sameSite ?? 'lax',
      })
    )

    res.setHeader('Set-Cookie', [...existingArr, ...nextArr])
  } else {
    console.warn('[callback] Not setting cookies:', {
      setAllCalled,
      bufferedLength: buffered.length,
      headersSent: res.headersSent,
    })
  }

  return res.redirect(next)
}
