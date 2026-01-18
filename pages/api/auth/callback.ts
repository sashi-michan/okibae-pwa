import { createServerClient } from '@supabase/ssr'
import { serialize } from 'cookie'
import type { NextApiRequest, NextApiResponse } from 'next'

const isDev = process.env.NODE_ENV === 'development'

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  res.setHeader('Cache-Control', 'no-store')

  const codeParam = req.query.code
  const code = typeof codeParam === 'string' ? codeParam : null

  const rawNext = typeof req.query.next === 'string' ? req.query.next : '/'
  const next = rawNext.startsWith('/') ? rawNext : '/'

  if (!code) {
    if (isDev) console.log('[callback] code: なし')
    return res.redirect('/login?error=no_code')
  }

  if (isDev) console.log('[callback] code: あり')

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
          const existing = res.getHeader('Set-Cookie')
          const existingArr = Array.isArray(existing)
            ? existing
            : existing
              ? [String(existing)]
              : []

          const nextArr = cookiesToSet.map(({ name, value, options }) =>
            serialize(name, value, options)
          )

          res.setHeader('Set-Cookie', [...existingArr, ...nextArr])
        },
      },
    }
  )

  const { error } = await supabase.auth.exchangeCodeForSession(code)

  if (error) {
    if (isDev) console.log('[callback] exchange: 失敗', error.name || error.message)
    return res.redirect('/login?error=auth_failed')
  }

  if (isDev) console.log('[callback] exchange: OK')
  return res.redirect(next)
}
