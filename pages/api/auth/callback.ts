import { createServerClient } from '@supabase/ssr'
import { serialize } from 'cookie'
import type { NextApiRequest, NextApiResponse } from 'next'

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  res.setHeader('Cache-Control', 'no-store')

  const codeParam = req.query.code
  const code = typeof codeParam === 'string' ? codeParam : null

  const rawNext = typeof req.query.next === 'string' ? req.query.next : '/'
  const next = rawNext.startsWith('/') ? rawNext : '/'

  if (!code) return res.redirect('/login?error=no_code')

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
    console.error('exchangeCodeForSession error:', error.message)
    return res.redirect('/login?error=auth_failed')
  }

  return res.redirect(next)
}
