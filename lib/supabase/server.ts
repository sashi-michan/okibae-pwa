import { createServerClient } from '@supabase/ssr'
import { serialize } from 'cookie'
import type { NextApiRequest, NextApiResponse } from 'next'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!

export function createServerSupabaseClient(
  req: NextApiRequest,
  res: NextApiResponse
) {
  return createServerClient(supabaseUrl, supabaseAnonKey, {
    cookies: {
      getAll() {
        return Object.keys(req.cookies).map((name) => ({
          name,
          value: req.cookies[name] || '',
        }))
      },
      setAll(cookiesToSet) {
        try {
          // ここが大事！ serializeを使って、複数のクッキーをまとめて配列でセットする
          res.setHeader(
            'Set-Cookie',
            cookiesToSet.map(({ name, value, options }) =>
              serialize(name, value, options)
            )
          )
        } catch (error) {
          // ヘッダー送信済みエラーは無視
        }
      },
    },
  })
}
