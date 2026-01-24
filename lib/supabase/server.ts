import { createServerClient } from '@supabase/ssr'
import { createClient } from '@supabase/supabase-js'
import { serialize } from 'cookie'
import type { NextApiRequest, NextApiResponse } from 'next'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
const supabaseServiceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY!

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
        } catch (error) {
          // ヘッダー送信済みエラーは無視
        }
      },
    },
  })
}

/**
 * Service Roleクライアント（RLSバイパス用）
 * アカウント削除など、管理者権限が必要な操作に使用
 */
export function createServiceRoleClient() {
  return createClient(supabaseUrl, supabaseServiceRoleKey, {
    auth: {
      autoRefreshToken: false,
      persistSession: false
    }
  })
}
