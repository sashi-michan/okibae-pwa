import { NextApiRequest, NextApiResponse } from 'next'
import Stripe from 'stripe'
import { createServerClient } from '@supabase/ssr'
import { serialize } from 'cookie'

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY!, {
  apiVersion: '2025-12-15.clover',
})

// クレジット購入設定
const CREDITS_PER_PURCHASE = 10 // 1回の購入で追加されるクレジット数

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    // Supabaseクライアント作成
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

    // ユーザー認証チェック
    const { data: { user }, error: authError } = await supabase.auth.getUser()

    if (authError || !user) {
      return res.status(401).json({ error: 'Unauthorized' })
    }

    // ユーザープロフィール取得
    const { data: profile } = await supabase
      .from('profiles')
      .select('email, stripe_customer_id')
      .eq('id', user.id)
      .single()

    if (!profile) {
      return res.status(404).json({ error: 'Profile not found' })
    }

    // Stripe Customer IDがない場合は作成
    let customerId = profile.stripe_customer_id
    let isNewCustomer = false

    if (!customerId) {
      const customer = await stripe.customers.create({
        email: profile.email,
        metadata: {
          supabase_user_id: user.id,
        },
      })
      customerId = customer.id
      isNewCustomer = true
      // DB保存はWebhookで行う（決済成功時に確実に保存）
    }

    // リクエスト元のURLを取得（Vercel環境で動的に変わるため）
    const protocol = req.headers['x-forwarded-proto'] || 'https'
    const host = req.headers['x-forwarded-host'] || req.headers.host
    const baseUrl = `${protocol}://${host}`

    // Checkout Session作成
    const session = await stripe.checkout.sessions.create({
      customer: customerId,
      payment_method_types: ['card'],
      line_items: [
        {
          price: process.env.STRIPE_CREDITS_PRICE_ID!,
          quantity: 1,
        },
      ],
      mode: 'payment',
      success_url: `${baseUrl}/?payment=success`,
      cancel_url: `${baseUrl}/?payment=canceled`,
      client_reference_id: user.id,
      metadata: {
        user_id: user.id,
        credits: String(CREDITS_PER_PURCHASE), // 追加するクレジット数
        is_new_customer: isNewCustomer ? 'true' : 'false', // デバッグ用
      },
    })

    return res.status(200).json({ url: session.url })
  } catch (error: any) {
    console.error('Stripe checkout error:', error)
    return res.status(500).json({ error: error.message })
  }
}
