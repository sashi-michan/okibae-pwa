import { NextApiRequest, NextApiResponse } from 'next'
import Stripe from 'stripe'
import { createClient } from '@supabase/supabase-js'
import { buffer } from 'micro'

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY!, {
  apiVersion: '2025-12-15.clover',
})

// Service Roleクライアント（RLSをバイパス）
const supabaseAdmin = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

// bodyParser無効化（Stripe署名検証のため生データが必要）
export const config = {
  api: {
    bodyParser: false,
  },
}

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  const buf = await buffer(req)

  // 署名ヘッダ取得（配列の場合は最初の要素）
  const sigHeader = req.headers['stripe-signature']
  const sig = Array.isArray(sigHeader) ? sigHeader[0] : sigHeader

  if (!sig) {
    console.error('Missing stripe-signature header')
    return res.status(400).json({ error: 'Missing signature' })
  }

  let event: Stripe.Event

  try {
    // Stripe署名検証
    event = stripe.webhooks.constructEvent(
      buf,
      sig,
      process.env.STRIPE_WEBHOOK_SECRET!
    )
  } catch (err: any) {
    console.error('Webhook signature verification failed:', err.message)
    return res.status(400).json({ error: `Webhook Error: ${err.message}` })
  }

  // イベント処理
  try {
    switch (event.type) {
      case 'checkout.session.completed': {
        const session = event.data.object as Stripe.Checkout.Session

        console.log('Checkout session completed:', session.id)

        // 決済完了チェック（未払いの場合はスキップ）
        if (session.payment_status !== 'paid') {
          console.log('Session completed but not paid yet:', session.id, session.payment_status)
          return res.status(200).json({ received: true }) // リトライ不要
        }

        const userId = session.metadata?.user_id
        const credits = parseInt(session.metadata?.credits || '0', 10)

        // Stripe Customer IDを取得（session.customerから）
        const stripeCustomerId =
          typeof session.customer === 'string'
            ? session.customer
            : session.customer?.id

        if (!userId) {
          console.error('Missing user_id in metadata for session:', session.id)
          // 実装ミスなのでログのみ（リトライしても治らない）
          return res.status(200).json({ received: true })
        }

        // Customer ID保存（session.customerから取得して常に保存）
        if (stripeCustomerId) {
          const { error: profileError } = await supabaseAdmin
            .from('profiles')
            .update({ stripe_customer_id: stripeCustomerId })
            .eq('id', userId)

          if (profileError) {
            console.error('Failed to update stripe_customer_id:', profileError)
            // 致命的ではないのでログのみ
          } else {
            console.log(`Updated stripe_customer_id for user ${userId}`)
          }
        }

        // クレジット加算（冪等性対応）
        if (credits > 0) {
          const { data: wasProcessed, error: creditError } = await supabaseAdmin.rpc('add_credits', {
            p_user_id: userId,
            p_amount: credits,
            p_stripe_event_id: event.id,
          })

          if (creditError) {
            console.error('Failed to add credits:', creditError)
            return res.status(500).json({ error: 'Failed to add credits' })
          }

          if (wasProcessed) {
            console.log(`Added ${credits} credits to user ${userId} (event: ${event.id})`)
          } else {
            console.log(`Event ${event.id} already processed, skipped credit addition`)
          }
        }

        break
      }

      default:
        console.log(`Unhandled event type: ${event.type}`)
    }

    return res.status(200).json({ received: true })
  } catch (error: any) {
    console.error('Webhook handler error:', error)
    return res.status(500).json({ error: error.message })
  }
}
