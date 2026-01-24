import type { NextApiRequest, NextApiResponse } from 'next'
import { createServerSupabaseClient, createServiceRoleClient } from '../../../lib/supabase/server'
import Stripe from 'stripe'

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY || '', {
  apiVersion: '2025-12-15.clover',
})

/**
 * アカウント復活API
 *
 * 退会済みユーザーが再ログインした際に、論理削除を解除して復活させる
 * クレジット残高は0のまま（再登録特典なし）
 */
export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  // POSTメソッドのみ許可
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    // 認証チェック用クライアント（通常のクライアント）
    const supabaseAuth = createServerSupabaseClient(req, res)

    // 認証チェック
    const {
      data: { user },
      error: authError,
    } = await supabaseAuth.auth.getUser()

    if (authError || !user) {
      return res.status(401).json({ error: 'Unauthorized' })
    }

    const userId = user.id

    // データベース更新用クライアント（Service Role、RLSバイパス）
    const supabase = createServiceRoleClient()

    // 1. profilesのdeleted_atをNULLに（論理削除解除）
    const { error: profileError } = await supabase
      .from('profiles')
      .update({
        deleted_at: null,
        updated_at: new Date().toISOString()
      })
      .eq('id', userId)

    if (profileError) {
      console.error('profiles復活エラー:', profileError)
      throw new Error('Failed to restore profile')
    }

    // 2. subscriptionsのdeleted_atをNULLに（論理削除解除）
    const { error: subError } = await supabase
      .from('subscriptions')
      .update({
        deleted_at: null,
        updated_at: new Date().toISOString()
      })
      .eq('user_id', userId)

    if (subError) {
      console.error('subscriptions復活エラー:', subError)
      throw new Error('Failed to restore subscription')
    }

    // 3. creditsは残高0のまま（updated_atのみ更新）
    const { error: creditsError } = await supabase
      .from('credits')
      .update({
        updated_at: new Date().toISOString()
      })
      .eq('user_id', userId)

    if (creditsError) {
      console.error('credits更新エラー:', creditsError)
      // クレジット更新失敗してもアカウント復活は完了
    }

    // 4. Stripe Customer IDを取得
    const { data: profileData } = await supabase
      .from('profiles')
      .select('stripe_customer_id')
      .eq('id', userId)
      .single()

    // 5. Stripe Customerのmetadataを更新（削除マーク解除）
    if (profileData?.stripe_customer_id) {
      try {
        await stripe.customers.update(profileData.stripe_customer_id, {
          metadata: {
            okibae_status: 'active',
            okibae_restored_at: new Date().toISOString(),
          },
        })
      } catch (stripeError) {
        console.error('Stripe更新エラー:', stripeError)
        // Stripe更新失敗してもDB側は復活済みなので続行
      }
    }

    return res.status(200).json({
      success: true,
      message: 'アカウントを復活しました'
    })

  } catch (error) {
    console.error('復活処理エラー:', error)
    return res.status(500).json({
      error: 'Internal server error',
      message: error instanceof Error ? error.message : 'Unknown error'
    })
  }
}
