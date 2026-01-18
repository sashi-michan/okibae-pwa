import type { NextApiRequest, NextApiResponse } from 'next'
import { createServerSupabaseClient, createServiceRoleClient } from '../../../lib/supabase/server'
import Stripe from 'stripe'

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY || '', {
  apiVersion: '2024-12-18.acacia',
})

/**
 * アカウント削除API（退会処理）
 *
 * 処理内容：
 * 1. profiles.deleted_at を設定（論理削除）
 * 2. credits.balance を 0 に設定
 * 3. subscriptions.deleted_at を設定（論理削除）
 * 4. Stripe Customerのmetadataを更新（okibae_status=deleted）
 * 5. ユーザーをログアウト
 *
 * 注意：trial_grantsは削除しない（再登録時の重複防止のため）
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

    // トランザクション開始（Supabaseはトランザクション未対応のため、順次実行）

    // 1. profilesテーブルの論理削除
    const { error: profileError } = await supabase
      .from('profiles')
      .update({ deleted_at: new Date().toISOString() })
      .eq('id', userId)

    if (profileError) {
      console.error('profiles更新エラー:', profileError)
      throw new Error('Failed to delete profile')
    }

    // 2. creditsの残高を0に
    const { error: creditsError } = await supabase
      .from('credits')
      .update({
        balance: 0,
        updated_at: new Date().toISOString()
      })
      .eq('user_id', userId)

    if (creditsError) {
      console.error('credits更新エラー:', creditsError)
      throw new Error('Failed to update credits')
    }

    // 3. subscriptionsテーブルの論理削除
    const { error: subError } = await supabase
      .from('subscriptions')
      .update({ deleted_at: new Date().toISOString() })
      .eq('user_id', userId)

    if (subError) {
      console.error('subscriptions更新エラー:', subError)
      throw new Error('Failed to delete subscription')
    }

    // 4. Stripe Customer IDを取得
    const { data: profileData, error: fetchError } = await supabase
      .from('profiles')
      .select('stripe_customer_id')
      .eq('id', userId)
      .single()

    if (fetchError) {
      console.error('profile取得エラー:', fetchError)
      // Stripe更新失敗してもDB側は削除済みなので続行
    }

    // 5. Stripe Customerのmetadataを更新（削除マーク）
    if (profileData?.stripe_customer_id) {
      try {
        await stripe.customers.update(profileData.stripe_customer_id, {
          metadata: {
            okibae_status: 'deleted',
            okibae_deleted_at: new Date().toISOString(),
          },
        })
      } catch (stripeError) {
        console.error('Stripe更新エラー:', stripeError)
        // Stripe更新失敗してもDB側は削除済みなので続行
      }
    }

    // 6. ユーザーをログアウト（認証用クライアントを使用）
    const { error: signOutError } = await supabaseAuth.auth.signOut()

    if (signOutError) {
      console.error('サインアウトエラー:', signOutError)
      // サインアウト失敗してもDB側は削除済みなので成功扱い
    }

    return res.status(200).json({
      success: true,
      message: '退会処理が完了しました'
    })

  } catch (error) {
    console.error('退会処理エラー:', error)
    return res.status(500).json({
      error: 'Internal server error',
      message: error instanceof Error ? error.message : 'Unknown error'
    })
  }
}
