export type Profile = {
  id: string
  email: string | null
  stripe_customer_id: string | null
  created_at: string
  updated_at: string
}

export type Subscription = {
  user_id: string
  status: 'free' | 'pro'
  stripe_subscription_id: string | null
  current_period_start: string | null
  current_period_end: string | null
  cancel_at_period_end: boolean
  has_launch_coupon: boolean
  created_at: string
  updated_at: string
}

export type Credits = {
  user_id: string
  balance: number
  total_used: number
  last_reset_at: string
  updated_at: string
}

// フロントエンドで使いやすいように統合した型
export type UserData = {
  profile: Profile
  subscription: Subscription
  credits: Credits
}
