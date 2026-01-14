import { createContext, useContext, useEffect, useRef, useState } from 'react'
import { User, Session } from '@supabase/supabase-js'
import { supabase } from '../lib/supabase/client'
import type { UserData } from '../types/database'
import { logger } from '../lib/logger'
import { useRouter } from 'next/router'

type AuthContextType = {
  user: User | null
  session: Session | null
  userData: UserData | null
  loading: boolean
  signInWithGoogle: () => Promise<void>
  signOut: () => Promise<void>
  refreshUserData: () => Promise<void>
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [session, setSession] = useState<Session | null>(null)
  const [userData, setUserData] = useState<UserData | null>(null)
  const [loading, setLoading] = useState(true)
  const didInitRef = useRef(false)
  const authFailureHandledRef = useRef(false)
  const router = useRouter()

  // 初期化完了ヘルパー（最初の1回だけ loading を解除）
  const finishInit = () => {
    if (!didInitRef.current) {
      didInitRef.current = true
      setLoading(false)
    }
  }

  // 認証エラー時の共通処理（一元化 & 二重発火防止）
  const handleAuthFailure = async (reason: 'restore_failed' | 'fetch_failed') => {
    // 二重発火防止ガード
    if (authFailureHandledRef.current) {
      logger.warn('handleAuthFailure already called, skipping duplicate execution')
      return
    }
    authFailureHandledRef.current = true

    logger.error(`Auth failure handled: ${reason}`)

    // ローカル状態を即座にクリア（UIの "生きて見える" 状態を防止）
    setUserData(null)
    setUser(null)
    setSession(null)

    // Supabase からログアウト（失敗しても画面遷移は行う）
    try {
      await supabase.auth.signOut()
    } catch (e) {
      logger.error('signOut failed in handleAuthFailure:', e)
    } finally {
      // ログイン画面へリダイレクト（理由をクエリパラメータで渡す）
      router.replace(`/login?reason=${reason}`)
    }
  }

  // リトライヘルパー（3回リトライ、各2秒待機）
  const retryWithDelay = async <T,>(
    fn: () => Promise<T>,
    retries: number = 3,
    delay: number = 2000
  ): Promise<T | null> => {
    for (let i = 0; i < retries; i++) {
      try {
        return await fn()
      } catch (error) {
        logger.warn(`Retry attempt ${i + 1}/${retries} failed:`, error)
        if (i < retries - 1) {
          await new Promise(resolve => setTimeout(resolve, delay))
        } else {
          logger.error(`Failed after ${retries} retries:`, error)
          return null
        }
      }
    }
    return null
  }

  // Result型で成否を明示的に返す
  type FetchResult =
    | { success: true; data: UserData }
    | { success: false; reason: 'restore_failed' | 'fetch_failed' }

  // ユーザーデータ取得関数（復活処理の再帰は1回まで）
  const fetchUserData = async (userId: string, didRestore = false): Promise<FetchResult> => {
    // Step 1: データ取得（リトライあり）
    const result = await retryWithDelay(async () => {
      const [profileRes, subscriptionRes, creditsRes] = await Promise.all([
        supabase.from('profiles').select('*').eq('id', userId).single(),
        supabase.from('subscriptions').select('*').eq('user_id', userId).single(),
        supabase.from('credits').select('*').eq('user_id', userId).single(),
      ])

      if (profileRes.error || subscriptionRes.error || creditsRes.error) {
        logger.error('Failed to fetch user data:', {
          profile: profileRes.error,
          subscription: subscriptionRes.error,
          credits: creditsRes.error,
        })
        throw new Error('Failed to fetch user data')
      }

      return {
        profile: profileRes.data,
        subscription: subscriptionRes.data,
        credits: creditsRes.data,
      }
    })

    if (!result) {
      logger.error('ユーザーデータ取得に失敗しました（リトライ上限）。', { userId })
      return { success: false, reason: 'fetch_failed' as const }
    }

    // Step 2: 退会済みユーザーチェック → 復活処理（1回まで）
    if (result.profile?.deleted_at) {
      if (didRestore) {
        // 復活処理後もまだ deleted_at が残っている場合はエラー
        logger.error('復活処理後も deleted_at が残っています。', { userId })
        return { success: false, reason: 'restore_failed' as const }
      }

      logger.info('退会済みアカウントを検出しました。復活処理を実行します。', { userId })
      const restoreResult = await retryWithDelay(async () => {
        const restoreResponse = await fetch('/api/account/restore', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
        })

        if (!restoreResponse.ok) {
          throw new Error('復活処理に失敗しました')
        }
        return true
      })

      if (!restoreResult) {
        logger.error('復活処理に失敗しました（リトライ上限）。', { userId })
        return { success: false, reason: 'restore_failed' as const }
      }

      logger.info('アカウント復活成功。ユーザーデータを再取得します。', { userId })
      // 復活後、データを再取得（didRestore=true で再帰は1回まで）
      return await fetchUserData(userId, true)
    }

    return { success: true, data: result }
  }

  useEffect(() => {
    // 初回セッション取得
    supabase.auth.getSession().then(async ({ data: { session } }) => {
      try {
        setSession(session)
        setUser(session?.user ?? null)

        if (session?.user) {
          const result = await fetchUserData(session.user.id)
          if (result.success) {
            setUserData(result.data)
          } else {
            // エラー発生時は handleAuthFailure で一元処理
            await handleAuthFailure(result.reason)
          }
        }
      } finally {
        finishInit()
      }
    })

    // 認証状態の変更を監視
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange(async (event, session) => {
      try {
        // ログイン成功時に authFailureHandledRef をリセット
        if (event === 'SIGNED_IN') {
          authFailureHandledRef.current = false
        }

        setSession(session)
        setUser(session?.user ?? null)

        if (session?.user) {
          const result = await fetchUserData(session.user.id)
          if (result.success) {
            setUserData(result.data)
          } else {
            // エラー発生時は handleAuthFailure で一元処理
            await handleAuthFailure(result.reason)
          }
        } else {
          setUserData(null)
        }
      } finally {
        finishInit()
      }
    })

    return () => subscription.unsubscribe()
  }, [])

  const signInWithGoogle = async () => {
    const { error } = await supabase.auth.signInWithOAuth({
      provider: 'google',
      options: {
        redirectTo: `${window.location.origin}/api/auth/callback?next=/`,
      },
    })
    if (error) throw error
  }

  const signOut = async () => {
    // ローカル状態を先に落とす（一瞬でも「まだログイン中」に見えないように）
    setUser(null)
    setSession(null)
    setUserData(null)

    const result = await retryWithDelay(async () => {
      const { error } = await supabase.auth.signOut()
      if (error) throw error
      return true
    })

    if (!result) {
      logger.error('ログアウトに失敗しました（リトライ上限）。')
      router.replace('/login?reason=signout_failed')
      throw new Error('Sign out failed after retries')
    }
  }

  const refreshUserData = async () => {
    if (user) {
      const result = await fetchUserData(user.id)
      if (result.success) {
        setUserData(result.data)
      } else {
        // エラー発生時は handleAuthFailure で一元処理
        await handleAuthFailure(result.reason)
      }
    }
  }

  return (
    <AuthContext.Provider
      value={{
        user,
        session,
        userData,
        loading,
        signInWithGoogle,
        signOut,
        refreshUserData,
      }}
    >
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}
