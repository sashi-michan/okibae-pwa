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
  authLoading: boolean
  errorReason: 'fetch_failed' | null
  signInWithGoogle: () => Promise<void>
  signOut: () => Promise<void>
  refreshUserData: () => Promise<void>
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [session, setSession] = useState<Session | null>(null)
  const [userData, setUserData] = useState<UserData | null>(null)
  const [authLoading, setAuthLoading] = useState(true)
  const [dataLoading, setDataLoading] = useState(false)
  const [errorReason, setErrorReason] = useState<'fetch_failed' | null>(null)
  const authFailureHandledRef = useRef(false)
  const fetchingUserIdRef = useRef<string | null>(null) // dedupe用：現在取得中のuserId
  const router = useRouter()

  // 初期化完了ヘルパー（auth確認完了）
  const finishAuthInit = () => {
    setAuthLoading(false)
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
    setAuthLoading(false)
    setDataLoading(false)

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

  // リトライヘルパー（バックオフ: 1s → 2s → 4s）
  const retryWithBackoff = async <T,>(
    fn: () => Promise<T>,
    delays: number[] = [1000, 2000, 4000],
    timeoutMs: number = 8000  // 各試行は8秒でタイムアウト（未使用、タブ復帰時の遅延対策）
  ): Promise<T | null> => {
    for (let i = 0; i < delays.length; i++) {
      try {
        // タブ復帰時のネットワーク遅延対策：タイムアウトなしで待つ
        return await fn()
      } catch (error) {
        logger.warn(`Retry attempt ${i + 1}/${delays.length} failed`, error)
        if (i < delays.length - 1) {
          await new Promise(resolve => setTimeout(resolve, delays[i]))
        } else {
          logger.error('All retry attempts failed')
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
    const result = await retryWithBackoff(async () => {
      const profileRes = await supabase.from('profiles').select('*').eq('id', userId).single()
      const subscriptionRes = await supabase.from('subscriptions').select('*').eq('user_id', userId).single()
      const creditsRes = await supabase.from('credits').select('*').eq('user_id', userId).single()

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

      logger.dev('退会済みアカウントを検出しました。復活処理を実行します。', { userId })
      const restoreResult = await retryWithBackoff(async () => {
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

  // userData を保証する関数（dedupe機能付き）
  const ensureUserData = async (session: Session | null, caller: string) => {
    if (!session?.user) {
      logger.dev(`[ensureUserData from ${caller}] no session, skipping`)
      // getSession 側では何も消さない（onAuthStateChange が先に取得中の可能性）
      // onAuthStateChange 側のみ、明示的にクリア
      if (caller === 'onAuthStateChange') {
        setUserData(null)
      }
      return
    }

    const userId = session.user.id

    // 同じuserIdで既に取得中なら skip（dedupe）
    if (fetchingUserIdRef.current === userId) {
      logger.dev(`[ensureUserData from ${caller}] already fetching userId=${userId}, skipping`)
      return
    }

    // 取得開始
    fetchingUserIdRef.current = userId
    setDataLoading(true)
    logger.dev(`[ensureUserData from ${caller}] userData fetch: start`)

    // 5秒後に警告ログ（取得は継続）
    const slowWarning = setTimeout(() => {
      logger.warn(`[ensureUserData from ${caller}] userData fetch is slow (>5s)`)
    }, 5000)

    // fetchUserData は内部でバックオフリトライ (1s, 2s, 4s) を実行
    const result = await fetchUserData(userId)

    clearTimeout(slowWarning)
    fetchingUserIdRef.current = null // 取得完了

    if (result.success) {
      logger.dev(`[ensureUserData from ${caller}] userData fetch: ok`)
      setUserData(result.data)
      setErrorReason(null)
    } else {
      // 3回リトライ後の失敗
      logger.error(`[ensureUserData from ${caller}] userData fetch: all retries failed`, { reason: result.reason })

      if (result.reason === 'restore_failed') {
        // 復活失敗は致命的エラー → ログアウト
        await handleAuthFailure(result.reason)
      } else {
        // fetch_failed はエラー表示のみ（ログアウトしない）
        setErrorReason('fetch_failed')
      }
    }
    setDataLoading(false)
  }

  useEffect(() => {
    let sessionResolved = false

    // Hard timeout (15秒): 認証状態が確定できない場合
    const hardTimeout = setTimeout(() => {
      if (!sessionResolved) {
        logger.error('[AuthContext] Hard timeout (15s): auth state unresolved')
        setAuthLoading(false)
        setErrorReason('fetch_failed')
      }
    }, 15000)

    // getSession の遅延警告 (3秒)
    const slowWarningTimeout = setTimeout(() => {
      logger.warn('[AuthContext] getSession is slow (>3s), will rely on onAuthStateChange')
    }, 3000)

    // 初回セッション取得（タイムアウトなし）
    supabase.auth.getSession().then(async ({ data: { session } }) => {
      clearTimeout(slowWarningTimeout)
      sessionResolved = true
      clearTimeout(hardTimeout)

      logger.dev('[AuthContext] getSession:', { hasSession: !!session, userId: session?.user?.id })

      // null上書き防止: 既に値がある場合は上書きしない
      setSession(prev => prev ?? session)
      setUser(prev => prev ?? (session?.user ?? null))
      finishAuthInit()

      // userDataを保証（dedupe機能付き）非同期で並列実行
      ensureUserData(session, 'getSession')
    }).catch((error) => {
      logger.error('[AuthContext] getSession error:', error)
      clearTimeout(hardTimeout)
      handleAuthFailure('fetch_failed')
    })

    // 認証状態の変更を監視
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange(async (event, session) => {
      logger.dev('[AuthContext] onAuthStateChange:', { event, hasSession: !!session })

      // 認証状態が確定したので authLoading を false にする
      sessionResolved = true
      clearTimeout(hardTimeout)
      finishAuthInit()

      // ログイン成功時に authFailureHandledRef をリセット
      if (event === 'SIGNED_IN') {
        authFailureHandledRef.current = false
      }

      setSession(session)
      setUser(session?.user ?? null)

      // userDataを保証（dedupe機能付き）非同期で並列実行
      ensureUserData(session, 'onAuthStateChange')
    })

    return () => {
      clearTimeout(hardTimeout)
      clearTimeout(slowWarningTimeout)
      subscription.unsubscribe()
    }
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

    const result = await retryWithBackoff(async () => {
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
      setDataLoading(true)
      const result = await fetchUserData(user.id)
      if (result.success) {
        setUserData(result.data)
        setErrorReason(null)
      } else {
        if (result.reason === 'restore_failed') {
          await handleAuthFailure(result.reason)
        } else {
          setErrorReason('fetch_failed')
        }
      }
      setDataLoading(false)
    }
  }

  return (
    <AuthContext.Provider
      value={{
        user,
        session,
        userData,
        loading: authLoading || dataLoading,
        authLoading,
        errorReason,
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
