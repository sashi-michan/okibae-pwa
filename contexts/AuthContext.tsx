import { createContext, useContext, useEffect, useState } from 'react'
import { User, Session } from '@supabase/supabase-js'
import { supabase } from '../lib/supabase/client'
import type { UserData } from '../types/database'

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

  // ユーザーデータ取得関数
  const fetchUserData = async (userId: string): Promise<UserData | null> => {
    try {
      const [profileRes, subscriptionRes, creditsRes] = await Promise.all([
        supabase.from('profiles').select('*').eq('id', userId).single(),
        supabase.from('subscriptions').select('*').eq('user_id', userId).single(),
        supabase.from('credits').select('*').eq('user_id', userId).single(),
      ])

      if (profileRes.error || subscriptionRes.error || creditsRes.error) {
        console.error('Failed to fetch user data:', {
          profile: profileRes.error,
          subscription: subscriptionRes.error,
          credits: creditsRes.error,
        })
        return null
      }

      return {
        profile: profileRes.data,
        subscription: subscriptionRes.data,
        credits: creditsRes.data,
      }
    } catch (error) {
      console.error('Error fetching user data:', error)
      return null
    }
  }

  useEffect(() => {
    // 初回セッション取得
    supabase.auth.getSession().then(async ({ data: { session } }) => {
      setSession(session)
      setUser(session?.user ?? null)

      if (session?.user) {
        const data = await fetchUserData(session.user.id)
        setUserData(data)
      }

      setLoading(false)
    })

    // 認証状態の変更を監視
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange(async (_event, session) => {
      setSession(session)
      setUser(session?.user ?? null)

      if (session?.user) {
        const data = await fetchUserData(session.user.id)
        setUserData(data)
      } else {
        setUserData(null)
      }

      setLoading(false)
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
    const { error } = await supabase.auth.signOut()
    if (error) throw error
    setUserData(null)
  }

  const refreshUserData = async () => {
    if (user) {
      const data = await fetchUserData(user.id)
      setUserData(data)
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
