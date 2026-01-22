import { createBrowserClient } from '@supabase/ssr'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY

// ▼▼▼ ここからデバッグ用に追加 ▼▼▼
if (typeof window !== 'undefined') {
  // ブラウザでのみ実行
  console.log('🕵️‍♀️ Supabase環境変数チェック:', {
    hasUrl: !!supabaseUrl,
    urlLength: supabaseUrl?.length,
    hasKey: !!supabaseAnonKey,
    // URLの先頭だけ見て、変な文字が入ってないか確認
    urlStart: supabaseUrl ? supabaseUrl.substring(0, 10) : 'MISSING',
  })
}

// もし環境変数がなくても、ここでクラッシュさせずに空文字で強行突破させる！
// （これで AuthContext までは処理が進むはず）
export const supabase = createBrowserClient(
  supabaseUrl || '',
  supabaseAnonKey || ''
)
// ▲▲▲ ここまで ▲▲▲
