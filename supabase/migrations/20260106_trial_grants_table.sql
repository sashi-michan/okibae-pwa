-- 無料トライアル付与履歴テーブル
-- 退会後も残し続けることで、再登録時の無料特典重複を防ぐ

CREATE TABLE IF NOT EXISTS trial_grants (
  eligibility_key TEXT PRIMARY KEY,  -- HMAC-SHA256(email)で生成
  first_granted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  granted_credits INTEGER NOT NULL DEFAULT 5,
  source TEXT NOT NULL DEFAULT 'signup',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- インデックス
CREATE INDEX IF NOT EXISTS idx_trial_grants_created_at ON trial_grants(created_at);

-- RLS有効化（Service Roleのみアクセス可能）
ALTER TABLE trial_grants ENABLE ROW LEVEL SECURITY;

-- 通常ユーザーからの権限を明示的に剥奪
REVOKE ALL ON trial_grants FROM anon, authenticated;

-- Service Roleのみアクセス許可
GRANT SELECT, INSERT ON TABLE trial_grants TO service_role;

-- Service Role用のポリシーを明示的に作成
CREATE POLICY "Service role can manage trial grants"
ON trial_grants
FOR ALL
TO service_role
USING (true)
WITH CHECK (true);

-- 通常ユーザー（authenticated, anon）は一切アクセス不可（ポリシーなし・権限剥奪済み）

-- コメント
COMMENT ON TABLE trial_grants IS '無料トライアル付与履歴。退会後も削除せず、再登録時の重複付与を防ぐ';
COMMENT ON COLUMN trial_grants.eligibility_key IS 'HMAC-SHA256で生成されたキー（個人情報非特定）';
COMMENT ON COLUMN trial_grants.first_granted_at IS '初回付与日時';
COMMENT ON COLUMN trial_grants.granted_credits IS '付与されたクレジット数';
COMMENT ON COLUMN trial_grants.source IS '付与元（signup等）';
