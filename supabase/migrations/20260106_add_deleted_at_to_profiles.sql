-- profiles テーブルに deleted_at カラム追加（論理削除用）

ALTER TABLE profiles ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;

-- インデックス追加（削除済みユーザーの検索用）
CREATE INDEX IF NOT EXISTS idx_profiles_deleted_at ON profiles(deleted_at) WHERE deleted_at IS NOT NULL;

-- コメント
COMMENT ON COLUMN profiles.deleted_at IS '退会日時（論理削除）';
