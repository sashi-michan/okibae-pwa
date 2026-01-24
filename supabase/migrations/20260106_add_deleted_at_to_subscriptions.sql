-- subscriptions テーブルに deleted_at カラム追加（論理削除用）

ALTER TABLE subscriptions ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;

-- インデックス追加
CREATE INDEX IF NOT EXISTS idx_subscriptions_deleted_at ON subscriptions(deleted_at) WHERE deleted_at IS NOT NULL;

-- コメント
COMMENT ON COLUMN subscriptions.deleted_at IS '退会日時（論理削除）';
