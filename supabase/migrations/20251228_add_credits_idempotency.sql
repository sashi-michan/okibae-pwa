-- Webhook処理済みイベントを記録するテーブル
CREATE TABLE IF NOT EXISTS webhook_events (
  id TEXT PRIMARY KEY, -- Stripe Event ID
  processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- インデックス（古いイベントのクリーンアップ用）
CREATE INDEX IF NOT EXISTS idx_webhook_events_processed_at ON webhook_events(processed_at);

-- クレジット追加関数（冪等性対応版・改善版）
CREATE OR REPLACE FUNCTION add_credits(
  p_user_id UUID,
  p_amount INTEGER,
  p_stripe_event_id TEXT
)
RETURNS BOOLEAN -- 処理したかどうかを返す
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public -- SECURITY DEFINER使用時のベストプラクティス
AS $$
BEGIN
  -- パラメータバリデーション
  IF p_stripe_event_id IS NULL OR length(p_stripe_event_id) = 0 THEN
    RAISE EXCEPTION 'stripe_event_id is required';
  END IF;

  IF p_amount IS NULL OR p_amount <= 0 THEN
    RAISE EXCEPTION 'amount must be > 0, got %', p_amount;
  END IF;

  IF p_user_id IS NULL THEN
    RAISE EXCEPTION 'user_id is required';
  END IF;

  -- イベントを記録（UNIQUE制約で二重処理を防止）
  -- ON CONFLICT DO NOTHINGで既に処理済みの場合はスキップ
  INSERT INTO webhook_events (id)
  VALUES (p_stripe_event_id)
  ON CONFLICT (id) DO NOTHING;

  -- INSERT成功かチェック
  IF NOT FOUND THEN
    -- 既に処理済み
    RAISE NOTICE 'Event % already processed', p_stripe_event_id;
    RETURN FALSE;
  END IF;

  -- クレジット残高を加算（UPSERT）
  INSERT INTO credits (user_id, balance, total_used, last_reset_at, updated_at)
  VALUES (p_user_id, p_amount, 0, NOW(), NOW())
  ON CONFLICT (user_id) DO UPDATE
  SET
    balance = credits.balance + p_amount,
    updated_at = NOW();

  RETURN TRUE; -- 処理完了
END;
$$;

-- Service Roleからの実行を許可
GRANT EXECUTE ON FUNCTION add_credits(UUID, INTEGER, TEXT) TO service_role;
GRANT SELECT, INSERT ON TABLE webhook_events TO service_role;

-- 古いイベントをクリーンアップする関数（オプション）
-- 30日以上前のイベントを削除
CREATE OR REPLACE FUNCTION cleanup_old_webhook_events()
RETURNS INTEGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  v_deleted_count INTEGER;
BEGIN
  DELETE FROM webhook_events
  WHERE processed_at < NOW() - INTERVAL '30 days';

  GET DIAGNOSTICS v_deleted_count = ROW_COUNT;
  RETURN v_deleted_count;
END;
$$;

GRANT EXECUTE ON FUNCTION cleanup_old_webhook_events() TO service_role;
