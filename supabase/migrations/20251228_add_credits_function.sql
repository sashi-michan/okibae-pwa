-- クレジット追加関数（Stripe Webhook用）
-- consume_credit と違い、残高を増やす（購入時の処理）

CREATE OR REPLACE FUNCTION add_credits(
  p_user_id UUID,
  p_amount INTEGER
)
RETURNS VOID
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
  -- クレジット残高を加算
  UPDATE credits
  SET
    balance = balance + p_amount,
    updated_at = NOW()
  WHERE user_id = p_user_id;

  -- レコードが存在しない場合は作成
  IF NOT FOUND THEN
    INSERT INTO credits (user_id, balance, total_used, last_reset_at, updated_at)
    VALUES (p_user_id, p_amount, 0, NOW(), NOW());
  END IF;
END;
$$;

-- Service Roleからの実行を許可
GRANT EXECUTE ON FUNCTION add_credits(UUID, INTEGER) TO service_role;
