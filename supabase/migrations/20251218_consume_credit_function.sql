-- クレジット消費関数（セキュリティ強化版）
-- 引数を受け取らず、自動的に「実行した本人」のIDを使います
CREATE OR REPLACE FUNCTION consume_credit(p_user_id UUID DEFAULT NULL) 
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
  target_user_id UUID;
BEGIN
  -- ログイン中のユーザーIDを自動取得
  target_user_id := auth.uid();

  -- もしAPIなどから管理者として実行された場合は引数を使う（保険）
  IF target_user_id IS NULL AND p_user_id IS NOT NULL THEN
    target_user_id := p_user_id;
  END IF;

  -- 残高を減らす処理
  UPDATE credits
  SET
    balance = balance - 1,
    total_used = total_used + 1,
    updated_at = NOW()
  WHERE user_id = target_user_id
    AND balance > 0;  -- 残高がある場合のみ更新

  -- 更新できなかった（残高不足 or ユーザーなし）場合のエラー
  IF NOT FOUND THEN
    RAISE EXCEPTION 'クレジットが不足しているか、ユーザーが見つかりません';
  END IF;
END;
$$;

-- 実行権限を付与
GRANT EXECUTE ON FUNCTION consume_credit(UUID) TO authenticated;