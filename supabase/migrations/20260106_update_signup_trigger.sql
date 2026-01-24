-- 新規登録時の無料クレジット付与ロジック修正
-- trial_grantsテーブルをチェックして、初回のみ5クレジット付与

-- pgcrypto拡張を有効化（HMAC-SHA256用）
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- HMAC-SHA256でeligibility_keyを生成する関数
CREATE OR REPLACE FUNCTION generate_eligibility_key(email TEXT)
RETURNS TEXT
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
DECLARE
  secret TEXT;
  normalized_email TEXT;
BEGIN
  -- 環境変数からHMACシークレットを取得
  -- Supabase Dashboard > Settings > Vault で設定
  -- または Database Settings > Custom Postgres Configuration で app.trial_hmac_secret を設定
  secret := current_setting('app.settings.trial_hmac_secret', true);

  IF secret IS NULL OR secret = '' THEN
    RAISE EXCEPTION 'TRIAL_HMAC_SECRET is not configured in database settings';
  END IF;

  -- メールアドレスを正規化（小文字化、トリム）
  normalized_email := LOWER(TRIM(email));

  -- HMAC-SHA256でハッシュ化
  RETURN encode(hmac(normalized_email::bytea, secret::bytea, 'sha256'), 'hex');
END;
$$;

-- 既存のhandle_new_user関数を更新
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
DECLARE
  eligibility_key TEXT;
  initial_credits INTEGER := 0;
  rowcount INTEGER;
BEGIN
  -- eligibility_keyを生成
  eligibility_key := generate_eligibility_key(NEW.email);

  -- trial_grantsに挿入を試みる（ON CONFLICT DO NOTHINGで同時登録対策）
  -- 挿入成功 = 初回登録
  INSERT INTO trial_grants (eligibility_key, first_granted_at, granted_credits, source)
  VALUES (eligibility_key, NOW(), 5, 'signup')
  ON CONFLICT (eligibility_key) DO NOTHING;

  -- 挿入が成功したかチェック
  GET DIAGNOSTICS rowcount = ROW_COUNT;

  -- 初回登録の場合のみ5クレジット付与
  IF rowcount > 0 THEN
    initial_credits := 5;
  END IF;

  -- profilesテーブルに挿入
  INSERT INTO public.profiles (id, email, created_at, updated_at)
  VALUES (NEW.id, NEW.email, NOW(), NOW());

  -- subscriptionsテーブルに挿入（フリープラン）
  INSERT INTO public.subscriptions (
    user_id,
    status,
    created_at,
    updated_at
  )
  VALUES (
    NEW.id,
    'free',
    NOW(),
    NOW()
  );

  -- creditsテーブルに挿入
  INSERT INTO public.credits (
    user_id,
    balance,
    total_used,
    last_reset_at,
    updated_at
  )
  VALUES (
    NEW.id,
    initial_credits,
    0,
    NOW(),
    NOW()
  );

  RETURN NEW;
END;
$$;

-- コメント
COMMENT ON FUNCTION generate_eligibility_key IS 'メールアドレスからHMAC-SHA256でeligibility_keyを生成';
COMMENT ON FUNCTION handle_new_user IS '新規ユーザー登録時の処理。trial_grantsをチェックして初回のみ5クレジット付与（同時登録対応）';
