# Supabaseマイグレーション手順書

## 退会機能・無料クレジット重複防止機能の適用方法

### 事前準備

1. **HMAC Secretの設定（必須！）**

   ⚠️ **重要：** この設定を忘れると新規登録が全て失敗します！

   Supabase Dashboardで以下のいずれかの方法でHMACシークレットを設定してください：

   #### 方法A: Database Settings（推奨）

   1. [Supabase Dashboard](https://app.supabase.com) にログイン
   2. プロジェクトを選択
   3. `Settings` > `Database` > `Custom Postgres Configuration`
   4. 以下を追加：
      ```
      app.settings.trial_hmac_secret = 'f7b960da86e932dbd404044b17d9648a08ebbfc4fbe5c2c842e9957c12d926f2'
      ```
   5. `Save` をクリック
   6. データベースが再起動されるまで待機（数秒〜1分）

   **設定確認方法：**
   ```sql
   SELECT current_setting('app.settings.trial_hmac_secret', true);
   -- 設定したシークレットが返ってくればOK
   ```

   #### 方法B: Vault（より安全）

   1. `Settings` > `Vault` を開く
   2. `New Secret` をクリック
   3. Name: `trial_hmac_secret`
   4. Value: `f7b960da86e932dbd404044b17d9648a08ebbfc4fbe5c2c842e9957c12d926f2`
   5. `Save` をクリック

   ※Vault使用時はSQL内の `current_setting()` を Vault API呼び出しに変更する必要があります

---

### マイグレーション適用手順

以下のSQLファイルを**順番通りに**実行してください：

#### 1. trial_grantsテーブル作成

1. `SQL Editor` を開く
2. `supabase/migrations/20260106_trial_grants_table.sql` の内容をコピー＆ペースト
3. `Run` をクリック
4. ✅ 成功メッセージを確認

**確認事項：**
- `trial_grants` テーブルが作成されている
- RLSが有効化されている
- Service Roleポリシーが設定されている

---

#### 2. profilesテーブルにdeleted_at追加

1. `SQL Editor` を開く
2. `supabase/migrations/20260106_add_deleted_at_to_profiles.sql` の内容をコピー＆ペースト
3. `Run` をクリック
4. ✅ 成功メッセージを確認

**確認事項：**
- `profiles` テーブルに `deleted_at` カラムが追加されている
- インデックス `idx_profiles_deleted_at` が作成されている

---

#### 3. subscriptionsテーブルにdeleted_at追加

1. `SQL Editor` を開く
2. `supabase/migrations/20260106_add_deleted_at_to_subscriptions.sql` の内容をコピー＆ペースト
3. `Run` をクリック
4. ✅ 成功メッセージを確認

**確認事項：**
- `subscriptions` テーブルに `deleted_at` カラムが追加されている
- インデックス `idx_subscriptions_deleted_at` が作成されている

---

#### 4. 新規登録ロジック更新（最重要）

1. `SQL Editor` を開く
2. `supabase/migrations/20260106_update_signup_trigger.sql` の内容をコピー＆ペースト
3. `Run` をクリック
4. ✅ 成功メッセージを確認

**確認事項：**
- `pgcrypto` 拡張が有効化されている
- `generate_eligibility_key()` 関数が作成されている
- `handle_new_user()` 関数が更新されている

**⚠️ エラーが出た場合：**

エラー: `unrecognized configuration parameter "app.settings.trial_hmac_secret"`
→ 事前準備の「HMAC Secretの設定」を完了してください

---

### テスト手順

#### 1. 新規ユーザー登録テスト（初回）

1. アプリで新規ユーザー登録
2. SQL Editorで確認：
   ```sql
   SELECT * FROM credits WHERE user_id = '[ユーザーID]';
   -- balance = 5 であることを確認

   SELECT * FROM trial_grants ORDER BY created_at DESC LIMIT 1;
   -- eligibility_key が記録されていることを確認
   ```

#### 2. 退会テスト

1. ログイン状態で「退会」ボタンをクリック
2. 確認モーダルで「退会する」をクリック
3. SQL Editorで確認：
   ```sql
   SELECT deleted_at FROM profiles WHERE email = '[退会したメール]';
   -- deleted_at にタイムスタンプが入っていることを確認

   SELECT balance FROM credits WHERE user_id = '[退会したユーザーID]';
   -- balance = 0 であることを確認
   ```

#### 3. 再登録テスト（無料クレジット防止）

1. 同じメールアドレスで再度ユーザー登録
2. SQL Editorで確認：
   ```sql
   SELECT * FROM credits WHERE user_id = '[新しいユーザーID]';
   -- balance = 0 であることを確認（5ではない！）

   SELECT COUNT(*) FROM trial_grants WHERE eligibility_key = (
     SELECT eligibility_key FROM trial_grants
     WHERE eligibility_key = generate_eligibility_key('[メールアドレス]')
   );
   -- COUNT = 1 であることを確認（重複していない）
   ```

---

## トラブルシューティング

### エラー: `TRIAL_HMAC_SECRET is not configured`

**原因：** Database SettingsまたはVaultでHMACシークレットが未設定

**解決策：**
1. 事前準備の「HMAC Secretの設定」を実施
2. データベースを再起動（必要に応じて）
3. 再度マイグレーションを実行

---

### エラー: `duplicate key value violates unique constraint`

**原因：** 同時登録によるPK衝突（修正済みのSQLでは発生しないはず）

**解決策：**
- `20260106_update_signup_trigger.sql` の最新版を使用していることを確認
- `ON CONFLICT DO NOTHING` が含まれていることを確認

---

### 既存ユーザーのtrial_grants記録

既存ユーザーにも無料クレジット重複防止を適用したい場合：

```sql
-- 既存の全ユーザーをtrial_grantsに記録
INSERT INTO trial_grants (eligibility_key, first_granted_at, granted_credits, source)
SELECT
  generate_eligibility_key(email),
  created_at,
  5,
  'migration'
FROM profiles
WHERE deleted_at IS NULL
ON CONFLICT (eligibility_key) DO NOTHING;
```

---

## ロールバック手順（緊急時）

```sql
-- 1. トリガー関数を元に戻す（元のhandle_new_user関数をバックアップから復元）

-- 2. 追加したカラムを削除
ALTER TABLE profiles DROP COLUMN IF EXISTS deleted_at;
ALTER TABLE subscriptions DROP COLUMN IF EXISTS deleted_at;

-- 3. trial_grantsテーブルを削除
DROP TABLE IF EXISTS trial_grants CASCADE;

-- 4. 関数を削除
DROP FUNCTION IF EXISTS generate_eligibility_key(TEXT);
```

⚠️ **注意：** ロールバックすると退会機能と無料クレジット防止機能が停止します。

---

## セキュリティチェックリスト

- ✅ `trial_grants` のRLSが有効化されている
- ✅ Service Roleポリシーのみが設定されている
- ✅ 通常ユーザー（authenticated/anon）の権限が明示的に剥奪されている（REVOKE）
- ✅ HMAC-SHA256でメールアドレスがハッシュ化されている
- ✅ `SECURITY DEFINER` で `search_path = public, pg_temp` が設定されている
- ✅ 同時登録対策（ON CONFLICT）が実装されている
- ✅ `GET DIAGNOSTICS` が正しい構文で実装されている
- ✅ `current_setting()` で設定未定義時の例外処理が実装されている

---

## 完了！

すべてのマイグレーションが成功したら、以下の機能が利用可能になります：

- ✅ 退会機能（論理削除）
- ✅ 退会後の再登録時の無料クレジット重複防止
- ✅ Stripe Customer の論理削除マーク
- ✅ 同時登録時の競合回避

問題が発生した場合は、このガイドのトラブルシューティングセクションを参照してください。
