# CLAUDE.md

## Project Purpose
OKIBAE is a web-based PWA tool designed for handmade creators (especially macramé artists) to easily take, process, and prepare product photos for online sales.

## Overview
- Allows users to upload a photo of their handmade item.
- Provides background replacement options:
  1. White background
  2. Beige background
  3. AI-generated fabric background
- Performs color blending and adds realistic shadows for natural presentation.
- Aims to streamline product photo preparation without requiring advanced photo editing skills.
- Target users are crafters who want quick, consistent, and attractive listing images.

## Rules
- Always show a diff and ask before applying any changes.
- Work only on branches that start with `feature/agent/`.
- Do not read `.env`, `secrets/**`, or `public/originals/**`.

## Tech Stack
- Next.js 14
- Tailwind CSS
- next-pwa for PWA support

## Current Goals (MVP - Priority Order)
1. 文言変更：「KIBAE」→「OKIBAE」
2. UI改善（操作順の誘導）
    - **Step1**：画像を選ぶ → プレビューエリアにそのまま表示
    - **Step2**：背景を選ぶ or おまかせ → プレビューエリアに仮合成した画像を表示
    - **Step3**：生成！ → 出力キャンバスに最終的な画像を表示
    - **Step4**：画像保存
    - 各ステップが順番通りになるようUI配置を改善（ボタン横並びの混乱防止）
3. Step2で仮合成画像が表示されるまで他の背景が選べないようにする
4. 連打ガード＆ローディングUI追加
5. 画像処理のクオリティアップ
    - 光源特定
    - パース推定
    - 自然な影の生成
    段階的ハイブリッドアプローチ: 
    1. Phase 1: OpenCVで基本的な光源・パース推定 
    2. Phase 2: 結果をAIモデルで補正・改善 
    3. Phase 3: ユーザーがアルゴリズム選択可能 
    メリット: 
      - コスト効率的 
      - 高速なプロトタイピング 
      - 後から精度向上可能
6. 出力比率 1:1 / 4:5 / 3:4 の切り替え機能
7. 布の質感アップ（後日）

## 影生成システム設計原則

### SSOT (Single Source of Truth) アーキテクチャ
- パラメータは元サイズで1回だけ計算、プレビュー用にスケール
- `compute_shadow_params()`: 正規化されたパラメータを計算
- `scale_for_preview()`: float精度を保持してスケール

### 一貫性保証の仕組み（崩れにくいシステム）
1. **α SSIM チェック**: 開発モードで preview≒final を常時検証 (SSIM≥0.985)
2. **奇数ガウシアンカーネル**: `blur_px = int(...) | 1` で強制的に奇数
3. **dx/dy float精度**: position vectors は整数丸めしない
4. **共有クロップシステム**: フルサイズで外接矩形計算→プレビューでスケール使用

### コード品質ルール
- 影の向き修正: UI角度(0°=上) → 数学角度(0°=右) 変換を正しく実装
- マージン二重移動回避: warpAffine でマージン調整しない（展開済みキャンバス使用）
- CONSISTENCY_CHECK=True で開発時検証を必須とする

## 最新進捗（2024-01-XX）

### ✅ 完了したこと
1. **minne作家向けデザインリニューアル完成**
   - 全体をくすみカラーで統一（brand色: #C2A2A8）
   - 生成ボタンをくすみピンク（#C792A3）に変更
   - 温かみのあるグラデーション背景（pink-50 → cream → orange-50）
   - カードデザインの洗練（影・ボーダー・アニメーション調整）

2. **カルーセル機能完成**
   - Slick風3スライド表示カルーセル実装
   - スワイプジェスチャー完全対応
   - 矢印ナビゲーション（42%位置、ミニマルデザイン）
   - 1:1アスペクト比画像表示
   - CSS詳細度バグ修正完了

3. **天気選択UI大幅改善**
   - プロ品質SVGアイコン実装（太陽・雲・傘）
   - くすみカラー適用（晴れ:#EDBC9D, くもり:#D6C5D5, 雨:#ACC3D6）
   - ミニマルで視認性の高いデザイン

4. **全体UI統一**
   - ボタンアニメーション調整（膨らみ効果削除）
   - NavBarのホバー効果統一
   - タイトル装飾（✨）削除
   - 影の切れ問題解決（gridコンテナpadding調整）

## 次の開発予定

### 🎯 直近の目標（優先度: 高）
1. **カスタムアイコン実装**
   - ユーザー作成アイコンをナビバーに表示
   - 既存の「Ki」アイコンと差し替え

2. **背景選択肢拡張**
   - 木目ナチュラル背景追加
   - 木目ホワイト背景追加
   - カルーセルの5スライド対応

### 🚀 中期目標（優先度: 中）
3. **ユーザー認証機能**
   - ログイン機能実装（ログインなしでも基本機能は利用可能）
   - 認証プロバイダー選定（Google/Apple/Email等）

4. **ログイン限定機能**
   - 画像保存履歴
   - お気に入り背景設定
   - 高解像度エクスポート
   - カスタム背景アップロード
   - その他プレミアム機能（1つ選定予定）

### 📝 技術的メモ
- くすみカラーパレットが作家さんに好評
- SVGアイコンの統一感が高い
- カルーセルのスワイプ対応が完璧に動作
- CSS詳細度問題は`!important`＋親セレクタで解決

## 🎉 最新完了タスク（2025-09-22）

### ✅ プロンプト大幅改善完了
1. **複数参考画像システム実装**
   - 2枚参考画像体制（テクスチャ + アングル）で品質大幅向上
   - 白画用紙のパース問題を完全解決
   - 参考画像の純度向上（商品なし背景のみ）

2. **商品の色調和機能実装**
   - 「背景環境に合わせた照明と色温度調整」プロンプト追加
   - 商品が背景から浮く問題を解決
   - 形状・比率は保持しつつ自然な馴染み感を実現

3. **5背景対応完了**
   - 白画用紙・木綿・コンクリート・木目ナチュラル・木目ホワイト
   - JPEG→PNG統一で画質向上
   - カルーセルUI対応とモーダル表示完成

4. **プロンプト構造刷新**
   - 「物理的配置」アプローチで参考画像混同を完全防止
   - 天気別詳細ライティング指示の統一化
   - 壁表示問題を参考画像最適化で解決

### ✅ UI/UX改善完了
- 「再生成！」ボタンで連続テスト対応
- カルーセル画質最適化（crisp-edges + backface-visibility）
- 装飾エフェクト削除でクリーンデザイン
- 「木綿布」→「木綿」表記修正

### ✅ 開発環境整備完了
- Git履歴クリーンアップ（不要JPEG削除）
- .gitignore最適化（開発ファイル除外）
- アスペクト比選択機能実装済み（Vertex AI制限で一時非表示）

### 🏆 達成した品質向上
- **白画用紙パース問題**: 解決済み
- **商品色調和問題**: 解決済み
- **参考画像混入問題**: 解決済み
- **壁表示問題**: 解決済み
- **画像生成成功率**: 大幅向上

### 💻 現在の技術状況
- ブランチ: `feature/agent/ui-styling`
- 開発サーバー: port 3002で安定動作
- AI画像生成: 高品質・高成功率で稼働中

## 🚀 ログイン機能実装（2025-01-20）

### ✅ Phase 1: 認証基盤完了
1. **Supabase認証実装**
   - Supabaseプロジェクト作成・設定
   - Google OAuth連携
   - `@supabase/supabase-js`, `@supabase/ssr` インストール
   - Supabaseクライアント設定（lib/supabase/client.ts, server.ts）

2. **認証機能実装**
   - AuthContext作成（contexts/AuthContext.tsx）
   - useAuthフック実装
   - ログインページUI作成（pages/login.tsx）
   - 未ログイン時の自動リダイレクト
   - NavBarにログアウトボタン追加
   - Reactフックルール違反修正（index.tsx）

3. **データベース統合**
   - テーブル作成（profiles, subscriptions, credits）
   - RLS（Row Level Security）ポリシー設定
   - 新規ユーザー自動登録トリガー実装
   - 先着100名判定機能（has_launch_couponフラグ）
   - TypeScript型定義作成（types/database.ts）
   - AuthContextにユーザーデータ自動取得機能追加
   - NavBarにクレジット残高表示（のこり 20/20）
   - Hydrationエラー修正（クライアントサイドレンダリング対応）

### 📋 データベース設計
```
profiles
  - id (uuid, FK to auth.users)
  - email
  - stripe_customer_id
  - created_at, updated_at

subscriptions
  - user_id (uuid, FK to auth.users)
  - status ('free' | 'pro')
  - stripe_subscription_id
  - current_period_start, current_period_end
  - cancel_at_period_end
  - has_launch_coupon (先着100名特典フラグ)
  - created_at, updated_at

credits
  - user_id (uuid, FK to auth.users)
  - balance (残高)
  - total_used (累計使用数)
  - last_reset_at (リセット日時)
  - updated_at
```

### 🎯 次回実装予定（Phase 2: 決済・サブスク）
1. **Stripeアカウント設定**
2. **Stripe連携パッケージインストール**
3. **サブスク購入フロー実装**（¥600/月、先着100名は3ヶ月¥400/月）
4. **Webhook実装**（サブスク状態同期）
5. **追加クレジット購入**（+30枚 ¥200）

### 💻 技術スタック更新
- **認証**: Supabase Authentication (Google OAuth)
- **データベース**: Supabase PostgreSQL
- **ブランチ**: `feature/agent/login-system`
- **コミット**:
  - `b9775f2` feat(auth): Implement Supabase authentication with Google login
  - `6018fe6` feat(db): Add user data management with Supabase