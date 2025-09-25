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

## 🔄 現在進行中のタスク（2025-09-16）

### ✅ 完了済み
1. **Vertex AI移行完成**
   - Google GenAI → Vertex AI SDK完全移行
   - 透かしなし画像生成実現（SynthID非可視透かしのみ）
   - `gemini-2.5-flash-image-preview`モデル使用
   - 認証：サービスアカウント + JSON キーファイル

2. **モーダル表示修正**
   - 背景画像クリック時の全画面モーダル実装
   - カード内表示 → 画面全体表示に修正

3. **プロンプト改善（複数回試行）**
   - ChatGPTアドバイスによる段階的改善
   - パース問題・参考画像混入問題を解決
   - 最新プロンプト：「物理的配置」アプローチで混同防止

### 🔥 現在の問題
- **白画用紙背景のパース調整がうまくいかない**
  - リネン・コンクリートは成功率高い
  - 白画用紙だけパースが平面になりがち
  - nano banana（gemini-2.5-flash-image-preview）の特性かも

### 🎯 次のアクション
1. **白背景用複数参考画像追加予定**
   - `sample_white_flat.jpeg` (平置き)
   - `sample_white_angled.jpeg` (斜め置き)
   - 複数ヒント提供でパース理解向上を狙う

### 💻 技術的状況
- 開発サーバー：2つのインスタンス稼働中（ports 3000/3001）
- ユーザーはPC再起動予定
- 現在のブランチ：`feature/agent/ui-styling`
- 最新コミット：Vertex AI移行完了