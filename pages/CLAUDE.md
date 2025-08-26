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