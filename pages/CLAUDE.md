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
5. 出力比率 1:1 / 4:5 / 3:4 の切り替え機能
6. 布の質感アップ（後日）