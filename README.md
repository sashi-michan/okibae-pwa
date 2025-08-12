# OKIBAE PWA Starter — Cutout版

- Replicateの背景消しAPI（851-labs/background-remover）に接続
- 背景ボタン＝即プレビュー → 背景消し完了で自動リファイン（色なじみ＋自然な影）
- PWA対応・可愛いUIそのまま

## 使い方
1) 依存入れる → `npm install`
2) 開発 → `npm run dev`（http://localhost:3000）

## 環境変数（重要）
- Vercel の Project Settings → Environment Variables に
  - `REPLICATE_API_TOKEN = <あなたのトークン>` を追加してデプロイ
- ローカル開発で使う場合は `.env.local` を作成：
```
REPLICATE_API_TOKEN=xxxxx
```

## モデル
- デフォルト：`851-labs/background-remover` の version `a029dff3...`
- 他サービス（remove.bg / Photoroom 等）にも差し替えやすい構造です。

Happy hacking 🫶
