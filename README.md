# OKIBAE

手作り作家さん向けの商品撮影背景置き換えアプリです。
商品写真を美しい背景に置き換えて、オンライン販売に最適な画像を生成します。

## 特徴

- **AI画像生成**: Google Vertex AI (Gemini 2.5 Flash) による高品質な背景置き換え
- **5種類の背景**: 白画用紙・木綿・コンクリート・木目ナチュラル・木目ホワイト
- **自動色調和**: 商品と背景の色味を自然に調整
- **PWA対応**: スマホからも快適に利用可能
- **使いやすいUI**: ステップバイステップの直感的なデザイン

## 技術スタック

- **フロントエンド**: Next.js 14, React, TypeScript, Tailwind CSS
- **AI画像生成**: Google Cloud Vertex AI
- **デプロイ**: Vercel
- **PWA**: next-pwa

## 環境変数

```env
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=global
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
```

## 開発

```bash
npm install
npm run dev
```

## 使い方

1. **画像を選ぶ**: 商品写真をアップロード
2. **背景を選ぶ**: 5種類の背景から選択
3. **天気を選ぶ**: 晴れ・曇り・雨から選択
4. **生成！**: AI が自然な背景置き換えを実行

## フィードバック

ご意見・ご要望は[こちらのフォーム](https://docs.google.com/forms/d/e/1FAIpQLSf_pkyMpQ0SQXJ--MhNItVSi9LRHW4OBNsUroergJYa396e6w/viewform)からお聞かせください。

---

Made with ❤️ for handmade creators