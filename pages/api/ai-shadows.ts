import { NextApiRequest, NextApiResponse } from 'next'
import { VertexAI } from '@google-cloud/vertexai'
import formidable from 'formidable'
import fs from 'fs'
import path from 'path'

type StyleKey = "white" | "linen" | "concrete" | "wood" | "white_wood";
type WeatherKey = "sunny" | "cloudy" | "rainy";

type BuildOpts = {
  weather?: WeatherKey;
  addPearlShadow?: boolean;    // 真珠など小パーツの副次影
  aspectRatio?: 'square' | 'original';
  originalSize?: { width: number; height: number };
};

// formidableでファイルアップロード処理
const parseForm = (req: NextApiRequest): Promise<{ fields: formidable.Fields; files: formidable.Files }> => {
  return new Promise((resolve, reject) => {
    const form = formidable({
      maxFileSize: 8 * 1024 * 1024, // 8MB
      keepExtensions: true,
    })
    
    form.parse(req, (err, fields, files) => {
      if (err) reject(err)
      else resolve({ fields, files })
    })
  })
}

// Vertex AI クライアントの初期化
const vertexAI = new VertexAI({
  project: process.env.GOOGLE_CLOUD_PROJECT!,
  location: process.env.GOOGLE_CLOUD_LOCATION || 'global',
  apiEndpoint: 'aiplatform.googleapis.com',  // globalエンドポイントを明示
});

// モデルインスタンスを毎回新規作成（コメントアウト）
// const generativeModel = vertexAI.preview.getGenerativeModel({
//   model: 'gemini-2.5-flash-image-preview',
// });

// 参考画像のパスマップ（複数画像対応）
const REFERENCE_IMAGES: Record<StyleKey, string[]> = {
  white: ['public/input_image/sample_white.png', 'public/input_image/sample_white_angled.png'],
  linen: ['public/input_image/sample_cotton.png', 'public/input_image/sample_cotton_angled.png'],
  concrete: ['public/input_image/sample_concrete.png', 'public/input_image/sample_concrete_angled.png'],
  wood: ['public/input_image/sample_wood.png', 'public/input_image/sample_wood_angled.png'],
  white_wood: ['public/input_image/sample_white_wood.png', 'public/input_image/sample_white_wood_angled.png'],
};

// 背景説明マップ
const BACKGROUND_DESCRIPTIONS: Record<StyleKey, string> = {
  white: "Reference photos showing textured white watercolor paper background from different angles",
  linen: "Reference photos showing natural beige cotton fabric background from different angles",
  concrete: "Reference photos showing light gray concrete surface background from different angles",
  wood: "Reference photos showing natural wood grain surface background from different angles",
  white_wood: "Reference photos showing white painted wood surface background from different angles",
};

// シミュレーション用の背景説明マップ
const SIMULATION_DESCRIPTIONS: Record<StyleKey, string> = {
  white: "a realistic textured white watercolor paper surface",
  linen: "a realistic natural beige cotton fabric surface",
  concrete: "a realistic light gray concrete surface",
  wood: "a realistic natural wood grain surface",
  white_wood: "a realistic white painted wood surface",
};

const WEATHER_MAP: Record<WeatherKey, string> = {
  sunny: "simulate warm daylight coming from the top right — include soft shadows and a bright, airy mood",

  cloudy: "simulate soft overcast daylight — include very gentle shadows and a calm, even mood",

  rainy: "simulate dim, scattered daylight — include minimal shadows and a quiet, muted mood",
};


// 参考画像説明マップ
const REFERENCE_DESCRIPTIONS: Record<StyleKey, string> = {
  white: `The second image should guide overall atmosphere and light feeling.
The third image helps define the paper texture and surface angle.
This is white watercolor / cold-press drawing paper with fine tooth and soft dimples.
This is not copy paper.
Render the paper surface with visible foreshortening from an oblique viewing angle.
Keep the paper flat but receding in space. Do not warp the product.`,

  linen: `The second image should guide overall atmosphere and light feeling.
The third image helps define the fabric texture and direction of folds.
This is soft Japanese cotton with visible neps.
This is not linen — avoid rough or European-style interpretations.`,

  concrete: `The second image should guide overall atmosphere and light feeling.
The third image helps define the concrete surface texture and perspective.
This is smooth, light gray concrete with subtle pores and fine speckling.`,

  wood: `The second image should guide overall atmosphere and light feeling.
The third image helps define the wood surface texture and grain direction.
This is a smooth, light-colored natural wood surface, with subtle linear grain.
The wood surface should follow the angle shown in the reference image.`,

  white_wood: `The second image should guide overall atmosphere and light feeling.
The third image helps define the wood surface texture and grain direction.
This is a smooth, white painted wood surface, with subtle linear grain.
The wood surface should follow the angle shown in the reference image.`,
};

// 天気説明マップ（詳細版）
const WEATHER_DESCRIPTIONS: Record<WeatherKey, string> = {
  sunny: `Simulate sunny daylight with soft shadows and a bright, airy mood.
Override the lighting from the second image — use it only for color tone and general mood.
The atmosphere should feel warm and cheerful, like a bright sunny day.`,

  cloudy: `Simulate overcast daylight with diffuse, soft light. No strong shadows.
Override the lighting from the second image — use it only for color tone and general mood.
The overall mood should feel calm, soft, and quiet — like a cloudy day.`,

  rainy: `Simulate rainy daylight with soft ambient light and no strong shadows.
Override the lighting from the second image — use it only for color tone and general mood.
The atmosphere should feel muted and introspective, like a quiet rainy afternoon.
Colors should be slightly desaturated to match the mood.`,
};

export function buildPrompt(style: StyleKey, opts: BuildOpts = {}) {
  const weatherDesc = WEATHER_DESCRIPTIONS[opts.weather ?? "sunny"];
  const referenceDesc = REFERENCE_DESCRIPTIONS[style] ?? REFERENCE_DESCRIPTIONS.white;
  const tinyPartShadow = opts.addPearlShadow
    ? "Include tiny secondary shadows from small decorative parts."
    : "";

  // Output サイズ指定
  let outputSpec = "square (1024×1024)";
  if (opts.aspectRatio === 'original' && opts.originalSize) {
    const { width, height } = opts.originalSize;
    outputSpec = `${width}×${height} (original aspect ratio)`;
  }

  return `Use ONLY the product from the 1st image.

Rebuild the background inspired by the mood, color tone, and lighting of the 2nd image,
and based on the surface texture and angle shown in the 3rd image.
Do not change the product's shape, proportions, or style in any way.
Adjust only the lighting and color temperature to harmonize with the background environment.
Do not add, remove, or modify any parts or components of the product.
Keep the product within the visible boundaries shown in the first image.

${referenceDesc}

Imagine the background in the 3rd image has been physically placed under the product.
Recreate the surface based on that image, not the original photo.

${weatherDesc}

Make the background look natural and seamless, as if the product was originally photographed on it.
Match the background's depth of field to the product's focus plane to avoid artificial blending.

Output: ${outputSpec}, clean photo, no props or text.
${tinyPartShadow ? `\n${tinyPartShadow}` : ''}`.trim();
}

// 参考画像を読み込む関数（複数画像対応）
async function loadReferenceImages(style: StyleKey, req: NextApiRequest): Promise<{ base64: string; mime: string }[]> {
  try {
    const imagePaths = REFERENCE_IMAGES[style];
    const baseUrl = `https://${req.headers.host}`;

    const imagePromises = imagePaths.map(async (imagePath) => {
      try {
        // ローカル環境ではファイルシステムから読み込み
        if (process.env.NODE_ENV === 'development') {
          const fullPath = path.join(process.cwd(), imagePath);
          const buffer = fs.readFileSync(fullPath);
          const base64 = buffer.toString('base64');
          return { base64, mime: 'image/png' };
        } else {
          // 本番環境（Vercel）ではHTTP経由で取得
          const imageUrl = `${baseUrl}/${imagePath.replace('public/', '')}`;
          const response = await fetch(imageUrl);
          if (!response.ok) {
            throw new Error(`Failed to fetch ${imageUrl}: ${response.status}`);
          }
          const arrayBuffer = await response.arrayBuffer();
          const base64 = Buffer.from(arrayBuffer).toString('base64');
          return { base64, mime: 'image/png' };
        }
      } catch (error) {
        console.error(`Failed to load image ${imagePath}:`, error);
        throw error;
      }
    });

    return await Promise.all(imagePromises);
  } catch (error) {
    console.error(`Failed to load reference images for style ${style}:`, error);
    throw new Error(`Reference images not found for style: ${style}`);
  }
}

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ ok: false, error: 'Method not allowed' })
  }

  try {
    
    // フォームデータ解析
    const { fields, files } = await parseForm(req)
    
    // スタイルとファイル取得
    const style = String(Array.isArray(fields.style) ? fields.style[0] : fields.style || 'white').toLowerCase()
    const weather = String(Array.isArray(fields.weather) ? fields.weather[0] : fields.weather || 'sunny').toLowerCase()
    const aspectRatio = String(Array.isArray(fields.aspectRatio) ? fields.aspectRatio[0] : fields.aspectRatio || 'square') as 'square' | 'original'
    const originalWidth = fields.originalWidth ? parseInt(String(Array.isArray(fields.originalWidth) ? fields.originalWidth[0] : fields.originalWidth)) : undefined
    const originalHeight = fields.originalHeight ? parseInt(String(Array.isArray(fields.originalHeight) ? fields.originalHeight[0] : fields.originalHeight)) : undefined
    const fileArray = Array.isArray(files.file) ? files.file : [files.file]
    const file = fileArray[0]
    
    if (!file) {
      console.error('No file provided')
      return res.status(400).json({ ok: false, error: 'file is required' })
    }
    
    if (!(style in REFERENCE_IMAGES)) {
      console.error(`Invalid style: ${style}`)
      return res.status(400).json({ ok: false, error: `invalid style: ${style}. Valid styles: ${Object.keys(REFERENCE_IMAGES).join(', ')}` })
    }

    if (!(weather in WEATHER_MAP)) {
      console.error(`Invalid weather: ${weather}`)
      return res.status(400).json({ ok: false, error: `invalid weather: ${weather}. Valid weather: ${Object.keys(WEATHER_MAP).join(', ')}` })
    }


    // 入力画像を読み込み
    const inputBuffer = fs.readFileSync(file.filepath)
    const inputBase64 = inputBuffer.toString('base64')
    const inputMime = file.mimetype || 'image/jpeg'
    
    // 参考画像を読み込み（複数）
    const referenceImages = await loadReferenceImages(style as StyleKey, req)
    
    const originalSize = originalWidth && originalHeight ? { width: originalWidth, height: originalHeight } : undefined
    const prompt = buildPrompt(style as StyleKey, {
      weather: weather as WeatherKey,
      aspectRatio,
      originalSize
    })

    // ログファイルに出力
    const logData = `=== ${new Date().toISOString()} ===\nStyle: ${style}\nWeather: ${weather}\nPrompt:\n${prompt}\n\n`
    fs.appendFileSync('debug.log', logData)

    console.log('=== GENERATED PROMPT ===')
    console.log(prompt)
    console.log('=== END PROMPT ===')


    // Vertex AI Gemini API呼び出し（2つの画像を送信）
    let result
    try {
      console.log('Calling Vertex AI with model:', 'gemini-2.5-flash-image-preview')
      console.log('Project:', process.env.GOOGLE_CLOUD_PROJECT)
      console.log('Location:', process.env.GOOGLE_CLOUD_LOCATION)

      // 毎回新しいモデルインスタンスを作成してセッション独立性を確保
      const freshGenerativeModel = vertexAI.preview.getGenerativeModel({
        model: 'gemini-2.5-flash-image-preview',
      });

      // Base64データの確認
      console.log('Input image MIME:', inputMime)
      console.log('Input base64 length:', inputBase64.length)
      console.log('Input base64 starts with:', inputBase64.substring(0, 20))
      console.log('Reference images count:', referenceImages.length)
      referenceImages.forEach((img, index) => {
        console.log(`Reference image ${index + 1} MIME:`, img.mime)
        console.log(`Reference image ${index + 1} base64 length:`, img.base64.length)
        console.log(`Reference image ${index + 1} base64 starts with:`, img.base64.substring(0, 20))
      })

      const parts = [
        { text: prompt },
        { inlineData: { mimeType: inputMime, data: inputBase64 } }, // 1ST IMAGE: 商品画像
        ...referenceImages.map(img => ({ inlineData: { mimeType: img.mime, data: img.base64 } })) // 2ND, 3RD IMAGE: 参考画像
      ];

      result = await freshGenerativeModel.generateContent({
        contents: [{
          role: 'user',
          parts: parts
        }],
        generationConfig: {
          maxOutputTokens: 1024,
          temperature: 0.1,  // 低めにして一貫性向上
          topP: 0.4         // 安定性重視
        }
      })
    } catch (e: any) {
      console.error('Vertex AI Error Details:')
      console.error('Status:', e?.cause?.status)
      console.error('Message:', e?.message)
      if (e?.cause?.response?.text) {
        console.error('Response body:', await e.cause.response.text())
      }
      throw e
    }


    // 画像レスポンス抽出（Vertex AI形式）
    const parts = result.response?.candidates?.[0]?.content?.parts ?? []
    const imagePart = parts.find((p: any) => p.inlineData && p.inlineData.mimeType?.startsWith('image/'))
    
    if (!imagePart) {
      console.error('No image in response:', result)
      return res.status(500).json({ ok: false, error: 'no image in response', raw: result })
    }

    const outMime = imagePart.inlineData?.mimeType || 'image/png'
    const outB64 = imagePart.inlineData?.data

    if (!outB64) {
      console.error('No image data in response')
      return res.status(500).json({ ok: false, error: 'no image data in response' })
    }


    return res.json({ 
      ok: true,
      imageBase64: `data:${outMime};base64,${outB64}`,
      mimeType: outMime,
      debug: {
        inputStyle: style,
        inputWeather: weather,
        outputMimeType: outMime,
        outputLength: outB64.length,
        referenceImagesUsed: REFERENCE_IMAGES[style as StyleKey]
      }
    })

  } catch (error: any) {
    console.error('AI shadows generation error:', error)
    
    // 参考画像の読み込みエラーを特別に処理
    if (error.message?.includes('Reference image not found')) {
      return res.status(500).json({ 
        ok: false, 
        error: 'Reference image loading failed',
        details: error.message
      })
    }
    
    return res.status(500).json({ 
      ok: false, 
      error: error?.message || 'AI generation failed',
      stack: process.env.NODE_ENV === 'development' ? error?.stack : undefined
    })
  }
}

// Next.jsでファイルアップロードを有効にする
export const config = {
  api: {
    bodyParser: false, // formidableを使うため無効化
  },
}