import { NextApiRequest, NextApiResponse } from 'next'
import { VertexAI } from '@google-cloud/vertexai'
import formidable from 'formidable'
import fs from 'fs'
import path from 'path'

type StyleKey = "white" | "linen" | "concrete";
type WeatherKey = "sunny" | "cloudy" | "rainy";

type BuildOpts = {
  weather?: WeatherKey;
  addPearlShadow?: boolean;    // 真珠など小パーツの副次影
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

const generativeModel = vertexAI.preview.getGenerativeModel({
  model: 'gemini-2.5-flash-image-preview',
});

// 参考画像のパスマップ
const REFERENCE_IMAGES: Record<StyleKey, string> = {
  white: 'input_image/sample_white.jpeg',
  linen: 'input_image/sample_linen.jpeg', 
  concrete: 'input_image/sample_concrete.jpeg',
};

// 背景説明マップ
const BACKGROUND_DESCRIPTIONS: Record<StyleKey, string> = {
  white: "A reference photo showing textured white watercolor paper background",
  linen: "A reference photo showing natural beige linen fabric background",
  concrete: "A reference photo showing light gray concrete surface background",
};

const WEATHER_MAP: Record<WeatherKey, string> = {
  sunny: "simulate warm daylight coming from the top right — include soft shadows and a bright, airy mood",

  cloudy: "simulate soft overcast daylight — include very gentle shadows and a calm, even mood",

  rainy: "simulate dim, scattered daylight — include minimal shadows and a quiet, muted mood",
};


export function buildPrompt(style: StyleKey, opts: BuildOpts = {}) {
  const weather = WEATHER_MAP[opts.weather ?? "sunny"];
  const backgroundDesc = BACKGROUND_DESCRIPTIONS[style] ?? BACKGROUND_DESCRIPTIONS.white;
  const tinyPartShadow = opts.addPearlShadow
    ? "Include tiny secondary shadows from small decorative parts."
    : "";

  return `You are an ecommerce photo retoucher.

1ST IMAGE: A product photo of a handmade item taken on a random background.  
2ND IMAGE: ${backgroundDesc}.

TASK: Replace the background in the first image with the same background texture and composition as the second image.  
Then, ${weather}.

Requirements:
- Keep the product from the first image exactly as it is (shape, color, texture)
- Use only the background from the second image (surface texture, lighting style)
- **Imagine placing the product directly onto the reference surface** - the background should look like the same table/paper that the product is naturally sitting on
- If the product appears to be photographed from above at an angle, make the background surface also appear tilted toward the camera in the same way
- The final result should look like one continuous photo where the product was always on that surface
- The output should look like a natural photo with appropriate lighting
- Final image must be cropped square (1:1), with no added objects or borders
${tinyPartShadow ? `- ${tinyPartShadow}` : ''}
  `.trim();
}

// 参考画像を読み込む関数
function loadReferenceImage(style: StyleKey): { base64: string; mime: string } {
  try {
    const imagePath = path.join(process.cwd(), REFERENCE_IMAGES[style]);
    const buffer = fs.readFileSync(imagePath);
    const base64 = buffer.toString('base64');
    const mime = 'image/jpeg'; // すべてjpegと仮定
    return { base64, mime };
  } catch (error) {
    console.error(`Failed to load reference image for style ${style}:`, error);
    throw new Error(`Reference image not found for style: ${style}`);
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
    
    // 参考画像を読み込み
    const referenceImage = loadReferenceImage(style as StyleKey)
    
    const prompt = buildPrompt(style as StyleKey, { weather: weather as WeatherKey })


    // Vertex AI Gemini API呼び出し（2つの画像を送信）
    let result
    try {
      console.log('Calling Vertex AI with model:', 'gemini-2.5-flash-image-preview')
      console.log('Project:', process.env.GOOGLE_CLOUD_PROJECT)
      console.log('Location:', process.env.GOOGLE_CLOUD_LOCATION)
      
      // Base64データの確認
      console.log('Input image MIME:', inputMime)
      console.log('Input base64 length:', inputBase64.length)
      console.log('Input base64 starts with:', inputBase64.substring(0, 20))
      console.log('Reference image MIME:', referenceImage.mime)
      console.log('Reference base64 length:', referenceImage.base64.length)
      console.log('Reference base64 starts with:', referenceImage.base64.substring(0, 20))
      
      result = await generativeModel.generateContent({
        contents: [{
          role: 'user',
          parts: [
            { text: prompt },
            { inlineData: { mimeType: inputMime, data: inputBase64 } },      // 1ST IMAGE: 商品画像
            { inlineData: { mimeType: referenceImage.mime, data: referenceImage.base64 } }, // 2ND IMAGE: 参考画像
          ]
        }],
        generationConfig: {
          maxOutputTokens: 1024,
          temperature: 0.1,
          topP: 0.8,
          // 画像とテキストの両方を出力として指定
          responseModalities: ['TEXT', 'IMAGE']
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
        referenceImageUsed: REFERENCE_IMAGES[style as StyleKey]
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