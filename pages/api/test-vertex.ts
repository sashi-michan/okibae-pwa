import { NextApiRequest, NextApiResponse } from 'next'
import { VertexAI } from '@google-cloud/vertexai'

// Vertex AI クライアントの初期化
const vertexAI = new VertexAI({
  project: process.env.GOOGLE_CLOUD_PROJECT!,
  location: process.env.GOOGLE_CLOUD_LOCATION || 'global',
  apiEndpoint: 'aiplatform.googleapis.com',
});

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ ok: false, error: 'Method not allowed' })
  }

  try {
    console.log('Testing Vertex AI with project:', process.env.GOOGLE_CLOUD_PROJECT)
    console.log('Testing Vertex AI with location:', process.env.GOOGLE_CLOUD_LOCATION)
    
    // まずテキストのみでテスト (正式なVertex AIモデル名)
    const textModel = vertexAI.getGenerativeModel({ model: 'gemini-1.5-flash-001' })
    const textResult = await textModel.generateContent({
      contents: [{
        role: 'user',
        parts: [{ text: 'Hello, can you respond with just "pong"?' }]
      }]
    })
    
    const textResponse = textResult.response?.candidates?.[0]?.content?.parts?.[0]?.text
    console.log('Text model response:', textResponse)
    
    // 次に画像対応モデルでテスト
    const imageModel = vertexAI.preview.getGenerativeModel({ model: 'gemini-2.5-flash-image-preview' })
    const imageResult = await imageModel.generateContent({
      contents: [{
        role: 'user',
        parts: [{ text: 'Hello, can you respond with just "image-pong"?' }]
      }]
    })
    
    const imageResponse = imageResult.response?.candidates?.[0]?.content?.parts?.[0]?.text
    console.log('Image model response:', imageResponse)
    
    return res.json({
      ok: true,
      textModel: textResponse,
      imageModel: imageResponse,
      project: process.env.GOOGLE_CLOUD_PROJECT,
      location: process.env.GOOGLE_CLOUD_LOCATION
    })
    
  } catch (error: any) {
    console.error('Vertex AI Test Error:', error.message)
    console.error('Status:', error?.cause?.status)
    return res.status(500).json({
      ok: false,
      error: error.message,
      status: error?.cause?.status
    })
  }
}