const { GoogleGenAI } = require('@google/genai');
const fs = require('fs');
require('dotenv').config();

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

async function testDirectAPI() {
  try {
    console.log('Reading test image...');
    const buffer = fs.readFileSync('./sample/test_bg1.jpg');
    const base64 = buffer.toString('base64');
    const mime = 'image/jpeg';

    const prompt = `You are an ecommerce photo retoucher. Generate a clean, soft ambient product photo with an airy mood. The product is placed near a window in a minimal interior; only the surface beneath the item is visible.
The item sits on textured white watercolor / cold-press drawing paper that lies perfectly flat (no curling or sheet edges). Preserve the subtle paper tooth and dimples.
Use one large diffused window key (sheer curtains/scrim) with a broad wall bounce; no other lights. Overcast daylight; keep lighting even.
Exactly one shadow; infer its direction from the product. The shadow is very faint with wide feathered edges and a tight contact shadow under touch points.
Camera: slight top-down (20–30°), short-telephoto look. Square 1:1, edge-to-edge.`;

    console.log('Calling Gemini API with direct prompt...');
    console.log('Prompt:', prompt);

    const result = await ai.models.generateContent({
      model: 'gemini-2.5-flash-image-preview',
      contents: [
        prompt,
        { inlineData: { mimeType: mime, data: base64 } },
      ],
    });

    console.log('API response received');

    const parts = result.candidates?.[0]?.content?.parts ?? [];
    const imagePart = parts.find((p) => p.inlineData && p.inlineData.mimeType?.startsWith('image/'));
    
    if (!imagePart) {
      console.error('No image in response:', result);
      return;
    }

    const outB64 = imagePart.inlineData?.data;
    if (!outB64) {
      console.error('No image data in response');
      return;
    }

    // Save result
    const outputBuffer = Buffer.from(outB64, 'base64');
    fs.writeFileSync('./direct_api_test_result.jpg', outputBuffer);
    
    console.log('✅ Success! Result saved as direct_api_test_result.jpg');
    console.log('Output image size:', outputBuffer.length, 'bytes');

  } catch (error) {
    console.error('❌ API test failed:', error);
  }
}

testDirectAPI();