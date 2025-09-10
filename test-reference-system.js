const FormData = require('form-data');
const fs = require('fs');
const { default: fetch } = require('node-fetch');

async function testReferenceSystem() {
  console.log('🧪 Testing Reference Image System...');
  
  const combinations = [
    { style: 'white', weather: 'sunny' },
    { style: 'linen', weather: 'cloudy' },
    { style: 'concrete', weather: 'rainy' }
  ];

  for (const combo of combinations) {
    try {
      console.log(`\n📸 Testing ${combo.style} + ${combo.weather}...`);
      
      // Create form data
      const form = new FormData();
      form.append('file', fs.createReadStream('./sample/test_bg1.jpg'));
      form.append('style', combo.style);
      form.append('weather', combo.weather);

      const response = await fetch('http://localhost:3001/api/ai-shadows', {
        method: 'POST',
        body: form
      });

      const result = await response.json();

      if (result.ok) {
        console.log(`✅ ${combo.style} + ${combo.weather}: SUCCESS`);
        console.log(`   Reference: ${result.debug.referenceImageUsed}`);
        console.log(`   Output size: ${Math.round(result.debug.outputLength / 1024)}KB`);
        
        // Save result for verification
        const base64Data = result.imageBase64.split(',')[1];
        const buffer = Buffer.from(base64Data, 'base64');
        fs.writeFileSync(`./test_${combo.style}_${combo.weather}.jpg`, buffer);
        console.log(`   Saved as: test_${combo.style}_${combo.weather}.jpg`);
      } else {
        console.log(`❌ ${combo.style} + ${combo.weather}: FAILED`);
        console.log(`   Error: ${result.error}`);
      }
      
      // Wait between requests to avoid rate limiting
      await new Promise(resolve => setTimeout(resolve, 2000));
      
    } catch (error) {
      console.log(`❌ ${combo.style} + ${combo.weather}: ERROR`);
      console.log(`   ${error.message}`);
    }
  }
  
  console.log('\n🎉 Reference system test completed!');
}

testReferenceSystem().catch(console.error);