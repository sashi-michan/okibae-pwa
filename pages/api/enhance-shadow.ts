import { NextApiRequest, NextApiResponse } from 'next'
import { spawn } from 'child_process'
import path from 'path'

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ ok: false, error: 'Method not allowed' })
  }

  try {
    const { cutoutImageBase64, backgroundColor, lightDirection } = req.body

    if (!cutoutImageBase64) {
      return res.status(400).json({ ok: false, error: 'Missing cutoutImageBase64' })
    }

    // Prepare input for Python script
    const inputData = {
      cutoutImageBase64,
      backgroundColor: backgroundColor || '#FFFFFF',
      lightDirection
    }

    // Execute Python script
    const result = await executePythonScript(inputData)
    
    if (result.ok) {
      res.json(result)
    } else {
      res.status(500).json(result)
    }

  } catch (error: any) {
    console.error('enhance-shadow error:', error)
    res.status(500).json({ ok: false, error: error.message || 'Internal server error' })
  }
}

function executePythonScript(inputData: any): Promise<any> {
  return new Promise((resolve, reject) => {
    const scriptPath = path.join(process.cwd(), 'python', 'shadow_enhancer.py')
    const python = spawn('python', [scriptPath])
    
    let stdout = ''
    let stderr = ''
    
    python.stdout.on('data', (data) => {
      stdout += data.toString()
    })
    
    python.stderr.on('data', (data) => {
      stderr += data.toString()
    })
    
    python.on('close', (code) => {
      if (code !== 0) {
        console.error('Python script error:', stderr)
        resolve({ ok: false, error: `Python script failed with code ${code}: ${stderr}` })
        return
      }
      
      try {
        const result = JSON.parse(stdout)
        resolve(result)
      } catch (parseError) {
        console.error('Failed to parse Python output:', stdout)
        resolve({ ok: false, error: 'Failed to parse Python script output' })
      }
    })
    
    python.on('error', (error) => {
      console.error('Failed to start Python script:', error)
      resolve({ ok: false, error: `Failed to start Python script: ${error.message}` })
    })
    
    // Send input data to Python script
    python.stdin.write(JSON.stringify(inputData))
    python.stdin.end()
  })
}