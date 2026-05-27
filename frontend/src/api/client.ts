import axios from 'axios'

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 30000
})

// 手语识别
export async function recognizeSignLanguage(frames: string[]) {
  const response = await api.post('/recognize', frames)
  return response.data
}

// 文字转手语视频
export async function generateSignVideo(text: string, format: string = 'base64_frames') {
  const response = await api.post('/generate', null, {
    params: { text, format }
  })
  return response.data
}

// 双向翻译
export async function translate(
  direction: 'sign_to_text' | 'text_to_sign',
  frames?: string[],
  text?: string
) {
  const response = await api.post('/translate', null, {
    params: { direction, frames, text }
  })
  return response.data
}

// 获取模型状态
export async function getModelStatus() {
  const response = await api.get('/models/status')
  return response.data
}

// 获取CSL词汇列表
export async function getCSLGlosses() {
  const response = await api.get('/csl/glosses')
  return response.data
}

export default api
