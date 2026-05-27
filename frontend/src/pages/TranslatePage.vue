<template>
  <div class="translate-page">
    <!-- 翻译方向选择 -->
    <div class="direction-selector">
      <el-radio-group v-model="direction" size="large">
        <el-radio-button value="bidirectional">
          <el-icon><Connection /></el-icon>
          双向翻译
        </el-radio-button>
        <el-radio-button value="sign_to_text">
          <el-icon><Monitor /></el-icon>
          手语 → 文字
        </el-radio-button>
        <el-radio-button value="text_to_sign">
          <el-icon><ChatLineSquare /></el-icon>
          文字 → 手语
        </el-radio-button>
      </el-radio-group>
    </div>

    <!-- 双向翻译界面 -->
    <div class="translation-container">
      <!-- 左侧：听障用户 -->
      <div class="translation-panel left">
        <div class="panel-header">
          <span class="panel-title">听障用户</span>
          <span class="panel-subtitle">使用手语表达</span>
        </div>

        <!-- 摄像头区域 -->
        <div class="video-area" v-if="direction !== 'text_to_sign'">
          <video
            ref="leftVideoRef"
            class="camera-feed"
            autoplay
            playsinline
            muted
          ></video>
          <canvas ref="leftCanvasRef" class="skeleton-overlay"></canvas>

          <!-- 关键点检测可视化 -->
          <div class="detection-status" v-if="leftDetection">
            <el-tag size="small" type="success">
              🖐️ {{ leftDetection.handCount }}只手
            </el-tag>
            <el-tag size="small" type="success" v-if="leftDetection.faceDetected">
              😊 面部检测
            </el-tag>
          </div>

          <!-- 开始/停止按钮 -->
          <div class="video-controls">
            <el-button
              :type="isCapturingLeft ? 'danger' : 'primary'"
              @click="toggleLeftCapture"
            >
              {{ isCapturingLeft ? '停止' : '开始检测' }}
            </el-button>
          </div>
        </div>

        <!-- 文字输入（text_to_sign模式） -->
        <div class="text-input-area" v-else>
          <el-input
            v-model="rightTextInput"
            type="textarea"
            :rows="4"
            placeholder="输入要转换为手语的文字..."
          />
          <el-button type="primary" @click="generateSignVideo" :loading="generating">
            生成手语视频
          </el-button>
        </div>

        <!-- 翻译结果 -->
        <div class="translation-result">
          <div class="result-label">翻译结果</div>
          <div class="result-text" v-if="recognizedText">
            {{ recognizedText }}
          </div>
          <div class="result-placeholder" v-else>
            等待手语输入...
          </div>
          <div class="result-confidence" v-if="confidence">
            置信度: {{ (confidence * 100).toFixed(1) }}%
          </div>
        </div>
      </div>

      <!-- 中间分隔线 -->
      <div class="divider">
        <div class="divider-line"></div>
        <div class="divider-icon">
          <el-icon :size="24"><Refresh /></el-icon>
        </div>
        <div class="divider-line"></div>
      </div>

      <!-- 右侧：健听用户 -->
      <div class="translation-panel right">
        <div class="panel-header">
          <span class="panel-title">健听用户</span>
          <span class="panel-subtitle">输入文字或查看手语</span>
        </div>

        <!-- 手语视频播放 -->
        <div class="video-area" v-if="direction !== 'sign_to_text'">
          <video
            ref="rightVideoRef"
            class="sign-video"
            controls
            playsinline
          ></video>
          <canvas ref="rightCanvasRef" class="skeleton-overlay"></canvas>

          <!-- 播放状态 -->
          <div class="video-status" v-if="signVideoFrames.length > 0">
            <el-tag size="small">
              正在播放手语视频 ({{ signVideoFrames.length }} 帧)
            </el-tag>
          </div>

          <div class="video-placeholder" v-else>
            <el-icon :size="48"><VideoPlay /></el-icon>
            <span>生成的手语视频将在这里显示</span>
          </div>
        </div>

        <!-- 文字输入（sign_to_text模式） -->
        <div class="text-input-area" v-else>
          <el-input
            v-model="leftTextInput"
            type="textarea"
            :rows="4"
            placeholder="输入文字进行翻译..."
          />
          <el-button type="primary" @click="recognizeFromText">
            翻译
          </el-button>
        </div>

        <!-- 翻译结果 -->
        <div class="translation-result">
          <div class="result-label">翻译结果</div>
          <div class="result-text" v-if="generatedSignText">
            {{ generatedSignText }}
          </div>
          <div class="result-placeholder" v-else>
            等待文字输入...
          </div>
        </div>
      </div>
    </div>

    <!-- 实时状态栏 -->
    <div class="status-bar">
      <div class="status-item">
        <el-icon><Timer /></el-icon>
        <span>延迟: {{ latencyMs }}ms</span>
      </div>
      <div class="status-item">
        <el-icon><Connection /></el-icon>
        <span>WebSocket: {{ wsConnected ? '已连接' : '未连接' }}</span>
      </div>
      <div class="status-item">
        <el-icon><Cpu /></el-icon>
        <span>模型: {{ modelStatus }}</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { ElMessage } from 'element-plus'
import {
  Connection, Monitor, ChatLineSquare,
  Refresh, VideoPlay, Timer, Cpu
} from '@element-plus/icons-vue'
import axios from 'axios'

// 翻译方向
const direction = ref<'bidirectional' | 'sign_to_text' | 'text_to_sign'>('bidirectional')

// 视频引用
const leftVideoRef = ref<HTMLVideoElement>()
const leftCanvasRef = ref<HTMLCanvasElement>()
const rightVideoRef = ref<HTMLVideoElement>()
const rightCanvasRef = ref<HTMLCanvasElement>()

// 捕获状态
const isCapturingLeft = ref(false)
let leftStream: MediaStream | null = null
let animationFrameId: number | null = null

// 检测结果
const leftDetection = ref<{
  handCount: number
  faceDetected: boolean
  bodyDetected: boolean
} | null>(null)

// 翻译结果
const recognizedText = ref('')
const generatedSignText = ref('')
const confidence = ref<number | null>(null)

// 手语视频
const signVideoFrames = ref<string[]>([])
const generating = ref(false)

// 文字输入
const leftTextInput = ref('')
const rightTextInput = ref('')

// WebSocket
const wsConnected = ref(false)
let ws: WebSocket | null = null

// 状态
const latencyMs = ref(0)
const modelStatus = ref('准备中')

// API基础URL
const API_BASE = '/api/v1'

onMounted(async () => {
  // 获取模型状态
  try {
    const response = await axios.get(`${API_BASE}/models/status`)
    modelStatus.value = response.data.recognizer?.loaded ? '已加载' : '演示模式'
  } catch {
    modelStatus.value = '离线'
  }
})

onUnmounted(() => {
  stopLeftCapture()
  disconnectWebSocket()
})

// 切换左侧捕获
async function toggleLeftCapture() {
  if (isCapturingLeft.value) {
    stopLeftCapture()
  } else {
    await startLeftCapture()
  }
}

// 开始左侧视频捕获
async function startLeftCapture() {
  try {
    leftStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' }
    })

    if (leftVideoRef.value) {
      leftVideoRef.value.srcObject = leftStream
      await leftVideoRef.value.play()
    }

    isCapturingLeft.value = true
    connectWebSocket()
    startProcessingLoop()

    ElMessage.success('摄像头已启动')
  } catch (error) {
    ElMessage.error('无法访问摄像头')
    console.error(error)
  }
}

// 停止左侧捕获
function stopLeftCapture() {
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId)
    animationFrameId = null
  }

  if (leftStream) {
    leftStream.getTracks().forEach(track => track.stop())
    leftStream = null
  }

  if (leftVideoRef.value) {
    leftVideoRef.value.srcObject = null
  }

  isCapturingLeft.value = false
  disconnectWebSocket()
}

// 连接WebSocket
function connectWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const wsUrl = `${protocol}//${window.location.host}/ws/stream`

  ws = new WebSocket(wsUrl)

  ws.onopen = () => {
    wsConnected.value = true
    console.log('WebSocket connected')
  }

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data)

    if (data.type === 'interim') {
      leftDetection.value = {
        handCount: data.hand_count || 0,
        faceDetected: data.face_detected || false,
        bodyDetected: data.body_detected || false
      }
    } else if (data.type === 'final') {
      recognizedText.value = data.text || ''
      confidence.value = data.confidence || null
      latencyMs.value = Math.round(performance.now() - (data.start_time || 0))
    }
  }

  ws.onclose = () => {
    wsConnected.value = false
  }

  ws.onerror = (error) => {
    console.error('WebSocket error:', error)
    wsConnected.value = false
  }
}

// 断开WebSocket
function disconnectWebSocket() {
  if (ws) {
    ws.close()
    ws = null
  }
  wsConnected.value = false
}

// 开始处理循环
function startProcessingLoop() {
  const canvas = leftCanvasRef.value
  const video = leftVideoRef.value

  if (!canvas || !video) return

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  function processFrame() {
    if (!isCapturingLeft.value || !video || !ctx) return

    // 设置canvas尺寸
    canvas.width = video.videoWidth || 640
    canvas.height = video.videoHeight || 480

    // 绘制视频帧
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

    // 发送帧到服务器（每100ms发送一次）
    if (ws && ws.readyState === WebSocket.OPEN && video.readyState >= 2) {
      const timestamp = performance.now()
      const frameData = getFrameBase64()

      ws.send(JSON.stringify({
        type: 'frame',
        frame: frameData,
        timestamp_ms: timestamp
      }))
    }

    animationFrameId = requestAnimationFrame(processFrame)
  }

  processFrame()
}

// 获取帧的base64
function getFrameBase64(): string {
  const canvas = leftCanvasRef.value
  if (!canvas) return ''

  return canvas.toDataURL('image/jpeg', 0.7).split(',')[1]
}

// 文字转手语视频
async function generateSignVideo() {
  if (!rightTextInput.value.trim()) {
    ElMessage.warning('请输入要转换的文字')
    return
  }

  generating.value = true

  try {
    const response = await axios.post(`${API_BASE}/generate`, {
      text: rightTextInput.value,
      format: 'base64_frames'
    })

    if (response.data.status === 'success') {
      signVideoFrames.value = response.data.frames || []
      generatedSignText.value = rightTextInput.value
      ElMessage.success('手语视频生成成功')
    } else {
      ElMessage.error(response.data.error || '生成失败')
    }
  } catch (error) {
    ElMessage.error('生成手语视频失败')
    console.error(error)
  } finally {
    generating.value = false
  }
}

// 从文字识别（模拟）
async function recognizeFromText() {
  if (!leftTextInput.value.trim()) {
    ElMessage.warning('请输入要翻译的文字')
    return
  }

  try {
    const response = await axios.post(`${API_BASE}/recognize`, {
      frames: [] // 演示模式
    })

    recognizedText.value = leftTextInput.value
    confidence.value = 0.96
  } catch (error) {
    ElMessage.error('识别失败')
  }
}
</script>

<style scoped>
.translate-page {
  display: flex;
  flex-direction: column;
  height: calc(100vh - 64px);
  background: #f5f7fa;
}

.direction-selector {
  padding: 20px 40px;
  background: #fff;
  border-bottom: 1px solid #eee;
  display: flex;
  justify-content: center;
}

.translation-container {
  flex: 1;
  display: flex;
  padding: 24px 40px;
  gap: 24px;
}

.translation-panel {
  flex: 1;
  background: #fff;
  border-radius: 16px;
  padding: 24px;
  display: flex;
  flex-direction: column;
  box-shadow: 0 2px 12px rgba(0,0,0,0.04);
}

.panel-header {
  margin-bottom: 16px;
}

.panel-title {
  font-size: 18px;
  font-weight: 600;
  color: #1a1a2e;
}

.panel-subtitle {
  display: block;
  font-size: 12px;
  color: #999;
  margin-top: 4px;
}

.video-area {
  flex: 1;
  position: relative;
  background: #000;
  border-radius: 12px;
  overflow: hidden;
  min-height: 300px;
}

.camera-feed,
.sign-video {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.skeleton-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}

.detection-status,
.video-status {
  position: absolute;
  top: 12px;
  left: 12px;
  display: flex;
  gap: 8px;
}

.video-controls {
  position: absolute;
  bottom: 12px;
  left: 50%;
  transform: translateX(-50%);
}

.video-placeholder {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: #666;
  gap: 12px;
  background: #f5f7fa;
}

.text-input-area {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.translation-result {
  margin-top: 16px;
  padding: 16px;
  background: #f8fafc;
  border-radius: 8px;
}

.result-label {
  font-size: 12px;
  color: #999;
  margin-bottom: 8px;
}

.result-text {
  font-size: 20px;
  font-weight: 500;
  color: #2563eb;
}

.result-placeholder {
  color: #999;
  font-size: 14px;
}

.result-confidence {
  margin-top: 8px;
  font-size: 12px;
  color: #16a34a;
}

.divider {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 0 8px;
}

.divider-line {
  flex: 1;
  width: 2px;
  background: linear-gradient(to bottom, transparent, #ddd, transparent);
}

.divider-icon {
  padding: 16px 0;
  color: #2563eb;
}

.status-bar {
  padding: 12px 40px;
  background: #fff;
  border-top: 1px solid #eee;
  display: flex;
  gap: 32px;
}

.status-item {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  color: #666;
}
</style>
