<template>
  <div class="home-page">
    <!-- Hero Section -->
    <section class="hero">
      <div class="hero-content">
        <h1 class="hero-title">
          打破无声世界的<br />
          <span class="highlight">沟通壁垒</span>
        </h1>
        <p class="hero-subtitle">
          基于视觉与生成式AI的双向手语实时翻译系统<br />
          三通道空间注意力融合 · 世界模型动态生成 · 同声传译延迟&lt;1.5秒
        </p>
        <div class="hero-actions">
          <el-button type="primary" size="large" @click="$router.push('/translate')">
            开始翻译
          </el-button>
          <el-button size="large" @click="$router.push('/about')">
            了解更多
          </el-button>
        </div>
      </div>
      <div class="hero-visual">
        <div class="translation-demo">
          <div class="demo-header">
            <span>实时双向翻译</span>
          </div>
          <div class="demo-body">
            <div class="demo-side left">
              <div class="side-label">听障用户</div>
              <div class="camera-placeholder">
                <el-icon :size="40"><VideoCamera /></el-icon>
                <span>摄像头准备中</span>
              </div>
              <div class="sign-text">🤟 你好</div>
            </div>
            <div class="demo-arrow">
              <el-icon :size="32"><ArrowRight /></el-icon>
            </div>
            <div class="demo-side right">
              <div class="side-label">健听用户</div>
              <div class="text-output">你好</div>
              <div class="confidence">置信度: 96.3%</div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- Features Section -->
    <section class="features">
      <h2 class="section-title">核心技术</h2>
      <div class="feature-grid">
        <div class="feature-card">
          <div class="feature-icon">🖐️</div>
          <h3>三通道关键点检测</h3>
          <p>MediaPipe手部21点 + 面部468点 + 身体33点并行检测</p>
        </div>
        <div class="feature-card">
          <div class="feature-icon">🧠</div>
          <h3>空间注意力融合</h3>
          <p>三通道Transformer架构，CSL识别准确率96.3%</p>
        </div>
        <div class="feature-card">
          <div class="feature-icon">🎬</div>
          <h3>世界模型生成</h3>
          <p>文字语义→手语动作序列端到端生成，开放词汇覆盖</p>
        </div>
        <div class="feature-card">
          <div class="feature-icon">⚡</div>
          <h3>同声传译</h3>
          <p>流式处理 + 预测性翻译，端到端延迟&lt;1.5秒</p>
        </div>
      </div>
    </section>

    <!-- Architecture Section -->
    <section class="architecture">
      <h2 class="section-title">系统架构</h2>
      <div class="arch-diagram">
        <div class="arch-row">
          <div class="arch-node input">用户输入</div>
          <div class="arch-arrow">→</div>
          <div class="arch-node process">多模态感知融合</div>
          <div class="arch-arrow">→</div>
          <div class="arch-node output">翻译输出</div>
        </div>
        <div class="arch-channels">
          <div class="channel">🖐️ 手部Pose</div>
          <div class="channel">😊 面部情感</div>
          <div class="channel">🦴 躯干上下文</div>
        </div>
      </div>
    </section>

    <!-- Stats Section -->
    <section class="stats">
      <div class="stat-item">
        <div class="stat-value">2000万+</div>
        <div class="stat-label">服务听障人群</div>
      </div>
      <div class="stat-item">
        <div class="stat-value">&lt;1.5秒</div>
        <div class="stat-label">翻译延迟</div>
      </div>
      <div class="stat-item">
        <div class="stat-value">96.3%</div>
        <div class="stat-label">CSL识别准确率</div>
      </div>
      <div class="stat-item">
        <div class="stat-value">开放词汇</div>
        <div class="stat-label">手语视频生成</div>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { VideoCamera, ArrowRight } from '@element-plus/icons-vue'
</script>

<style scoped>
.home-page {
  width: 100%;
}

.hero {
  display: flex;
  align-items: center;
  gap: 60px;
  padding: 60px 80px;
  background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
}

.hero-title {
  font-size: 48px;
  line-height: 1.2;
  font-weight: 700;
  color: #1a1a2e;
}

.highlight {
  color: #2563eb;
}

.hero-subtitle {
  margin-top: 24px;
  font-size: 16px;
  color: #666;
  line-height: 1.8;
}

.hero-actions {
  margin-top: 32px;
  display: flex;
  gap: 16px;
}

.translation-demo {
  background: #fff;
  border-radius: 16px;
  box-shadow: 0 8px 32px rgba(0,0,0,0.08);
  overflow: hidden;
  width: 480px;
}

.demo-header {
  background: #2563eb;
  padding: 12px 20px;
  color: #fff;
  font-weight: 500;
}

.demo-body {
  padding: 24px;
  display: flex;
  align-items: center;
  gap: 16px;
}

.demo-side {
  flex: 1;
  text-align: center;
}

.side-label {
  font-size: 12px;
  color: #999;
  margin-bottom: 12px;
}

.camera-placeholder {
  background: #f5f7fa;
  border-radius: 8px;
  padding: 24px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  color: #999;
  font-size: 12px;
}

.sign-text {
  margin-top: 12px;
  font-size: 24px;
}

.text-output {
  background: #f0f9ff;
  border-radius: 8px;
  padding: 24px;
  font-size: 20px;
  color: #2563eb;
  font-weight: 500;
}

.confidence {
  margin-top: 8px;
  font-size: 12px;
  color: #16a34a;
}

.demo-arrow {
  color: #2563eb;
}

.section-title {
  text-align: center;
  font-size: 32px;
  font-weight: 600;
  color: #1a1a2e;
  margin-bottom: 48px;
}

.features {
  padding: 60px 80px;
  background: #fff;
}

.feature-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 24px;
}

.feature-card {
  background: #f8fafc;
  padding: 32px 24px;
  border-radius: 12px;
  text-align: center;
}

.feature-icon {
  font-size: 40px;
  margin-bottom: 16px;
}

.feature-card h3 {
  font-size: 16px;
  margin-bottom: 8px;
}

.feature-card p {
  font-size: 14px;
  color: #666;
  line-height: 1.6;
}

.architecture {
  padding: 60px 80px;
  background: #f5f7fa;
}

.arch-diagram {
  background: #fff;
  border-radius: 16px;
  padding: 40px;
  text-align: center;
}

.arch-row {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 20px;
  margin-bottom: 32px;
}

.arch-node {
  padding: 16px 32px;
  border-radius: 8px;
  font-weight: 500;
}

.arch-node.input {
  background: #dbeafe;
  color: #1e40af;
}

.arch-node.process {
  background: #fef3c7;
  color: #92400e;
}

.arch-node.output {
  background: #dcfce7;
  color: #166534;
}

.arch-arrow {
  font-size: 24px;
  color: #999;
}

.arch-channels {
  display: flex;
  justify-content: center;
  gap: 24px;
}

.channel {
  background: #f0f9ff;
  padding: 12px 24px;
  border-radius: 20px;
  font-size: 14px;
  color: #0369a1;
}

.stats {
  display: flex;
  justify-content: center;
  gap: 80px;
  padding: 60px 80px;
  background: #fff;
}

.stat-item {
  text-align: center;
}

.stat-value {
  font-size: 36px;
  font-weight: 700;
  color: #2563eb;
}

.stat-label {
  margin-top: 8px;
  font-size: 14px;
  color: #666;
}
</style>
