<template>
  <div class="about-page">
    <div class="about-header">
      <h1>关于多模态手语翻译系统</h1>
      <p class="subtitle">基于视觉与生成式AI的双向手语实时翻译系统</p>
    </div>

    <!-- 项目背景 -->
    <section class="section">
      <h2>项目背景</h2>
      <div class="content">
        <p>据第二次全国残疾人抽样调查，我国听障人士超过2000万，手语是他们融入社会的主要工具。然而，手语普及面临"双向鸿沟"：普通人不懂手语，手语使用者阅读文字也有障碍。</p>
        <p>现有翻译工具多为单向（文字→手语视频），且生成的只是一个"通用手语视频"，缺乏自然交互感。</p>
      </div>
    </section>

    <!-- 核心技术 -->
    <section class="section">
      <h2>核心技术</h2>
      <div class="tech-grid">
        <div class="tech-card">
          <div class="tech-icon">🖐️</div>
          <h3>三通道关键点检测</h3>
          <ul>
            <li>MediaPipe手部21点关键点 × 2</li>
            <li>面部468点关键点 + 52个blendshape</li>
            <li>身体33点姿态关键点</li>
          </ul>
        </div>
        <div class="tech-card">
          <div class="tech-icon">🧠</div>
          <h3>三通道空间注意力融合</h3>
          <ul>
            <li>并行Transformer架构</li>
            <li>手-面-体特征独立编码</li>
            <li>跨通道空间注意力融合</li>
            <li>CSL准确率96.3%</li>
          </ul>
        </div>
        <div class="tech-card">
          <div class="tech-icon">🎬</div>
          <h3>世界模型手语生成</h3>
          <ul>
            <li>文字语义→手语动作序列</li>
            <li>端到端可控生成</li>
            <li>面部表情+口型同步</li>
            <li>开放词汇覆盖</li>
          </ul>
        </div>
        <div class="tech-card">
          <div class="tech-icon">⚡</div>
          <h3>同声传译</h3>
          <ul>
            <li>流式处理架构</li>
            <li>预测性翻译</li>
            <li>端到端延迟&lt;1.5秒</li>
            <li>WebRTC实时通信</li>
          </ul>
        </div>
      </div>
    </section>

    <!-- 技术架构图 -->
    <section class="section">
      <h2>技术架构</h2>
      <div class="arch-container">
        <div class="arch-flow">
          <div class="flow-box input">
            <div class="flow-title">用户A（听障）</div>
            <div class="flow-desc">手语视频采集</div>
          </div>
          <div class="flow-arrow">→</div>
          <div class="flow-box process">
            <div class="flow-title">多模态感知融合引擎</div>
            <div class="flow-channels">
              <span>🖐️ 手部Pose</span>
              <span>😊 面部情感</span>
              <span>🦴 躯干上下文</span>
            </div>
          </div>
          <div class="flow-arrow">→</div>
          <div class="flow-box output">
            <div class="flow-title">用户B（健听）</div>
            <div class="flow-desc">文字/语音输出</div>
          </div>
        </div>
        <div class="arch-note">
          <p>技术栈：Python + MediaPipe + OpenPose + LLMWerModel + Stable Diffusion/T2I + FastAPI + WebRTC</p>
        </div>
      </div>
    </section>

    <!-- 创新点 -->
    <section class="section">
      <h2>创新点</h2>
      <div class="innovation-list">
        <div class="innovation-item">
          <div class="innovation-num">①</div>
          <div class="innovation-content">
            <h3>手-面-体三通道空间注意力融合</h3>
            <p>现有手语识别方法多依赖单一手部关键点，忽略面部表情在手语语法中的关键作用。本项目首次提出三通道并行Transformer架构，在CSL数据集上准确率达96.3%。</p>
          </div>
        </div>
        <div class="innovation-item">
          <div class="innovation-num">②</div>
          <div class="innovation-content">
            <h3>基于世界模型的动态手语视频生成</h3>
            <p>传统文字转手语视频依赖预录动作库，动作生硬、词汇覆盖率低。本项目集成轻量级世界模型，实现文字语义到手语动作序列的端到端生成，词汇覆盖从2万词级提升至开放词汇级别。</p>
          </div>
        </div>
        <div class="innovation-item">
          <div class="innovation-num">③</div>
          <div class="innovation-content">
            <h3>双向同声传译（延迟&lt;1.5秒）</h3>
            <p>通过流式处理+预测性翻译，将端到端延迟压缩至1.5秒以内，实现真正意义上的"同声传译"，而非"先说完再翻译"的等待式翻译。</p>
          </div>
        </div>
      </div>
    </section>

    <!-- 应用场景 -->
    <section class="section">
      <h2>应用场景</h2>
      <div class="scene-grid">
        <div class="scene-card">
          <div class="scene-icon">🏥</div>
          <h3>公共服务场景</h3>
          <p>医院、银行、政务大厅的无障碍服务终端</p>
        </div>
        <div class="scene-card">
          <div class="scene-icon">🎓</div>
          <h3>在线教育</h3>
          <p>手语教育视频的自动化生成</p>
        </div>
        <div class="scene-card">
          <div class="scene-icon">📱</div>
          <h3>社交应用</h3>
          <p>实时视频通话的同声翻译</p>
        </div>
        <div class="scene-card">
          <div class="scene-icon">🏛️</div>
          <h3>政策驱动</h3>
          <p>国家《无障碍环境建设条例》强制要求</p>
        </div>
      </div>
    </section>

    <!-- 社会价值 -->
    <section class="section highlight-section">
      <h2>社会价值</h2>
      <div class="impact-stats">
        <div class="impact-item">
          <div class="impact-value">2000万+</div>
          <div class="impact-label">听障人士</div>
        </div>
        <div class="impact-item">
          <div class="impact-value">百亿元</div>
          <div class="impact-label">潜在经济价值/年</div>
        </div>
      </div>
      <p class="impact-desc">促进2000万听障人士的社会融入，构建更加包容的无障碍社会环境</p>
    </section>
  </div>
</template>

<style scoped>
.about-page {
  max-width: 1200px;
  margin: 0 auto;
  padding: 40px;
}

.about-header {
  text-align: center;
  margin-bottom: 60px;
}

.about-header h1 {
  font-size: 36px;
  color: #1a1a2e;
}

.subtitle {
  margin-top: 12px;
  color: #666;
  font-size: 18px;
}

.section {
  margin-bottom: 60px;
}

.section h2 {
  font-size: 28px;
  color: #1a1a2e;
  margin-bottom: 24px;
  padding-bottom: 12px;
  border-bottom: 2px solid #2563eb;
}

.content p {
  color: #666;
  line-height: 1.8;
  margin-bottom: 16px;
}

.tech-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 24px;
}

.tech-card {
  background: #f8fafc;
  padding: 24px;
  border-radius: 12px;
}

.tech-icon {
  font-size: 36px;
  margin-bottom: 12px;
}

.tech-card h3 {
  font-size: 18px;
  margin-bottom: 12px;
}

.tech-card ul {
  list-style: none;
  color: #666;
}

.tech-card li {
  padding: 4px 0;
  font-size: 14px;
}

.arch-container {
  background: #f8fafc;
  padding: 40px;
  border-radius: 16px;
}

.arch-flow {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 20px;
  margin-bottom: 24px;
}

.flow-box {
  padding: 20px 32px;
  border-radius: 12px;
  text-align: center;
}

.flow-box.input {
  background: #dbeafe;
  color: #1e40af;
}

.flow-box.process {
  background: #fef3c7;
  color: #92400e;
}

.flow-box.output {
  background: #dcfce7;
  color: #166534;
}

.flow-title {
  font-weight: 600;
  margin-bottom: 8px;
}

.flow-channels {
  display: flex;
  gap: 12px;
  font-size: 12px;
  margin-top: 8px;
}

.flow-arrow {
  font-size: 24px;
  color: #999;
}

.arch-note {
  text-align: center;
  color: #666;
  font-size: 14px;
}

.innovation-list {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.innovation-item {
  display: flex;
  gap: 20px;
  background: #f8fafc;
  padding: 24px;
  border-radius: 12px;
}

.innovation-num {
  font-size: 36px;
  font-weight: 700;
  color: #2563eb;
  opacity: 0.3;
}

.innovation-content h3 {
  font-size: 18px;
  margin-bottom: 8px;
}

.innovation-content p {
  color: #666;
  line-height: 1.6;
}

.scene-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 24px;
}

.scene-card {
  background: #f8fafc;
  padding: 24px;
  border-radius: 12px;
  text-align: center;
}

.scene-icon {
  font-size: 36px;
  margin-bottom: 12px;
}

.scene-card h3 {
  font-size: 16px;
  margin-bottom: 8px;
}

.scene-card p {
  font-size: 14px;
  color: #666;
}

.highlight-section {
  background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
  padding: 40px;
  border-radius: 16px;
  text-align: center;
}

.impact-stats {
  display: flex;
  justify-content: center;
  gap: 80px;
  margin-bottom: 24px;
}

.impact-item {
  text-align: center;
}

.impact-value {
  font-size: 48px;
  font-weight: 700;
  color: #2563eb;
}

.impact-label {
  font-size: 14px;
  color: #666;
  margin-top: 8px;
}

.impact-desc {
  color: #1a1a2e;
  font-size: 16px;
}
</style>
