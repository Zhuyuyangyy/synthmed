<template>
  <div class="app-container">
    <nav class="navbar">
      <div class="nav-brand">
        <span class="brand-icon">🤟</span>
        <span class="brand-text">多模态手语翻译系统</span>
      </div>
      <div class="nav-links">
        <router-link to="/">首页</router-link>
        <router-link to="/translate">双向翻译</router-link>
        <router-link to="/about">关于系统</router-link>
      </div>
      <div class="nav-status">
        <el-tag :type="connected ? 'success' : 'info'" size="small">
          {{ connected ? '已连接' : '未连接' }}
        </el-tag>
      </div>
    </nav>

    <main class="main-content">
      <router-view />
    </main>

    <footer class="footer">
      <p>多模态手语翻译系统 - 基于视觉与生成式AI的双向手语实时翻译 | 延迟 &lt; 1.5秒</p>
    </footer>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'

const connected = ref(false)

onMounted(async () => {
  // 检查后端连接状态
  try {
    const response = await fetch('/api/v1/models/status')
    if (response.ok) {
      connected.value = true
    }
  } catch {
    connected.value = false
  }
})
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
  background: #f5f7fa;
  color: #333;
}

.app-container {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.navbar {
  background: #fff;
  padding: 0 40px;
  height: 64px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  position: sticky;
  top: 0;
  z-index: 100;
}

.nav-brand {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 20px;
  font-weight: 600;
  color: #2563eb;
}

.brand-icon {
  font-size: 28px;
}

.nav-links {
  display: flex;
  gap: 32px;
}

.nav-links a {
  text-decoration: none;
  color: #666;
  font-size: 15px;
  transition: color 0.2s;
}

.nav-links a:hover,
.nav-links a.router-link-active {
  color: #2563eb;
}

.nav-status {
  display: flex;
  align-items: center;
}

.main-content {
  flex: 1;
  padding: 0;
  width: 100%;
}

.footer {
  background: #fff;
  text-align: center;
  padding: 24px;
  color: #999;
  font-size: 14px;
  border-top: 1px solid #eee;
}
</style>
