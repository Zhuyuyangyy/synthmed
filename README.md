# SynthMed - AI医学影像合成数据平台

> 基于扩散模型(Latent Diffusion Model)的科研级医学影像合成平台

## 🎯 项目定位

**通用科研基础设施** — 不仅是单一工具，而是为医学AI研究提供合成数据服务。

## 🔬 五大核心模块

### 第一阶段：核心算法 — 医学影像梦工厂
- **Latent Diffusion Model (LDM)** — 高质量影像生成
- **ControlNet** — 解剖学拓扑约束
- **多模态一致性损失** — MRI-CT跨模态对齐
- **解剖学拓扑约束** — 分割掩码引导病灶生成

### 第二阶段：后端中枢 — 高并发算力调度
- **Spring Boot 3** — 任务调度
- **Redis** — 缓存/会话管理
- **RabbitMQ** — 异步任务队列
- **MinIO** — 海量影像存储

### 第三阶段：前端交互 — 零代码影像编辑器
- **Vue 3 + Ant Design Vue** — 专业医疗工作站UI
- **交互式涂鸦生成** — 画掩码→生影像
- **ECharts监控面板** — GPU/进度实时监控

### 第四阶段：3D验证 — 全方位解剖校对
- **Three.js体绘制** — 2D切片→3D模型
- **多平面重建(MPR)** — 冠状/矢状/横断面联动
- **切片漂移检测** — 验证3D解剖一致性

### 第五阶段：SCI实验验证 — 以假乱真
- **专家图灵测试** — 真假影像盲评
- **下游任务提升** — 分割网络上对比实验
- **消融实验** — 语义约束有效性验证

## 📊 技术栈

```
AI算法:     PyTorch, Diffusers, ControlNet, OpenCV
后端:       Spring Boot 3, Redis, RabbitMQ, FastAPI
前端:       Vue 3, Ant Design Vue, ECharts, Three.js
存储:       MinIO, MySQL, LocalFS
GPU:        CUDA, Multi-GPU调度
```

## 🚀 快速启动

```bash
cd backend
pip install -r requirements.txt
python app.py  # 端口8015

cd frontend
npm install && npm run dev  # 端口5174
```

## 🎯 应用场景

- [ ] 口腔牙齿CBCT/CT影像合成
- [ ] 肺部结节CT/MRI合成
- [ ] 脑部MRI影像合成
- [ ] 超声影像合成
- [ ] 中医舌象/面象合成
