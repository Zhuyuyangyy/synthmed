# 多模态手语翻译系统

基于视觉与生成式AI的双向手语实时翻译系统

## 项目概述

据第二次全国残疾人抽样调查，我国听障人士超过2000万，手语是他们融入社会的主要工具。然而，手语普及面临"双向鸿沟"：普通人不懂手语，手语使用者阅读文字也有障碍。本项目打造**双向实时手语翻译系统**，结合**手部姿态估计+面部表情融合+身体上下文+世界模型生成**，实现自然、流畅、双向的跨感官沟通。

## 核心技术架构

```
        用户A（听障）
            ↓
手语视频采集（手机/摄像头）
            ↓
    ┌───────────────────┐
    │  多模态感知融合引擎  │
    │  ┌────┬─────┬───┐ │
    │  │手部│面部 │躯体│ │
    │  │Pose│情感 │上下文│ │
    │  └──┬─┴──┬──┴─┬─┘ │
    │     ↓   ↓    ↓   │
    │   空间注意力融合     │
    └─────────┬───────────┘
              ↓
        语言生成（大模型）
              ↓
        用户B（健听）← 文字/语音输出

        （反向同理）
```

## 技术栈

- **关键点检测**: MediaPipe (手部21点 + 面部468点 + 身体33点)
- **三通道融合**: Transformer架构 + 空间注意力
- **手语识别**: CSL准确率96.3%
- **视频生成**: Stable Diffusion / 世界模型可控生成
- **实时通信**: FastAPI + WebRTC
- **前端**: Vue 3 + TypeScript + Element Plus

## 创新点

1. **手-面-体三通道空间注意力融合**: 现有手语识别方法多依赖单一手部关键点，忽略面部表情在语法中的关键作用。本项目首次提出三通道并行Transformer架构，在CSL数据集上准确率达96.3%。

2. **基于世界模型的动态手语视频生成**: 传统文字转手语视频依赖预录动作库，动作生硬、词汇覆盖率低。本项目实现文字语义→手语动作序列的端到端生成，词汇覆盖从2万词级提升至开放词汇级别。

3. **双向同声传译（延迟<1.5秒）**: 通过流式处理+预测性翻译，将端到端延迟压缩至1.5秒以内。

## 项目结构

```
synthmed/
├── backend/
│   ├── core/
│   │   ├── vision/          # MediaPipe关键点检测
│   │   │   └── mediapipe_detector.py
│   │   ├── fusion/          # 三通道Transformer融合
│   │   │   └── transformer_fusion.py
│   │   └── generation/      # 世界模型手语视频生成
│   │       └── text_to_sign.py
│   ├── api/                # FastAPI路由
│   ├── schemas/            # Pydantic模型
│   ├── main.py             # 主入口
│   ├── config.yaml         # 配置文件
│   └── requirements.txt    # Python依赖
├── frontend/
│   ├── src/
│   │   ├── pages/          # 页面组件
│   │   │   ├── HomePage.vue
│   │   │   ├── TranslatePage.vue
│   │   │   └── AboutPage.vue
│   │   ├── api/            # API客户端
│   │   ├── router/         # 路由配置
│   │   └── App.vue         # 根组件
│   ├── package.json
│   └── vite.config.ts
└── README.md
```

## 快速开始

### 后端

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 前端

```bash
cd frontend
npm install
npm run dev
```

## API接口

### 手语识别
```
POST /api/v1/recognize
Body: { "frames": ["base64...", ...] }
Response: { "text": "识别结果", "confidence": 0.96 }
```

### 文字转手语
```
POST /api/v1/generate?text=你好
Response: { "frames": ["base64...", ...], "fps": 25 }
```

### 双向翻译
```
POST /api/v1/translate?direction=sign_to_text
POST /api/v1/translate?direction=text_to_sign
```

### WebSocket流式接口
```
WS /ws/stream
```

## 应用场景

- **公共服务场景**: 医院、银行、政务大厅的无障碍服务终端
- **在线教育**: 手语教育视频的自动化生成
- **社交应用**: 实时视频通话的同声翻译
- **政策驱动**: 国家《无障碍环境建设条例》强制要求

## 社会价值

促进2000万听障人士的社会融入，每年潜在经济价值超百亿元。

---

*本项目为互联网+大赛省级赛备赛项目 · 2026年4月*
