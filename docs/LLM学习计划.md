# LLM 学习计划 - 程序员版

> 8 周系统学习路径，从原理到实践

---

## 📅 整体规划

| 阶段 | 周数 | 目标 | 产出 |
|------|------|------|------|
| **Phase 1** | 1-2 周 | 理解 Transformer 核心原理 | 能手写简化版注意力机制 |
| **Phase 2** | 3-4 周 | 掌握 LLM 训练全流程 | 读完 5 篇核心论文 |
| **Phase 3** | 5-6 周 | 动手实践 | 微调一个开源模型 |
| **Phase 4** | 7-8 周 | 工程化能力 | 搭建一个完整的 LLM 应用 |

---

## 🎯 Week 1：Transformer 基础

### 学习任务
- [ ] 阅读 [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)（2 小时）
- [ ] 理解 Self-Attention 的数学公式
- [ ] 理解 Multi-Head Attention 的设计思想
- [ ] 了解 Position Encoding 的作用

### 动手任务
- [ ] 用 NumPy 实现简化版的 Self-Attention（不使用框架）
- [ ] 对比不同位置编码方案（Sinusoidal vs Learned）

### 验证标准
- 能手绘 Transformer 架构图
- 能解释 Attention 计算的时间复杂度

---

## 🎯 Week 2：从零实现 GPT

### 学习任务
- [ ] 观看 Andrej Karpathy [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nI)（2 小时）
- [ ] 理解 Tokenizer（BPE 算法）
- [ ] 理解语言建模的训练目标（Next Token Prediction）

### 动手任务
- [ ] 跟着视频完成 nanoGPT 实现
- [ ] 用自己的数据集训练一个小模型（如莎士比亚文本）
- [ ] 尝试不同的模型规模，观察生成质量变化

### 代码重点
```python
# 核心理解目标
1. Token embedding 和 Position embedding
2. Masked Self-Attention（因果注意力）
3. Layer Norm 和残差连接
4. 温度采样和 Top-k 采样
```

---

## 🎯 Week 3：预训练与 Scaling

### 阅读任务
- [ ] 论文：GPT-3 [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [ ] 论文：[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)

### 关键理解点
- [ ] 什么是 In-Context Learning？
- [ ] 模型性能如何随参数/数据/算力缩放？
- [ ] 为什么大模型会"涌现"新能力？

### 思考题
1. GPT-3 的 175B 参数如何估算训练成本？
2. Few-shot、One-shot、Zero-shot 的区别是什么？

---

## 🎯 Week 4：对齐技术（Alignment）

### 阅读任务
- [ ] 论文：InstructGPT [Training language models to follow instructions](https://arxiv.org/abs/2203.02155)
- [ ] 论文：RLHF 详解 [Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325)

### 关键理解点
- [ ] 什么是 RLHF（人类反馈强化学习）？
- [ ] SFT（监督微调）vs RLHF 的作用分别是什么？
- [ ] 奖励模型（Reward Model）如何训练？

### 动手任务
- [ ] 用 OpenAI API 对比 base 模型和 instruct 模型的区别
- [ ] 尝试设计一个 prompt，让模型输出符合预期的格式

---

## 🎯 Week 5：推理与 Prompt 工程

### 学习任务
- [ ] 论文：Chain-of-Thought [Chain-of-Thought Prompting Elicits Reasoning](https://arxiv.org/abs/2201.11903)
- [ ] 学习 Prompt Engineering 最佳实践

### 关键理解点
- [ ] CoT 为什么能提升推理能力？
- [ ] Zero-shot CoT vs Few-shot CoT
- [ ] Self-Consistency 解码策略

### 动手任务
- [ ] 在 GSM8K（数学题数据集）上测试不同 prompting 策略
- [ ] 实现一个自动 CoT 生成器

---

## 🎯 Week 6：开源模型与微调

### 学习任务
- [ ] 了解 Llama、Qwen、ChatGLM 等开源模型架构
- [ ] 学习 LoRA / QLoRA 微调原理
- [ ] 了解模型量化（INT8/INT4）

### 动手任务
- [ ] 使用 Hugging Face PEFT 库进行 LoRA 微调
- [ ] 准备一个领域数据集（如技术文档、客服对话）
- [ ] 对比全参数微调 vs LoRA 的效果和成本

### 推荐工具
```bash
# 环境准备
pip install transformers peft bitsandbytes accelerate

# 关键库
- transformers：模型加载和推理
- peft：参数高效微调（LoRA）
- bitsandbytes：量化支持
- trl：RLHF 训练
```

---

## 🎯 Week 7：RAG 与工具调用

### 学习任务
- [ ] 理解 RAG（检索增强生成）架构
- [ ] 学习向量数据库（Chroma、Milvus、Pinecone）
- [ ] 了解 Function Calling / Tool Use

### 动手任务
- [ ] 搭建一个简单的 RAG 系统
  - 文档切分 → Embedding → 向量存储 → 检索 → 生成
- [ ] 实现一个带工具调用的 Agent（如调用计算器、搜索）

### 项目代码结构
```
rag-system/
├── data/               # 原始文档
├── embedding.py        # 文本向量化
├── vector_store.py     # 向量数据库操作
├── retriever.py        # 检索逻辑
├── generator.py        # LLM 生成
└── main.py            # 交互入口
```

---

## 🎯 Week 8：项目实战

### 选择以下一个项目完成：

#### 选项 A：智能代码助手
- 基于公司代码库做 RAG
- 支持代码问答、生成、解释
- 集成到 IDE 或命令行

#### 选项 B：自动化报告生成
- 读取数据源（数据库、API、文件）
- 自动分析并生成业务报告
- 支持多轮修改和迭代

#### 选项 C：多 Agent 协作系统
- 多个 Agent 分工协作（研究员、写手、审核员）
- 使用 LangGraph 或类似框架
- 实现一个复杂任务的自动化流程

---

## 📚 扩展阅读（按优先级）

### 必读论文
1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer
2. [GPT-3](https://arxiv.org/abs/2005.14165) - Few-Shot Learning
3. [InstructGPT](https://arxiv.org/abs/2203.02155) - RLHF
4. [Chain-of-Thought](https://arxiv.org/abs/2201.11903) - 推理
5. [Llama 2](https://arxiv.org/abs/2307.09288) - 开源模型

### 选读论文
- [BERT](https://arxiv.org/abs/1810.04805) - 双向编码器
- [T5](https://arxiv.org/abs/1910.10683) - Text-to-Text 框架
- [GPT-4](https://arxiv.org/abs/2303.08774) - 多模态能力
- [RAG](https://arxiv.org/abs/2005.11401) - 检索增强
- [Toolformer](https://arxiv.org/abs/2302.04761) - 工具学习

---

## 🛠️ 推荐工具链

| 用途 | 工具 |
|------|------|
| 模型仓库 | Hugging Face |
| 微调框架 | PEFT、TRL、Axolotl |
| 向量数据库 | Chroma、Milvus、Weaviate |
| Agent 框架 | LangChain、LlamaIndex、AutoGPT |
| 部署推理 | vLLM、TGI、llama.cpp |
| 实验追踪 | Weights & Biases、TensorBoard |

---

## ✅ 学习检查清单

### 基础能力
- [ ] 能解释 Transformer 的完整 forward 流程
- [ ] 能计算模型的参数量和 FLOPs
- [ ] 能比较不同位置编码方案的优缺点

### 进阶能力
- [ ] 能实现简化版的 GPT 推理代码
- [ ] 能用 LoRA 微调一个 7B 模型
- [ ] 能搭建一个完整的 RAG 系统

### 工程能力
- [ ] 能优化模型的推理延迟
- [ ] 能部署模型到生产环境
- [ ] 能设计评估方案验证模型效果

---

*学习计划创建时间：2026-03-02*  
*建议学习时长：每周 8-10 小时*  
*适用对象：有编程基础，想深入理解 LLM 的工程师*
