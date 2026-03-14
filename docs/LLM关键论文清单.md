# LLM 关键论文清单

> 按优先级排序，附核心贡献和阅读建议

---

## 🌟 第一优先级（必读）

### 1. Attention Is All You Need (2017)
- **作者**：Vaswani et al. (Google)
- **链接**：[arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- **核心贡献**：提出 Transformer 架构，Attention 机制取代 RNN
- **重点章节**：
  - Section 3: Model Architecture
  - Section 3.2: Multi-Head Attention
  - Section 3.4: Position-wise Feed-Forward Networks
- **阅读建议**：理解 Attention 的数学公式，能手绘架构图
- **代码实现**：[The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)

---

### 2. Language Models are Few-Shot Learners (GPT-3, 2020)
- **作者**：Brown et al. (OpenAI)
- **链接**：[arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
- **核心贡献**：
  - 175B 参数，证明模型规模可以带来质的飞跃
  - 提出 In-Context Learning（无需微调，仅通过 prompt 就能学习）
- **重点章节**：
  - Section 2: Approach（Zero-shot, One-shot, Few-shot 定义）
  - Section 3.9: Scaling Laws
  - Section 4: Results（各种任务的性能）
- **阅读建议**：关注"涌现能力"的概念和实验设计

---

### 3. Training Language Models to Follow Instructions (InstructGPT, 2022)
- **作者**：Ouyang et al. (OpenAI)
- **链接**：[arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- **核心贡献**：引入 RLHF，让模型输出更符合人类偏好
- **重点章节**：
  - Section 2: Methods（SFT + RM + RLHF 流程）
  - Section 2.2: Reward Modeling
  - Section 2.3: Reinforcement Learning
- **阅读建议**：理解对齐（Alignment）的重要性，SFT 和 RLHF 的作用

---

### 4. Chain-of-Thought Prompting Elicits Reasoning (2022)
- **作者**：Wei et al. (Google)
- **链接**：[arXiv:2201.11903](https://arxiv.org/abs/2201.11903)
- **核心贡献**：通过"一步步想"的 prompt，解锁 LLM 的推理能力
- **重点章节**：
  - Section 3: Experiments（不同任务上的效果）
  - Figure 1: Chain-of-Thought 示例
- **阅读建议**：这是 Prompt Engineering 的里程碑，实践比理论更重要

---

### 5. Llama 2: Open Foundation and Fine-Tuned Chat Models (2023)
- **作者**：Touvron et al. (Meta)
- **链接**：[arXiv:2307.09288](https://arxiv.org/abs/2307.09288)
- **核心贡献**：开源大模型的最佳实践，详细的训练细节
- **重点章节**：
  - Section 2: Pretraining（数据、训练设置）
  - Section 3: Fine-tuning（SFT + RLHF）
  - Section 4: Safety（红队测试、安全措施）
- **阅读建议**：学习开源模型的工程实践和安全考虑

---

## ⭐ 第二优先级（重要）

### 6. BERT: Pre-training of Deep Bidirectional Transformers (2018)
- **作者**：Devlin et al. (Google)
- **链接**：[arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
- **核心贡献**：双向编码器，开启预训练+微调范式
- **对比阅读**：与 GPT 的 Decoder-only 架构对比

---

### 7. Scaling Laws for Neural Language Models (2020)
- **作者**：Kaplan et al. (OpenAI)
- **链接**：[arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
- **核心贡献**：建立模型规模、数据量、算力与性能的定量关系
- **关键公式**：L(N) ∝ N^(-α)，性能与参数量的幂律关系

---

### 8. Retrieval-Augmented Generation for Knowledge-Intensive NLP (RAG, 2020)
- **作者**：Lewis et al. (Facebook/Meta)
- **链接**：[arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
- **核心贡献**：结合检索和生成，解决知识更新和幻觉问题

---

### 9. LoRA: Low-Rank Adaptation of Large Language Models (2021)
- **作者**：Hu et al. (Microsoft)
- **链接**：[arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- **核心贡献**：参数高效微调方法，大幅降低微调成本
- **阅读建议**：理解低秩近似的数学原理

---

### 10. RLHF 详解 (Anthropic, 2022)
- **作者**：Bai et al.
- **链接**：[Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
- **核心贡献**：Constitutional AI，用 AI 代替人类进行反馈

---

## 📖 第三优先级（按需阅读）

### 架构优化
| 论文 | 贡献 |
|------|------|
| [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) | SwiGLU 激活函数 |
| [RMSNorm](https://arxiv.org/abs/1910.07467) | 替代 LayerNorm，训练更稳定 |
| [RoFormer](https://arxiv.org/abs/2104.09864) | RoPE 旋转位置编码 |
| [Multi-Query Attention](https://arxiv.org/abs/1911.02150) | 加速推理的注意力变体 |

### 训练优化
| 论文 | 贡献 |
|------|------|
| [ZeRO](https://arxiv.org/abs/1910.02054) | 大规模模型训练内存优化 |
| [FlashAttention](https://arxiv.org/abs/2205.14135) | IO-aware 的 Attention 加速 |

### 推理与部署
| 论文 | 贡献 |
|------|------|
| [LLM.int8()](https://arxiv.org/abs/2208.07339) | 大模型量化 |
| [vLLM](https://arxiv.org/abs/2309.06180) | PagedAttention 高效推理 |

---

## 🎯 阅读策略

### 新手路线（每周 1 篇）
```
Week 1: Attention Is All You Need（理解基础架构）
Week 2: GPT-3（理解大模型的力量）
Week 3: Chain-of-Thought（理解推理能力）
Week 4: InstructGPT（理解对齐技术）
Week 5: Llama 2（理解开源实践）
```

### 高效阅读法
1. **先看摘要**：确定是否值得精读
2. **看图和表**：往往比文字更直观
3. **跳过附录**：第一轮先抓主线
4. **做笔记**：用自己的话总结核心贡献

### 代码复现
- [Hugging Face Papers](https://huggingface.co/papers)：每日更新的论文+讨论
- [Papers With Code](https://paperswithcode.com/)：论文+官方/社区代码实现

---

## 📚 相关资源

### 综述文章
- [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223) - 最全面的 LLM 综述
- [Harnessing the Power of LLMs in Practice](https://arxiv.org/abs/2304.13712) - 实践指南

### 博客解读
- [Lil'Log - Transformer Architecture](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/)
- [Jay Alammar - The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Andrej Karpathy - State of GPT](https://www.youtube.com/watch?v=bZQun8Y4L2A)

---

*清单创建时间：2026-03-02*  
*建议：根据学习目标选择阅读，不必全部读完*
