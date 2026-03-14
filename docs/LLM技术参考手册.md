# LLM 技术参考手册

> 基于论文、官方文档和工程实践的技术细节

---

## 一、Transformer 架构技术规格

### 1.1 原版 Transformer (Vaswani et al., 2017)

**论文**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

**架构参数 (Base 版本)**:
```
参数项              数值
─────────────────────────────────────
Encoder 层数        6
Decoder 层数        6
模型维度 d_model    512
FFN 中间维度        2048 (4 × d_model)
注意力头数          8
每头维度            64 (512 ÷ 8)
Dropout             0.1
位置编码            Sinusoidal
```

**计算复杂度分析** (论文 Section 3.2):
```
Layer Type      Complexity    Sequential Ops    Maximum Path Length
─────────────────────────────────────────────────────────────────────
Self-Attention  O(n² · d)     O(1)              O(1)
Recurrent       O(n · d²)     O(n)              O(n)
Convolutional   O(k · n · d²) O(1)              O(logₖ(n))
```
- n: 序列长度
- d: 模型维度
- k: 卷积核大小

**关键结论**: Self-Attention 在路径长度上优于 RNN，但序列长度的平方复杂度是瓶颈。

### 1.2 位置编码公式 (Sinusoidal)

```python
# 论文公式 (Section 3.5)
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

# pos: 位置
# i: 维度索引
# d_model: 模型维度
```

**设计原理**:
- 波长从 2π 到 10000·2π 几何级数增长
- 允许模型学习相对位置: PE(pos+k) 可表示为 PE(pos) 的线性函数

---

## 二、GPT 系列模型规格

### 2.1 GPT-3 (Brown et al., 2020)

**论文**: [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)

**模型规格**:
```
模型版本        参数量      层数    维度    头数    上下文长度    训练数据量
────────────────────────────────────────────────────────────────────────────
GPT-3 Small    125M       12      768     12      2048         ~300B tokens
GPT-3 Medium   350M       24      1024    16      2048         ~300B tokens
GPT-3 Large    760M       24      1536    16      2048         ~300B tokens
GPT-3 XL       1.3B       24      2048    24      2048         ~300B tokens
GPT-3 2.7B     2.7B       32      2560    32      2048         ~300B tokens
GPT-3 6.7B     6.7B       32      4096    32      2048         ~300B tokens
GPT-3 13B      13B        40      5140    40      2048         ~300B tokens
GPT-3 175B     175B       96      12288   96      2048         ~300B tokens (实际 ~499B)
```

**训练成本估算** (论文和后续分析):
- 175B 模型训练: ~3.14 × 10²³ FLOPs
- V100 GPU 估算: ~355 年 GPU 时间
- 实际使用数千 V100，训练数周
- 估计成本: $4.6M - $12M (Lambda Labs 估算)

**Few-shot 性能关键发现** (论文 Table 3.3):
- 随着模型规模增大，Few-shot 性能持续提升
- 175B 模型在很多任务上超越 Fine-tuned SOTA

### 2.2 GPT-4 (OpenAI, 2023)

**技术报告**: [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774)

**公开规格** (OpenAI 官方):
```
参数项              规格
────────────────────────────────────────
上下文长度          8K / 32K tokens
知识截止日期        2023年4月 (部分版本)
多模态              支持图像输入

注意: OpenAI 未公开 GPT-4 具体参数量
```

**性能基准** (技术报告):
- MMLU (多任务语言理解): 86.4% (人类 89.8%)
- HellaSwag (常识推理): 95.3%
- 模拟律师考试: 前 10% (对比 GPT-3.5 后 10%)

---

## 三、开源模型技术规格

### 3.1 LLaMA / LLaMA 2 (Meta)

**论文**: 
- LLaMA: [Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- LLaMA 2: [Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)

**LLaMA 规格**:
```
模型        参数量      维度    层数    头数    训练数据        上下文
────────────────────────────────────────────────────────────────────────
LLaMA-7B    6.7B       4096    32      32      1.0T tokens     2048
LLaMA-13B   13.0B      5120    40      40      1.0T tokens     2048
LLaMA-33B   32.5B      6656    60      52      1.4T tokens     2048
LLaMA-65B   65.2B      8192    80      64      1.4T tokens     2048
```

**LLaMA 2 规格**:
```
模型            参数量      维度    层数    上下文    预训练数据
────────────────────────────────────────────────────────────────────
LLaMA-2-7B      6.7B       4096    32      4096      2.0T tokens
LLaMA-2-13B     13.0B      5120    40      4096      2.0T tokens
LLaMA-2-70B     70B        8192    80      4096      2.0T tokens
```

**架构改进** (LLaMA 2 论文 Section 2):
- 使用 RoPE (Rotary Position Embedding) 替代绝对位置编码
- 使用 SwiGLU 激活函数替代 ReLU
- 使用 RMSNorm 替代 LayerNorm (Pre-normalization)
- 70B 模型使用 Grouped-Query Attention (GQA) 加速推理

### 3.2 国产模型规格

**Qwen (通义千问)**:
```
模型                参数量      上下文      特点
─────────────────────────────────────────────────────────────────
Qwen-7B             7.7B       32K         开源，可商用
Qwen-14B            14.2B      32K         开源，可商用
Qwen-72B            72.7B      32K         开源，可商用
Qwen-1.5-110B       110B       32K         MoE 架构

参考: https://github.com/QwenLM/Qwen
```

**DeepSeek**:
```
模型                参数量      上下文      特点
─────────────────────────────────────────────────────────────────
DeepSeek-7B         7B         4K          开源
DeepSeek-67B        67B        4K          开源
DeepSeek-V2         236B(MoE)  128K        MLA 注意力机制

参考: https://github.com/deepseek-ai/DeepSeek-V2
```

---

## 四、训练与推理计算量

### 4.1 训练 FLOPs 估算

**公式** (Kaplan et al., 2020):
```
训练 FLOPs ≈ 6 × N × D

N: 模型参数量
D: 训练数据 tokens 数量
6: 前向(2) + 后向(4) 的系数

示例: GPT-3 175B
= 6 × 175B × 300B
= 3.15 × 10²³ FLOPs
```

### 4.2 推理 FLOPs 估算

**公式**:
```
推理 FLOPs ≈ 2 × N × tokens_generated

示例: 生成 1000 tokens，使用 7B 模型
= 2 × 7B × 1000
= 14 × 10¹² FLOPs
= 14 TFLOPs
```

### 4.3 GPU 算力参考

```
GPU 型号          FP16 算力       显存        适用场景
─────────────────────────────────────────────────────────────
RTX 4090         82.6 TFLOPS     24GB        本地开发，7B-13B
A100 40GB        312 TFLOPS      40GB        生产部署，70B(量化)
A100 80GB        312 TFLOPS      80GB        生产部署，70B(全精度)
H100 80GB        989 TFLOPS      80GB        大规模训练/推理

数据来源: NVIDIA 官方规格
```

---

## 五、Embedding 模型规格

### 5.1 OpenAI Embedding

**官方文档**: https://platform.openai.com/docs/guides/embeddings

```
模型                    维度      最大输入      价格 (每 1M tokens)
─────────────────────────────────────────────────────────────────
text-embedding-3-small  1536      8191          $0.02
text-embedding-3-large  3072      8191          $0.13
text-embedding-ada-002  1536      8191          $0.10 (legacy)
```

**MTEB 排行榜性能** (截至 2024):
- text-embedding-3-large: 64.6% (MTEB Average)
- text-embedding-3-small: 62.3% (MTEB Average)

参考: https://huggingface.co/spaces/mteb/leaderboard

### 5.2 开源 Embedding 模型

**BGE (BAAI)**:
```
模型                    维度      最大长度    MTEB 排名
─────────────────────────────────────────────────────
BGE-large-zh            1024      512         中文第一梯队
BGE-large-en-v1.5       1024      512         Top 5
BGE-m3                  1024      8192        多语言，长文本

参考: https://github.com/FlagOpen/FlagEmbedding
```

**M3E (moka-ai)**:
```
模型                    维度      特点
─────────────────────────────────────────
m3e-base                768       轻量级
m3e-large               1024      性能更好

参考: https://github.com/wangshunshun/m3e
```

---

## 六、向量数据库性能对比

### 6.1 基准测试数据

**ANN-Benchmarks** (近似最近邻搜索): https://ann-benchmarks.com/

测试数据集: SIFT (128维, 100万向量)

```
系统            召回率@10    查询时间(ms)    内存占用
─────────────────────────────────────────────────────────
FAISS (IVF)     0.95         0.5             ~2GB
Milvus          0.95         0.8             ~3GB
Weaviate        0.92         1.2             ~4GB
Chroma          0.88         2.5             ~2GB
Pinecone        0.94         1.0             N/A (托管)
```

### 6.2 选型建议 (基于官方文档)

| 场景 | 推荐 | 理由 |
|------|------|------|
| 快速原型 | Chroma | pip install 即可，无需额外依赖 |
| 生产级 | Milvus/Zilliz | 分布式，十亿级向量支持 |
| 无运维 | Pinecone | 全托管，自动扩缩容 |
| 复杂查询 | Weaviate | GraphQL 接口，混合搜索 |

---

## 七、量化技术规格

### 7.1 量化方法对比

**GGUF (llama.cpp)**:
```
格式            位数      精度损失    适用模型
─────────────────────────────────────────────────
Q4_0            4-bit     ~中         通用
Q4_K_M          4-bit     ~低         推荐
Q5_K_M          5-bit     ~很低       高质量需求
Q8_0            8-bit     ~极低       几乎无损

参考: https://github.com/ggerganov/llama.cpp/blob/master/docs/quantization.md
```

**AWQ/GPTQ**:
```
方法            特点                        性能损失
─────────────────────────────────────────────────────────
GPTQ            逐层量化，适合 GPU 推理      ~1-2% 
AWQ             保护重要权重，精度更好       ~<1%

参考论文:
- GPTQ: https://arxiv.org/abs/2210.17323
- AWQ: https://arxiv.org/abs/2306.00978
```

### 7.2 显存需求公式

```
显存(GB) ≈ (参数数量 × 精度位数) / (8 × 10⁹)

示例 1: 7B 模型，FP16
= (7 × 10⁹ × 16) / (8 × 10⁹)
= 14 GB

示例 2: 7B 模型，INT4
= (7 × 10⁹ × 4) / (8 × 10⁹)
= 3.5 GB (实际约 4-5GB 含开销)
```

---

## 八、长上下文技术

### 8.1 上下文长度演进

```
模型/技术            上下文长度      关键技术
─────────────────────────────────────────────────────────
GPT-3                2K             原始 Transformer
GPT-4                8K/32K         未知
Claude 2             100K           未知
Claude 3             200K           未知
GPT-4 Turbo          128K           未知
Gemini 1.5 Pro       1M/2M          未知
Llama 2              4K             RoPE 外推
Llama 3              8K             预训练扩展
LongRoPE (论文)      2M             位置编码插值

参考:
- LongRoPE: https://arxiv.org/abs/2402.13753
```

### 8.2 位置编码扩展方法

**RoPE 外推技术**:
```
方法                    原理                        效果
────────────────────────────────────────────────────────────
NTK-aware              调整频率基                  支持 8K+
YaRN                   温度缩放 + 注意力缩放        支持 128K+
Dynamic NTK            动态调整                    推理时自适应

参考:
- NTK-aware: https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/
- YaRN: https://arxiv.org/abs/2309.00071
```

---

## 九、评估基准

### 9.1 主流基准测试

**MMLU (Massive Multitask Language Understanding)**:
```
描述: 涵盖 57 个学科的多项选择题
数据来源: 专业考试题
指标: 准确率
SOTA (2024): ~90% (GPT-4 级)
人类水平: ~89.8%

参考: https://arxiv.org/abs/2009.03300
```

**HellaSwag (常识推理)**:
```
描述: 句子补全，测试常识推理
指标: 准确率
GPT-4: 95.3%
人类: ~95%

参考: https://arxiv.org/abs/1905.07830
```

**HumanEval (代码生成)**:
```
描述: 164 个编程问题，函数补全
指标: Pass@k (k 次尝试通过率)
GPT-4: 67% (Pass@1)

参考: https://arxiv.org/abs/2107.03374
```

### 9.2 中文评估基准

**C-Eval**:
```
描述: 涵盖 52 个学科的中文多项选择题
包含: 初中、高中、大学、职业考试
参考: https://arxiv.org/abs/2305.08322
```

**CMMLU**:
```
描述: 中国文化背景的多任务理解
特点: 更多中国文化相关题目
参考: https://arxiv.org/abs/2306.09212
```

---

## 十、参考资源索引

### 10.1 必读论文

| 论文 | 引用 | 链接 |
|------|------|------|
| Attention Is All You Need | Transformer 开山 | [arXiv](https://arxiv.org/abs/1706.03762) |
| GPT-3 | Few-shot learning | [arXiv](https://arxiv.org/abs/2005.14165) |
| InstructGPT | RLHF | [arXiv](https://arxiv.org/abs/2203.02155) |
| LLaMA | 开源模型 | [arXiv](https://arxiv.org/abs/2302.13971) |
| LLaMA 2 | 开源 Chat | [arXiv](https://arxiv.org/abs/2307.09288) |
| LoRA | 高效微调 | [arXiv](https://arxiv.org/abs/2106.09685) |
| RAG | 检索增强 | [arXiv](https://arxiv.org/abs/2005.11401) |

### 10.2 官方文档

| 资源 | 链接 |
|------|------|
| OpenAI API | https://platform.openai.com/docs |
| Hugging Face | https://huggingface.co/docs |
| LangChain | https://python.langchain.com/docs |
| LlamaIndex | https://docs.llamaindex.ai |
| vLLM | https://docs.vllm.ai |

### 10.3 排行榜

| 排行榜 | 链接 |
|--------|------|
| MTEB (Embedding) | https://huggingface.co/spaces/mteb/leaderboard |
| LMSYS Chatbot Arena | https://chat.lmsys.org |
| Open LLM Leaderboard | https://huggingface.co/spaces/open-llm-leaderboard |

---

*文档版本: v1.0*  
*最后更新: 2024*  
*所有参数均来自官方论文或文档*
