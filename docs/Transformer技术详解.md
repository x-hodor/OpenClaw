# Transformer 技术详解

> 深入理解大语言模型的核心架构

---

## 一、为什么需要 Transformer？

### RNN/LSTM 的问题
在 Transformer 出现之前，序列建模主要依赖 RNN/LSTM：

```
问题 1：长距离依赖困难
输入："The cat, which was sitting on the mat, ... , ___"
                                                    ↑
                    需要联系开头的 "cat"，但距离太远，梯度消失

问题 2：无法并行计算
每一步都依赖前一步的隐藏状态，只能串行处理
时间复杂度：O(n) 的序列长度
```

### Transformer 的突破
- **Self-Attention**：任意两个位置直接交互，距离不再是问题
- **并行计算**：Attention 可矩阵化，GPU 友好

---

## 二、Self-Attention 详解

### 2.1 核心思想
> 每个词都能"看到"其他所有词，并根据相关性加权聚合信息。

### 2.2 计算过程

```python
import torch
import torch.nn as nn
import math

def self_attention(Q, K, V, mask=None):
    """
    Q, K, V: [batch_size, seq_len, d_model]
    """
    d_k = Q.size(-1)
    
    # Step 1: 计算相似度（点积）
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # [batch, seq_len, seq_len]
    
    # Step 2: 应用 Mask（因果注意力）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Step 3: Softmax 得到注意力权重
    attn_weights = torch.softmax(scores, dim=-1)
    
    # Step 4: 加权求和
    output = torch.matmul(attn_weights, V)
    # [batch, seq_len, d_model]
    
    return output, attn_weights
```

### 2.3 直观理解
```
输入序列：[我  喜欢  深度  学习]

注意力矩阵（每个词关注谁）：
       我    喜欢   深度   学习
我    0.4   0.3   0.2   0.1    ← "我" 主要关注自己
喜欢  0.3   0.4   0.2   0.1    ← "喜欢" 关注主语和宾语
深度  0.1   0.2   0.4   0.3    ← "深度" 与"学习"强相关
学习  0.1   0.2   0.3   0.4    ← "学习" 关注修饰词"深度"
```

### 2.4 为什么要除以 √d_k？
```
当 d_k 较大时，点积结果的方差会变大，导致 softmax 梯度消失。
除以 √d_k 可以将方差控制在合理范围。
```

---

## 三、Multi-Head Attention

### 3.1 动机
单一的注意力机制可能只捕捉到一种相关性，多 head 可以让模型从多个"角度"理解关系：
- Head 1：句法关系（主谓宾）
- Head 2：语义关系（同义词）
- Head 3：指代关系（代词-名词）

### 3.2 实现
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性投影
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # 1. 线性投影并分头
        # [batch, seq, d_model] -> [batch, num_heads, seq, d_k]
        Q = self.W_Q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. 计算注意力（每个 head 独立）
        attn_output, attn_weights = self_attention(Q, K, V, mask)
        
        # 3. 拼接多头结果
        # [batch, num_heads, seq, d_k] -> [batch, seq, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # 4. 最终线性投影
        return self.W_O(attn_output)
```

---

## 四、位置编码（Position Encoding）

### 4.1 为什么需要？
> Attention 本身是位置无关的，需要显式注入位置信息。

### 4.2 Sinusoidal 位置编码（原版 Transformer）
```python
import numpy as np

def get_sinusoidal_position_encoding(seq_len, d_model):
    """
    使用正弦/余弦函数生成位置编码
    """
    position = np.arange(seq_len)[:, np.newaxis]  # [seq_len, 1]
    div_term = np.exp(np.arange(0, d_model, 2) * 
                      -(np.log(10000.0) / d_model))
    
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)  # 偶数维
    pe[:, 1::2] = np.cos(position * div_term)  # 奇数维
    
    return pe

# 特性：PE(pos + k) 可以通过 PE(pos) 线性表示
# 这让模型能学习到相对位置关系
```

### 4.3 可学习位置编码（GPT/BERT）
```python
# 更简单，直接作为可训练参数
position_embedding = nn.Embedding(max_seq_len, d_model)
positions = torch.arange(seq_len)
pe = position_embedding(positions)
```

### 4.4 RoPE（旋转位置编码）
现代 LLM（Llama、Qwen）使用的方案：
```python
# 将位置信息编码到 Q、K 的旋转中
# 优点：外推性好，支持更长序列
```

---

## 五、完整 Transformer Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # 1. 多头自注意力
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # 2. 前馈网络（FFN）
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # 现代 LLM 常用 GELU 替代 ReLU
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 注意力子层（Pre-Norm 结构，现代 LLM 常用）
        attn_out = self.attention(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + self.dropout1(attn_out)  # 残差连接
        
        # FFN 子层
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout2(ffn_out)  # 残差连接
        
        return x
```

### 关键设计
| 组件 | 作用 |
|------|------|
| **残差连接** | 缓解梯度消失，支持深层网络 |
| **Layer Norm** | 稳定训练，放在注意力之前（Pre-Norm）效果更好 |
| **FFN** | 增加非线性表达能力，d_ff 通常是 d_model 的 4 倍 |
| **Dropout** | 防止过拟合 |

---

## 六、因果注意力（Causal/Masked Attention）

### 6.1 什么是因果注意力？
> 解码器（Decoder）只能看到当前位置及之前的位置，不能"偷看"未来。

### 6.2 Mask 实现
```python
def create_causal_mask(seq_len):
    """
    生成下三角矩阵
    [[1, 0, 0, 0],
     [1, 1, 0, 0],
     [1, 1, 1, 0],
     [1, 1, 1, 1]]
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask  # [seq_len, seq_len]

# 应用时，将 mask=0 的位置填充为 -inf
scores = scores.masked_fill(mask == 0, float('-inf'))
```

### 6.3 与编码器注意力的区别
- **Encoder**：双向注意力（可以看到全部上下文）
- **Decoder**：因果注意力（只能看前面）
- **GPT（Decoder-only）**：全部使用因果注意力

---

## 七、参数量计算

### 7.1 单层的参数
```
Multi-Head Attention:
  - W_Q, W_K, W_V: 3 × d_model × d_model
  - W_O: d_model × d_model
  - 总计: 4 × d_model²

FFN:
  - W1: d_model × d_ff
  - W2: d_ff × d_model
  - 通常 d_ff = 4 × d_model
  - 总计: 8 × d_model²

LayerNorm: 2 × d_model（可忽略）

单层总计: ~12 × d_model²
```

### 7.2 GPT-3 175B 的估算
```
配置: d_model=12288, num_layers=96, vocab_size=50257

Embedding: 2 × vocab_size × d_model ≈ 1.2B
Transformer Blocks: 96 × 12 × 12288² ≈ 174B
总计: ~175B 参数
```

---

## 八、常见变体

| 模型 | 架构 | 特点 |
|------|------|------|
| **Transformer** | Encoder + Decoder | 原版，用于机器翻译 |
| **BERT** | Encoder-only | 双向编码，用于理解任务 |
| **GPT** | Decoder-only | 因果生成，用于文本生成 |
| **T5** | Encoder-Decoder | Text-to-Text 统一框架 |
| **Llama** | Decoder-only | RoPE、SwiGLU、RMSNorm |

### 为什么现代 LLM 都用 Decoder-only？
1. **Scaling 效率**：Decoder-only 架构在相同参数下计算效率更高
2. **生成能力**：自回归生成更适合开放式任务
3. **统一框架**：理解和生成可以用同一个模型完成

---

## 九、面试常考问题

1. **Self-Attention 的时间复杂度是多少？**
   - 序列长度的平方 O(n²)，是 Transformer 的主要瓶颈

2. **为什么需要 Multi-Head？**
   - 单 head 表达能力有限，多 head 能捕捉不同子空间的信息

3. **LayerNorm vs BatchNorm？**
   - LayerNorm 对序列长度不敏感，适合变长序列

4. **Transformer 和 RNN 的区别？**
   - 并行性、长距离依赖、计算复杂度

5. **如何理解 Attention 的"Q、K、V"？**
   - Query：我要查询什么
   - Key：我有什么信息
   - Value：信息的具体内容

---

*文档创建时间：2026-03-02*
