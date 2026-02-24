# Qwen3-TTS Transformer Dynamo 导出深度优化指南

**记录日期**：2026-02-05
**核心目标**：攻克 Qwen3-TTS Decoder 中最复杂的 Transformer 骨干网络，通过消除“符号化死循环”和“参数树失配”，实现 `dynamo=True` 下的有状态导出。

---

## 1. 核心挑战：为什么它是“硬骨头”？

在导出 Part 2 (Transformer) 时，我们遇到了 Torch Dynamo 三大顶级报错：
1. **Memo Disaster**：符号化推理引擎在处理 Transformers 冗余切片时陷入死循环。
2. **Pytree Mismatch**：`torch.export` 对 `*args` 参数结构的极其严格的校验。
3. **SDPA Guard Error**：原生缩放点积注意力在符号化追踪时无法处理潜在的“零长度”分支。

---

## 2. 顶级故障排除与解决方案

### 2.1 解决符号化死循环 (Memo Disaster)
**报错现象**：`AssertionError: u8 possible memo disaster`
**根源分析**：`transformers` 源码中的 `eager_attention_forward` 包含了一行：
`causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]`
当 `key_states.shape[-2]` 是一个复杂的符号表达式（如 `past_len + N`）时，Dynamo 的内存化（Memoization）机制会因为递归生成的切片公式过于复杂而崩溃。
**解决方案 (Monkey Patch)**：
在导出前，手动劫持并替换该函数，移除动态切片，直接进行加法掩码。
```python
def optimized_eager_attention_forward(module, query, key, value, attention_mask, scaling, ...):
    # 移除 causal_mask = mask[:, :, :, :key_len] 的切片操作
    # 直接相加，前提是我们在包装器中已经生成了正确长度的 mask
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling + attention_mask
    ...
```

### 2.2 嵌套参数树匹配 (Pytree Structure)
**报错现象**：`UserError: Detected mismatch between inputs (2 elements) and dynamic_shapes (17 elements)`
**根源分析**：由于 `forward(hidden, *past_kv_flat)` 使用了可变参数，Dynamo 认为输入结构是 `(Tensor, Tuple(16 Tensors))`。如果 `dynamic_shapes` 传的是打平的列表，则会报结构不匹配。
**解决方案**：
显式构造嵌套元组作为 `dynamic_shapes`：
```python
dynamic_shapes = (
    {0: batch, 1: num_frames}, # hidden
    tuple([{0: batch, 2: past_seq}] * 16) # KV Cache 部分必须是一个嵌套元组
)
```

### 2.3 强制 Eager 模式避开 SDPA Guard
**报错现象**：`suggested fixes: torch._check(key.shape[2] != 0)`
**根源分析**：PyTorch 原生的 SDPA 实现在内核分发时会检查序列长度是否为 0，这在符号化追踪中会产生一个不可消除的算子分叉（Guard）。
**解决方案**：
强制模型回退到手动实现的 Eager 模式：
`model.config.decoder_config._attn_implementation = "eager"`

---

## 3. KV Cache 的符号化重构

为了让 16 个 KV 张量在滑动窗口（72 帧）中正确流转，我们放弃了复杂的 `Cache` 类，采用了 **TraceableKVStack**：
- **无状态化**：将 Cache 变为纯张量输入输出。
- **符号化拼接**：使用 `torch.cat` 拼接新帧。
- **定长负索引**：利用 `k_combined[:, :, -window_size:, :]` 获取最新窗口。Dynamo 对定长负切片的支持非常稳健。

---

## 4. 精度验证结果 (1.7B Base)

通过上述重构，导出的 Transformer 模型达到了极高的数值一致性：
- **New Hidden (Transformer 输出)**：Max Error ≈ **5e-8** (极佳)
- **Next KV Cache (状态迭代)**：Max Error ≈ **1e-6** (极佳)
- **文件大小**：仅 **0.75 MB** (算子图极度精简，无常量折叠开销)

---

## 5. 开发者建议清单

- [ ] **执行 Monkey Patch**：对依赖动态长度切片的 Transformers 内部函数进行局部替换。
- [ ] **严格匹配 Pytree**：检查 `dynamic_shapes` 的嵌套层次是否与 `forward` 参数完全一致。
- [ ] **优先 Eager 注意力**：在 Dynamo 导出语境下，手动 Eager 实现比黑盒 SDPA 更易追踪。
- [ ] **标量算子化**：将所有的 `past_len` 获取等逻辑保持在 Tensor 域（如 `size(2)`），不要转回 Python `int`。

---
*本文档由 Antigravity 记录，作为 Qwen3-TTS 深度定制化导出的进阶指南。*
