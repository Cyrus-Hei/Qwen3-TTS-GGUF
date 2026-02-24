# Qwen3-TTS Decoder Dynamo 导出技术指南 (Part 1 & 3)

**记录日期**：2026-02-04
**核心目标**：将 Qwen3-TTS Decoder 拆分导出，并从旧版 JIT Trace (`dynamo=False`) 迁移至 Torch 2.x 现代导出路径 (`dynamo=True`)，解决流式推理中的数据依赖型形状（Data-Dependent Shapes）问题。

---

## 1. 背景：为什么转向 Dynamo？

在旧版导出中，为了维持流式状态，我们使用了大量的 `torch.zeros(...) + size` 技巧来“欺骗”编译器，但在 DirectML (DML) 或高级推理后端下，这会导致：
1. **常量折叠失效**：产生冗余计算。
2. **DML 报错**：动态形状引发的 `8007023E` 等 Reshape 错误。
3. **导出限制**：无法利用 Torch 2.x 的符号化推理（Symbolic Reasoning）能力。

---

## 2. 技术挑战：数据依赖型形状 (Unbacked Symbols)

在流式推断中，我们需要根据 `is_last` 信号动态决定：
- **Part 1**：跳过多少历史帧进行预卷积。
- **Part 3**：下发多少帧音频，截取多少帧作为下一次的历史。

直接使用 Python 的 `if` 或 Tensor 切片（如 `wav[:, :valid_len]`）会触发 Dynamo 的 `PendingUnbackedSymbolNotFound` 报错，因为编译器无法在导出期确定这个长度的来源。

---

## 3. 核心解决方案：符号化解耦架构

遵循 **《参考经验 06 - Paddable 架构》**，我们对模型逻辑进行了颠覆性重构：

### 3.1 废除“物理切断”，拥抱“逻辑描述”
**原则**：模型内部严禁执行 `tensor[:valid]` 这种丢弃数据的操作。
- **做法**：模型始终返回“全量波形”（物理容器），同时额外输出 `start_samples`（起始偏移）和 `valid_samples`（有效长度）。
- **优势**：导出的张量形状仅取决于输入 `N` 的线性映射（如 `N * 1920`），形状表达式是静态可预测的。

### 3.2 符号化状态更新 (Gather vs Slicing)
为了获取 `next_conv_hist`（最后 4 帧缓存），不再使用动态切片，而是使用 `torch.gather`。
```python
# 生成固定长度的索引序列 [0, 1, 2, 3]
indices = torch.arange(4, device=device)
# 逻辑末尾计算符号化：(num_finalize_idx - 4) + [0, 1, 2, 3]
gather_indices = (num_finalize_idx - 4) + indices
gather_indices = torch.clamp(gather_indices, min=0).view(1, 1, -1).expand(B, Hidden, -1)
# 执行符号化拉取
next_conv_hist = torch.gather(accumulated, 2, gather_indices)
```

### 3.3 无分支控制流 (Eliminating Branching)
将 Python 的 `if is_last:` 逻辑全部下沉为 Tensor 算子：
```python
# 使用 torch.where 代替 if
num_finalize = torch.where(
    is_last.view(-1) > 0.5, 
    total_acc_t, 
    torch.clamp(total_acc_t - lookahead_t, min=0)
).view(-1)
```

---

## 4. 拆分导出实践

### Part 1: Pre-Conv & RVQ
- **职责**：离散码 -> 隐空间特征。
- **优化点**：使用 `torch.narrow` 取代物理索引，确保 `hidden` 的输出维度被 Dynamo 标记为符号轴 `num_frames`。

### Part 3: Upsample & CNN Decoder
- **职责**：隐空间 -> 24kHz 波形。
- **优化点**：解决了最难的 Lookahead (前瞻) 对齐问题。通过返回 `start_samples` 标量，实现了流式推理中完美的帧级对齐，且无需破坏计算图的完整性。

---

## 5. 导出验证成果 (1.7B Base 模型)

| 指标 | 验证结果 | 结论 |
|:---|:---|:---|
| **导出路径** | `torch.onnx.export(..., dynamo=True)` | **PASS** |
| **文件大小** | Part 1 (0.1MB), Part 3 (0.44MB) | 极度精简 |
| **数值误差 (Max)** | 4.2e-4 (波形) | 在 FP32 正常范围内 |
| **符号一致性** | 状态缓存误差 0.00000 | **完美对齐** |

---

## 6. 开发者 Checklist

- [ ] **禁止 `tensor[start:end]`**：如果 start 或 end 是变量，改用 `torch.narrow` 或 `torch.gather`。
- [ ] **禁止 `item()`**：严禁将 Tensor 转为 Python 标量参与形状计算。
- [ ] **优先 `torch.where`**：逻辑判断必须算子化。
- [ ] **返回容器 + 长度**：不要在模型内部“剪裁”最终输出。

---

*本文档为 Qwen3-TTS 项目 Dynamo 导出路径的标准化参考。*
*记录人：Antigravity*
*日期：2026-02-04*
