# Week 01 — 环境 + PyTorch + Transformer 入门

> 多模态大模型 13 周计划 · W1（2026.10.26–11.01）
> 本周关键词：环境可复现 / 训练循环 / QKV / 读论文

## 1. 本周目标

- [ ] 跑通 Miniconda + CUDA + PyTorch，`torch.cuda.is_available() == True`
- [ ] 用 PyTorch 写完整训练循环，MLP 在 MNIST 上 test acc ≥ 97%
- [ ] 手写 scaled dot-product attention + multi-head（含 mask）
- [ ] 读 Attention 论文 §1–5，产出 QKV 笔记
- [ ] 建公开 repo + 本周学习日志

## 2. 环境

| 项 | 值 |
|---|---|
| OS | `<Windows 11 / Ubuntu 22.04>` |
| Python | `<3.11.14>` |
| CUDA | `<12.9>` |
| PyTorch | `<2.x.x>` |
| GPU | `<RTX 4090 8G / CPU>` |

验证命令与输出：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# <2.x.x True>
```

复现步骤：

```bash
conda create -n mm python=3.11 -y
conda activate mm
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install jupyter matplotlib
```

## 3. 目录结构

```
week01/
├── README.md
├── environment.md          # 环境安装与踩坑记录
├── mnist_mlp.py            # 训练循环 + MNIST
├── attention.py            # 手写 attention + 单测
├── notebooks/
│   └── transformer_intro.ipynb
├── notes/
│   ├── qkv.md              # QKV / 多头 / 位置编码
│   └── attention_paper.md  # 三遍法 6 栏笔记
└── results/
    ├── mnist_curve.png
    └── mnist_acc.txt
```

## 4. 任务与验收

| 任务 | 状态 | 证据 | 验收标准 |
|---|---|---|---|
| 环境搭建 | ☐ | `environment.md` | 别人按文档能复现 |
| PyTorch 训练循环 | ☐ | `mnist_mlp.py` | 能解释 forward/backward/optimizer |
| MNIST ≥97% | ☐ | `results/mnist_acc.txt` | 实测 test acc 写进 README |
| 手写 attention | ☐ | `attention.py` + 单测 | 形状/数值正确，mask 生效 |
| QKV 笔记 | ☐ | `notes/qkv.md` | 能 5 分钟讲清 attention 在算什么 |
| 读论文 | ☐ | `notes/attention_paper.md` | 6 栏笔记完整 |
| 公开 repo | ☐ | repo 链接 | week01/ 已提交 |

## 5. 运行方式

```bash
python mnist_mlp.py --epochs 5 --batch-size 64
python -m pytest attention.py -q
jupyter lab
```

预期输出：

```
Epoch 5 | train_loss <0.05> | test_acc <0.97+>
attention tests passed
```

## 6. 结果

- MNIST test accuracy：`<0.97x>`
- 训练曲线：`results/mnist_curve.png`
- attention 单测：`<全部通过>`

## 7. QKV 一句话总结

- **Q/K/V**：Query 是“我在找什么”，Key 是“我有什么”，Value 是“我实际提供什么”。
- **公式**：`Attention(Q,K,V) = softmax(QKᵀ / √d_k) V`
- **为什么要 scale**：防止点积过大导致 softmax 饱和、梯度消失。
- **Multi-head**：让不同头学不同关系（语法/位置/共指等），再拼接。
- **位置编码**：attention 本身无顺序，必须注入位置信息。
- **一句话**：attention 是在算“每个 token 应该从其他 token 取多少信息”。

## 8. 论文笔记（三遍法 6 栏）

| 栏 | 内容 |
|---|---|
| 问题 | 如何不用 RNN/CNN，直接建模长距离依赖 |
| 假设 | 纯 attention + 位置编码足够 |
| 方法 | scaled dot-product / multi-head / residual + LayerNorm |
| 数据与指标 | WMT 翻译 BLEU |
| 结论 | 可并行、长依赖更好、可扩展 |
| 局限与可改点 | `<自己写 1–2 条>` |

## 9. 资料使用记录

- 默认资料：PyTorch Docs、The Annotated Transformer、Attention 论文
- 视频：`<只看了一次 3B1B/李宏毅 attention 建直觉，或“未看”>`
- 没看懂、下周要补：`<...>`

## 10. 本周复盘

- 完成：
- 未完成 / 原因：
- 下周只改一件事：
- 时间投入：技术 `<__h>` / 数学 `<__h>` / 英语 `<__h>`

## 11. 链接

- Repo：`<https://github.com/...>`
- 论文：https://arxiv.org/abs/1706.03762
- 参考实现：http://nlp.seas.harvard.edu/annotated-transformer/
```
