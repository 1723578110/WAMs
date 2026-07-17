# MaskWAM 技术调研报告

## 1. 200 字以内核心结论

MaskWAM 是一个对象中心的 World-Action Model (WAM)，论文标题为 **MaskWAM: Unifying Mask Prompting and Prediction for World-Action Models**，arXiv v1 于 2026-06-11 发布，PDF 日期为 2026-06-12。其核心是把目标 mask 同时作为首帧视觉 prompt 和未来预测目标，与 RGB future 和 action chunk 在 Wan 2.2/MoT/flow-matching 框架内联合建模。论文报告 LIBERO 98.4%、RoboTwin 2.0 92.2%、真实机器人语言歧义任务 84.9%。但官方仓库当前多为 skeleton：训练、推理、评测、模型 forward/loss 均未实现，权重和数据未开放。因此工程复现评级为 **低**：可以按论文重实现，暂不能直接复现论文指标。

## 2. 研究对象确认

| 项 | 结论 | 证据 |
|---|---|---|
| 准确标题 | MaskWAM: Unifying Mask Prompting and Prediction for World-Action Models | arXiv API `2606.13515v1`；论文首页 PDF p.1 |
| 作者 | Hanyang Yu, Haitao Lin, Jingbo Zhang, Wenyao Zhang, Chenghao Gu, Heng Li, Ping Tan | arXiv API；论文首页 PDF p.1；README lines 9-18 |
| 机构 | HKUST；Tencent Robotics X；Tsinghua University | 论文首页 PDF p.1；README lines 17-18 |
| 版本/发布日期 | arXiv v1, published/updated 2026-06-11 16:02:42 UTC；PDF 日期 2026-06-12 | arXiv API；论文 PDF p.1 |
| 官方论文 | https://arxiv.org/abs/2606.13515 / https://arxiv.org/pdf/2606.13515v1 | arXiv API |
| 项目主页 | https://hanyangyu1021.github.io/maskwam.github.io/ | 论文 PDF p.1；README lines 5-7 |
| 官方代码仓库 | https://github.com/hanyangyu1021/maskwam | README line 25；GitHub API `full_name=hanyangyu1021/maskwam` |
| 模型权重 | 未开放；README 仅列 TODO “Release model checkpoints” | README lines 98-115；GitHub releases 为空 |
| 数据集地址 | 未开放；仓库仅有 `data/libero`、`data/robotwin` 占位路径 | `configs/train/libero.yaml` lines 22-25；`configs/train/robotwin.yaml` lines 22-24 |
| 开源许可证 | MIT License，仅覆盖当前仓库软件和文档 | `LICENSE` lines 1-20；`pyproject.toml` license=MIT |
| 实际开放范围 | README、安装说明、配置占位、模型/训练/评测 skeleton；关键训练/推理/评测逻辑均 `NotImplementedError` | README lines 94-119；`scripts/train.py` lines 20-23；`maskwam/models/wam.py` lines 87-103 |
| 同名项目 | 未发现另一个同名官方 MaskWAM；搜索结果回到 arXiv、项目页、该 GitHub | Web/GitHub/HF 搜索；HF model/dataset search 返回空 |

## 3. 核心方法与架构

### 范式与问题

MaskWAM 明确属于 **World-Action Model (WAM)**，不是纯 VLA。论文把 WAM 定义为通过未来视频预测辅助动作生成；MaskWAM 在此基础上加入对象 mask：一方面预测 future masks，作为对象中心语义监督；另一方面可在首帧输入 target mask，解决语言歧义和相似物体目标指代问题。论文明确说明输入为当前 RGB 观测 `I0`、proprioceptive state `s0`、语言指令 `l`、可选首帧 mask `M0`；输出为 action chunk `a1:K`、未来 RGB `I1:T`、未来 mask `M1:T`（论文 Sec.3 / PDF p.3）。

### ASCII 架构图

```text
Inputs
  RGB I0 / history       optional first-frame mask M0       language l       proprio s0
        |                         |                           |                |
        |                         |                           |                |
        +---- render masks as RGB-compatible frames ----------+                |
        |                                                                         |
        v                                                                         v
  Frozen causal 3D Video VAE E                                             action/noisy action
        |                                                                         |
        |-- RGB latents zv: C x L x H' x W'                                      |
        |-- mask latents zm: C x L x H' x W'                                     |
        v                                                                         |
  concat channel-wise z=[zv;zm]: 2C x L x H' x W'                                 |
        |                                                                         |
  expanded patch embedding: C -> 2C                                               |
        |                                                                         |
        +-------------------- Unified DiT / MoT --------------------+-------------+
                             visual branch                          action expert
                             frozen T5 text via cross-attn          joint attention
                                   |                                      |
                      predict RGB + mask latent velocities          predict action velocities
                                   |                                      |
                         L_video + L_mask + L_act                   action chunk
```

### 模型架构表

| 模块 | 论文明确说明 | 代码确认 | 根据实现推断/注意 |
|---|---|---|---|
| Backbone | Built on Wan 2.2；统一 DiT/MoT；visual branch + lightweight action expert；language 用 frozen T5 cross-attention；Video VAE frozen | public skeleton 不是 Wan 2.2 实现，而是 `nn.TransformerEncoder` 占位 | 论文实际模型参数量、Wan 2.2 具体规模未披露 |
| 输入模态 | RGB、proprioception、语言、可选首帧 target mask | `forward(rgb, language, mask_prompt)` 缺少 state 参数，且未实现 | 代码 skeleton 与论文完整输入不一致 |
| 输出模态 | action chunk、future RGB、future masks | `forward` docstring 返回 `actions/future_rgb/future_mask`，但 raise `NotImplementedError` | 无可运行 decoder |
| mask 编码 | mask 渲染为三通道 RGB-compatible image；RGB 和 mask 都经同一个 frozen causal 3D Video VAE 编码 | skeleton 有单独 `MaskEncoder(vit_small_patch16)` | 代码 skeleton 不符合论文“同一 VAE 编码 rendered masks”的关键设计 |
| latent 融合 | RGB latent `zv` 与 mask latent `zm` 沿 channel concat 为 `2C` | skeleton tokenizer 拼 RGB/mask/language token | 论文是 latent channel concat，不是公开代码中的 ViT token encoder |
| patch embedding | 输入 patch embedding 从 `C` 扩展到 `2C`；原 RGB channel 继承 pretrained 权重，新增 mask channel zero init；输出 head 从 `C` 扩到 `2C` | 未实现 | zero init 只针对 mask channel |
| 注意力/交互 | block-wise causal attention mask；视觉 branch 与 action expert 通过 joint attention 交互 | skeleton `TransformerEncoderLayer(d_model=768,nhead=12,dropout=0.1)` | 真实 block-wise causal mask 未开放 |
| 训练目标 | `L = Lvideo + Lmask + Lact`；joint flow matching；visual timestep `tau_v`，action timestep `tau_a` 独立/解耦 | skeleton loss weights `{rgb:1, mask:1, action:1}` | 损失公式细节、velocity parameterization、timestep 分布未披露 |
| mask dropout | 训练中 `M0` 以 `p=0.5` 替换为 zero tensor | config `use_mask_prompt: true` | 代码未实现 dropout |
| 推理 | 首帧 SAM3 生成 `M0`；之后不实时 tracking；one denoising step 得到部分去噪 RGB-mask latent；action expert 生成 chunk；使用 KV-cache | 未实现 `predict_action` | 控制频率、延迟、显存未披露 |
| 论文实验图像尺寸 | `384 x 320` RGB frames，prediction horizon `T=8` | train configs 写 `image_size:224`, `pred_horizon:4` | 以论文实验配置为准；代码配置疑似占位 |
| action dim/chunk | 论文公式 `K x D`，未给全局数值 | LIBERO config: `action_dim=7`, `chunk=8`; RoboTwin config: `action_dim=14`, `chunk=16` | 这些来自未实现配置，不能确认论文实验真实值 |
| 参数量/层数/hidden/head/FFN | 未披露 | skeleton: hidden 768、12 layers、12 heads；FFN size 未显式设置，PyTorch 默认 2048 若实例化 | 这不是论文 Wan 2.2 实际参数 |

## 4. 训练数据表

| 数据/评测来源 | 任务/embodiment | 规模 | 模态/相机/分辨率/频率 | action/state 维度 | 处理与标注 | 公开性 |
|---|---|---|---|---|---|---|
| LIBERO | `libero_spatial`, `libero_object`, `libero_goal`, `libero_long` | 论文未披露训练 episodes；表 1 报四套 benchmark 成功率 | 论文实验帧为 `384 x 320`, `T=8`；控制频率未披露 | 未披露；代码占位 config 写 `action_dim=7`, chunk=8 | mask 预测用于训练；是否使用 mask prompt 评测 LIBERO 未明确；论文说可无 test-time visual prompt | LIBERO 本身公开；MaskWAM 处理后数据/脚本未开放 |
| RoboTwin 2.0 | Hammer, Bell, Card, Burger, Stand, Shoe 六个随机任务 | 每任务 500 randomized episodes for training | 论文未披露相机数、帧率、控制频率；代码 eval config 占位列了另 4 个 tasks，和论文表 2 不一致 | 未披露；代码占位写 bimanual `action_dim=14`, chunk=16 | 随机 instructions/environments/object placements；future mask prediction | RoboTwin 本身公开；MaskWAM 数据处理未开放 |
| 真实机器人语言清晰任务 | Dual-arm Xtrainer；Task1 stack bowls, Task2 hang mug, Task3 drawer+pen, Task4 fold towel | 平均 100 demos/task；100 eval trials/task | RealSense D455 head Eye-on-Base + D405 Eye-on-Hand；分辨率/帧率/控制频率未披露 | 未披露 | Qwen3-VL 解析 instruction；SAM3 分割/跟踪；人工校验 | 私有，未开放 |
| 真实机器人语言歧义任务 | Task5 grasp bowl, Task6 pick cup, Task7 pick/place bottle, Task8 grasp cosmetics | 平均 100 demos/task；60 trials/task/setting | 同上；16 个相似 cosmetics 的精细目标选择用于 Task8 | 未披露 | human point prompt 首帧初始化 SAM3；SAM3 跨 episode tracking；低质量 episode 人工 keyframe 修正 | 私有，未开放 |
| mask annotation pipeline | 语言清晰/歧义任务 | 91% episode 无需人工修正；标注 50 episodes 约 3 min；歧义任务首帧 point prompt 额外 5-10 s/episode | SAM3 online/offline；Qwen3-VL | 不适用 | 首轮 human verification；低质量重提示并重新传播 | 工具公开情况依赖 Qwen3-VL/SAM3；标注数据未开放 |

## 5. 训练 Recipe 和超参数表

| 项 | 论文实际实验配置 | 代码确认 | 复现判断 |
|---|---|---|---|
| 初始化 checkpoint | Wan 2.2；pretrained Video VAE 与 text encoder | 未提供 checkpoint URL | 关键缺口 |
| 冻结模块 | Video VAE 和 T5 text encoder frozen | 未实现训练 | 可按论文实现 |
| 可训练模块 | 扩展 patch embedding、DiT/MoT、action expert、输出 head 等 | skeleton 有 RGBEncoder/MaskEncoder/tokenizer/backbone/action_head 但未实现 | 真实模块未开放 |
| loss | `L = Lvideo + Lmask + Lact` | config loss weights 均为 1.0 | 具体 flow matching loss 形式未披露 |
| noise/timestep | visual RGB+mask 共享 `tau_v`；action 独立 `tau_a` | 未实现 | timestep 分布未披露 |
| mask prompt dropout | `p=0.5` zero-mask dropout | 未实现 | 论文明确 |
| 图像尺寸/horizon | `384 x 320`, prediction horizon `T=8` | configs 写 `224`, `pred_horizon=4` | 冲突，复现应优先论文 |
| optimizer/betas/epsilon | 未披露 | 未实现 | 未披露 |
| LR/scheduler/warmup | 未披露 | config: LR `1e-4`, warmup 2000 | 占位配置，不可视为论文实验 |
| batch size | 未披露 | LIBERO 64；RoboTwin 32 | 占位配置 |
| steps/epochs | 未披露 | LIBERO 100k；RoboTwin 150k | 占位配置 |
| weight decay | 未披露 | 0.05 | 占位配置 |
| precision | 未披露 | bf16 | 占位配置 |
| grad clipping | 未披露 | 1.0 | 占位配置 |
| EMA/dropout | EMA 未披露；模型 dropout 未披露 | skeleton dropout 0.1 | 占位配置 |
| GPU/TPU/训练时间 | 未披露 | 未披露 | 复现成本不可精确估算 |
| 并行策略 | 未披露 | 未披露 | FSDP/DeepSpeed/ZeRO 均未披露 |

## 6. Benchmark 性能表

### LIBERO，成功率 %

| Method | Type | Spatial | Object | Goal | Long | Avg | 证据 |
|---|---:|---:|---:|---:|---:|---:|---|
| WorldVLA | VLA | 87.6 | 96.2 | 83.4 | 60.0 | 81.8 | Table 1, PDF p.6 |
| GR00T-N1 | VLA | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 | Table 1, PDF p.6 |
| pi0 | VLA | 96.8 | 98.8 | 95.8 | 85.2 | 94.1 | Table 1, PDF p.6 |
| pi0.5 | VLA | 98.6 | 98.2 | 98.0 | 92.4 | 96.8 | Table 1, PDF p.6 |
| Motus | WAM | 96.8 | 99.8 | 96.6 | 97.6 | 97.7 | Table 1, PDF p.6 |
| FastWAM | WAM | 98.2 | 100.0 | 97.0 | 95.2 | 97.6 | Table 1, PDF p.6 |
| Ours RGB-only | WAM | 96.8 | 99.6 | 97.0 | 95.8 | 97.3 | Table 1, PDF p.6 |
| Ours Mask-only | WAM | 97.2 | 99.8 | 97.4 | 96.0 | 97.6 | Table 1, PDF p.6 |
| Ours MaskWAM | WAM | 98.8 | 100.0 | 98.2 | 96.4 | 98.4 | Table 1, PDF p.6 |

### RoboTwin 2.0，成功率 %

| Method | Hammer | Bell | Card | Burger | Stand | Shoe | Avg | 证据 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| pi0 | 68 | 72 | 81 | 79 | 63 | 74 | 72.8 | Table 2, PDF p.6 |
| FastWAM | 83 | 87 | 92 | 94 | 80 | 90 | 87.7 | Table 2, PDF p.6 |
| Ours RGB-only | 82 | 87 | 91 | 93 | 79 | 92 | 87.3 | Table 2, PDF p.6 |
| Ours Mask-only | 85 | 90 | 93 | 93 | 81 | 91 | 88.8 | Table 2, PDF p.6 |
| Ours MaskWAM | 88 | 93 | 95 | 97 | 85 | 95 | 92.2 | Table 2, PDF p.6 |

### 真实机器人语言清晰任务，成功率 %

| Method | Type | Task1 | Task2 | Task3 | Task4 | Avg | 证据 |
|---|---|---:|---:|---:|---:|---:|---|
| pi0 | VLA | 57 | 54 | 54 | 58 | 55.8 | Table 3, PDF p.6 |
| pi0.5 | VLA | 83 | 55 | 74 | 77 | 72.3 | Table 3, PDF p.6 |
| FastWAM | WAM | 88 | 76 | 77 | 75 | 79.0 | Table 3, PDF p.6 |
| Ours RGB-only | WAM | 86 | 77 | 76 | 78 | 79.3 | Table 3, PDF p.6 |
| Ours MaskWAM | WAM | 91 | 82 | 81 | 83 | 84.3 | Table 3, PDF p.6 |

### 真实机器人语言歧义与 OOD，成功率 %

| Setting | Method | Type | Task5 | Task6 | Task7 | Task8 | Avg | 证据 |
|---|---|---|---:|---:|---:|---:|---:|---|
| In Domain | pi0-mask | VLA | 96.7 | 93.3 | 38.3 | 23.3 | 62.9 | Table 5, PDF p.17 |
| In Domain | pi0-coord | VLA | 53.3 | 88.3 | 20.0 | 5.0 | 41.7 | Table 5 |
| In Domain | FastWAM-coord | WAM | 35.0 | 55.0 | 13.3 | 1.7 | 26.3 | Table 5 |
| In Domain | Ours | WAM | 100.0 | 100.0 | 88.3 | 83.3 | 92.9 | Table 5 |
| Distractors | pi0-mask | VLA | 86.7 | 75.0 | 31.7 | 18.3 | 52.9 | Table 5 |
| Distractors | pi0-coord | VLA | 46.7 | 61.7 | 11.7 | 0.0 | 30.0 | Table 5 |
| Distractors | FastWAM-coord | WAM | 15.0 | 23.3 | 5.0 | 0.0 | 10.8 | Table 5 |
| Distractors | Ours | WAM | 100.0 | 98.3 | 85.0 | 78.3 | 90.4 | Table 5 |
| Novel Instances | pi0-mask | VLA | 71.7 | 65.0 | 26.7 | 15.0 | 44.6 | Table 5 |
| Novel Instances | pi0-coord | VLA | 41.7 | 51.7 | 8.3 | 0.0 | 25.4 | Table 5 |
| Novel Instances | FastWAM-coord | WAM | 18.3 | 28.3 | 6.7 | 0.0 | 13.3 | Table 5 |
| Novel Instances | Ours | WAM | 86.7 | 76.7 | 70.0 | 65.0 | 74.6 | Table 5 |
| Lighting | pi0-mask | VLA | 75.0 | 68.3 | 28.3 | 13.3 | 46.3 | Table 5 |
| Lighting | pi0-coord | VLA | 48.3 | 66.7 | 15.0 | 3.3 | 33.3 | Table 5 |
| Lighting | FastWAM-coord | WAM | 31.7 | 36.7 | 8.3 | 0.0 | 19.2 | Table 5 |
| Lighting | Ours | WAM | 93.3 | 91.7 | 73.3 | 68.3 | 81.7 | Table 5 |

### 公平性判断

论文内自家 ablation（RGB-only、Mask-only、Ours-no-pred、Ours-coord）最公平，因为架构和训练域更接近。与 pi0/pi0.5/GR00T-N1/WorldVLA/Motus/FastWAM 的比较存在潜在数据、预训练规模、输入形式、训练协议差异；论文未逐项披露 baseline 是否使用额外预训练、额外数据或完全相同输入，因此这些比较应视为 benchmark-level 对比，不应视为严格 controlled comparison。

## 7. 消融与局限

### 消融

| 设计问题 | 结果 | 证据 | 解释 |
|---|---:|---|---|
| RGB-only vs joint RGB+mask prediction | LIBERO Avg 97.3 -> 98.4 | Table 1, PDF p.6 | future mask 是语义正则，减少背景/干扰物 shortcut |
| Mask-only vs joint | LIBERO Avg 97.6 -> 98.4；RoboTwin 88.8 -> 92.2 | Table 1/2 | mask 捕获空间目标，RGB 保留纹理/动力学；联合最好 |
| 无 future mask prediction 但有 mask prompt | Avg Succ. 21.6 | Table 4, PDF p.8 | prompt alone 不足，必须用预测目标把 mask 接入主表征 |
| 坐标文本 prompt | Ours-coord Avg Succ. 18.2 | Table 4 | 坐标文本与视觉空间对齐差 |
| 首帧 mask 噪声 | mild erosion/dilation/shift/dropout 仅有限下降；严重错位/dropout 明显下降 | Fig.8, PDF p.15 | mask 主要作为目标身份和空间锚点 |
| 直接下采样 mask / 独立 3D CNN mask encoder | 定性报告 worse than rendered mask + pretrained RGB VAE | Appendix C, PDF p.17 | 单独 mask 表征与 RGB latent mismatch |
| ControlNet-style mask injection | 定性报告 did not work well | Appendix C | side branch 未和 future prediction 绑定，模型会 underuse mask |
| VLA zero-init gated mask fusion | 模型倾向保持 mask pathway weak/ignored | Appendix C | zero init 稳定但不保证学习新 mask 信号 |

### 局限

论文明确局限包括：依赖 mask supervision 和 deployment 时 segmentation-derived prompts；复杂真实场景下自动 mask extraction 仍不简单；受计算约束，未做大规模 RGB-mask-action pretraining（Sec.6 / PDF p.8）。工程上更大的限制是官方可运行代码、权重、数据处理和私有真实机器人数据均未开放。

## 8. 最小复现步骤

当前不能“直接跑官方仓库复现”。最小复现只能按论文重实现：

1. 获取基础环境：PyTorch、Transformers、diffusers、timm、Hydra、LIBERO/RoboTwin 2.0；官方 `requirements.txt` 给出了依赖范围。
2. 选择 Wan 2.2 video model checkpoint；冻结 causal 3D Video VAE 和 T5 text encoder。
3. 实现 mask renderer：把 binary/instance mask 渲染为 RGB-compatible image，背景单独颜色。
4. 用同一个 frozen Video VAE 编码 RGB frames 和 rendered mask frames，得到 `zv`、`zm`。
5. 扩展 DiT patch embedding `C -> 2C`：原 RGB 权重拷贝，新增 mask channel zero init；输出 head 扩为 `2C`。
6. 实现 MoT：visual branch 去噪 RGB+mask latent；action expert 接收 proprioception 和 noisy action，通过 joint attention 使用 visual context。
7. 实现 block-wise causal attention mask，以及 joint flow matching：RGB/mask 共用 `tau_v`，action 独立 `tau_a`，loss 为 `Lvideo + Lmask + Lact`。
8. 训练时以 `p=0.5` 对首帧 mask prompt 做 zero dropout。
9. 数据准备：LIBERO/RoboTwin 生成 RGB、state/action、语言、future mask；真实机器人按 Qwen3-VL + SAM3 + human verification 标注。
10. 推理：首帧用 SAM3 生成 target mask；只做 one-step partial denoising；action expert 生成 action chunk；实现 KV-cache。
11. 评测：LIBERO 四套 suite；RoboTwin 表 2 六任务；真实机器人按论文 100/60 trials 协议。

### 复现资源估算

论文未披露 GPU 类型、数量、训练时长、参数量和显存，因此无法给出可信精确估算。保守判断：若重实现基于 Wan 2.2/MoT 并处理 384x320、T=8 的视频 latent，单机消费级 GPU 很可能不足；至少需要多张高显存 GPU 做 fine-tuning。真实机器人复现还需要 Dual-arm Xtrainer 或等效双臂平台、RealSense D455/D405、多任务演示采集和人工校验流程。

整体可复现性评级：**低**。理由：方法思想清楚，表格完整；但核心可运行代码、权重、数据处理、真实数据、训练细节和硬件配置缺失。

## 9. 未披露信息清单

- 真实 MaskWAM 参数量。
- Wan 2.2 backbone 具体尺寸、层数、hidden size、attention heads、FFN size。
- MoT action expert 的层数、宽度、attention 细节。
- action chunk 的论文实验实际 `K`、每个 benchmark 的 action/state 维度。
- optimizer、betas、epsilon、scheduler 细节。
- batch size、gradient accumulation、epochs/steps、总样本/token 数。
- dropout、EMA、gradient clipping、precision 的论文实验值。
- flow matching velocity target 公式、timestep distribution/noise schedule。
- GPU/TPU 类型、数量、训练时间、并行策略。
- 推理控制频率、端到端延迟、FPS、显存、硬件。
- LIBERO 训练样本数、评测 episode 数、随机种子。
- RoboTwin 输入相机、分辨率、帧率、控制频率；论文表 2 六任务与代码 eval config 四任务不一致的原因。
- 真实机器人 action/state 维度、控制频率、相机分辨率/帧率。
- baseline 的完整训练数据、预训练、输入对齐和评测协议差异。
- 模型权重、训练数据、数据准备工具、可运行训练/推理/评测脚本。

## 10. 一手资料来源

- arXiv API: https://export.arxiv.org/api/query?id_list=2606.13515
- arXiv paper: https://arxiv.org/abs/2606.13515
- Project page: https://hanyangyu1021.github.io/maskwam.github.io/
- Official GitHub: https://github.com/hanyangyu1021/maskwam
- Local inspected commit: `7939ba8e99a2dac0a55a86e85bcc181391c4a7a2`, committed 2026-06-12 10:47:14 +08:00
- Key inspected files: `README.md`, `LICENSE`, `configs/train/libero.yaml`, `configs/train/robotwin.yaml`, `configs/eval/*.yaml`, `maskwam/models/wam.py`, `scripts/train.py`, `maskwam/engine/trainer.py`
