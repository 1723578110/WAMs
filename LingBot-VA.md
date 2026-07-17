# LingBot-VA 工程复现调研报告

> 资料范围：仅使用 arXiv 论文/PDF/源码、官方项目页、官方 GitHub README/代码/配置、官方 Hugging Face/ModelScope 链接入口。关键数字均标注来源；未能从一手资料确认的字段写“未披露”。

## 1. 200 字以内核心结论

LingBot-VA 是面向机器人控制的自回归 Video-Action World Model：用 Wan2.2 视频生成先验、MoT/因果注意力、KV cache 和异步执行，把未来视频预测与动作反解合成一个流匹配策略。论文完整预训练不可完全复现：16K 小时混合语料含内部数据且算力未披露；但开源 shared-backbone 权重、RoboTwin/LIBERO 后训练代码和部分数据，使 post-training/评测具备中等可复现性。

## 2. 研究对象确认

| 项目 | 结论 | 来源与证据 |
|---|---|---|
| 准确标题 | **Causal World Modeling for Robot Control** | 论文 `main.tex:51`；arXiv API `2601.21998v2` |
| 模型名 | **LingBot-VA** | 论文源码 `main.tex:16` 定义 `\methodname{LingBot-VA}`；GitHub README 标题 `README.md:1` |
| 作者 | Lin Li, Qihang Zhang, Yiming Luo, Shuai Yang, Ruilin Wang, Fei Han, Mingrui Yu, Zelin Gao, Nan Xue, Xing Zhu, Yujun Shen, Yinghao Xu | 论文源码 `main.tex:53-66`；arXiv API author list |
| 机构 | 论文作者栏未列逐作者 affiliation；官方源码模板含 Ant Group/Robbyant 标识 | 论文源码 `main.tex:53-66` 只列作者/贡献；`robbyant.cls:23-25` 含 `antgroup.png`/`robbyant.png` |
| 版本/日期 | arXiv v2；首次发布 2026-01-29 17:07:43Z，更新 2026-03-22 15:37:40Z | arXiv API `published`/`updated` |
| 官方论文 | [arXiv:2601.21998](https://arxiv.org/abs/2601.21998)，仓库内 `LingBot_VA_paper.pdf` | arXiv API comment；GitHub 文件 |
| 项目主页 | [technology.robbyant.com/lingbot-va](https://technology.robbyant.com/lingbot-va) | arXiv API comment；README badges `README.md:4-7` |
| 代码仓库 | [github.com/robbyant/lingbot-va](https://github.com/robbyant/lingbot-va) | arXiv API comment；README `README.md:4-7` |
| 模型权重 | `robbyant/lingbot-va-base`、`robbyant/lingbot-va-posttrain-robotwin`、`robbyant/lingbot-va-posttrain-libero-long`，均提供 Hugging Face 与 ModelScope 链接 | `README.md:69-73` |
| 数据集地址 | `robbyant/robotwin-clean-and-aug-lerobot`、`robbyant/libero-long-lerobot`，均提供 Hugging Face 与 ModelScope 链接 | `README.md:77-80` |
| 许可证 | Apache License 2.0 | `LICENSE.txt:1`；`README.md:553-555` |
| 实际开放范围 | 开源推理/后训练代码、shared-backbone 权重、RoboTwin/LIBERO-Long 后训练数据；完整 16K 小时预训练混合语料和预训练 recipe 细节未完全开放 | README 模型/数据表 `README.md:69-80`；论文数据说明 p.10 / `sections/3_experiments.tex:8-23` |
| 同名/易混淆对象 | 官方仓库同时包含 `LingBot_VA2_paper.pdf`，为后续 LingBot-VA 2.0；本报告只调研 `LingBot_VA_paper.pdf`/arXiv 2601.21998 | 官方仓库文件列表；README 对一代模型表述 |

## 3. 模型架构表

**范式判断：**论文明确把它称为 video-action world model / causal world modeling；它不是传统“图像+语言直接回归动作”的 VLA，而是以自回归世界模型预测未来视觉状态，再通过 inverse dynamics 产生动作。按机器人术语可归为 **WAM / World Model + Action Model**，同时具备 VLA 输入接口。

| 模块 | 论文明确说明 | 代码确认 | 复现注意 |
|---|---|---|---|
| Backbone | Wan2.2-5B 视频流，`d_v=3072`，30 层 Transformer；动作流同深度、`d_a=768`，约 +350M 参数，总 5.3B（论文 p.10；`sections/3_experiments.tex:37-38`） | 开源 `WanTransformer3DModel` 默认 30 层、24 heads、head dim 128、inner dim 3072、FFN 14336、text dim 4096、action dim 30（`wan_va/modules/model.py:600-608,621,627,637-649`） | README 模型写明 `w/ shared backbone`（`README.md:71-73`）；论文双流 MoT 与开源 shared-backbone 实现存在差异，工程复现应以开源配置为准 |
| 视觉编码 | Wan2.2 causal VAE，压缩比 `4 x 16 x 16`，再 spatial patchify by 2；多视角沿宽度拼接，`N=192` spatial tokens/frame（论文 p.10；`sections/3_experiments.tex:40`） | `patch_size=[1,2,2]`，latent channel 48，patch embedding 到 3072（`wan_va/modules/model.py:599-626`） | 原始 Wan2.2 VAE 权重路径需用户配置，未随仓库直接包含 |
| 语言编码 | 冻结 T5 text encoder，通过 cross-attention 注入（论文 p.10；`sections/3_experiments.tex:42`） | `text_dim=4096`，`condition_embedder.text_embedder`（`wan_va/modules/model.py:605,628-635,682`） | T5 具体 checkpoint 名称未在论文/配置中明确 |
| 动作表示 | 统一双臂 30D：每臂 7D EEF + 7D joints + 1D gripper，总 `(7+7+1)*2=30`（论文 p.10；`sections/3_experiments.tex:12-18`） | RoboTwin 使用通道 `0-6,28,7-13,29` 共 16 个有效/对齐通道；LIBERO 使用 `0-6` 共 7 维（`va_robotwin_cfg.py:33-40`；`va_libero_cfg.py:33-38`） | 无效维 padding/掩码；动作按分位数归一化 |
| 动作编码/解码 | action encoder/decoder 是 hidden dim 256 的单层 MLP（论文 p.10；`sections/3_experiments.tex:41`） | 开源 shared-backbone 实现中 `action_embedder = Linear(30,3072)`，`action_proj_out = Linear(3072,30)`（`wan_va/modules/model.py:627,649`） | 论文版 `d_a=768` action stream 未在当前公开代码中一比一呈现 |
| 模态交互 | MoT：视频/动作各自 self-attention、FFN、LayerNorm，经跨模态 attention 交互；共享文本 cross-attention（论文 p.6,p.10；`sections/2_method.tex:181-199`） | 开源模型把 latent/action/noisy/condition token 拼成统一序列并用 mask/position id 控制（`wan_va/modules/model.py:712-798`） | 训练/推理 attention mode 不同 |
| 自回归与 chunk | 视频下采样因子 `tau=4`，预测 `K` 视频帧对应 `tau*K` actions；训练随机 `K in [1,4]`，实验部署 `K=4`（论文 p.6,p.10；`sections/2_method.tex:187-189`；`sections/3_experiments.tex:43`） | 训练代码随机 `chunk_size=torch.randint(1,5)`，`window_size=torch.randint(4,65)`（`wan_va/train.py:238-245`） | README 自定义数据建议视频 5-15 fps（`README.md:290`） |
| 生成方式 | 条件 flow matching；推理 Euler：视频 3 steps 到 `s=0.6`，动作 10 steps 到 `s=1.0`；视频 CFG=5.0，动作 CFG=1.0（论文 p.10；`sections/3_experiments.tex:45-47`） | RoboTwin 配置视频 25 steps、动作 50 steps；LIBERO 视频 20 steps、动作 50 steps；CFG 5/1（`va_robotwin_cfg.py:24-31`；`va_libero_cfg.py:24-31`） | 论文通用推理步数与 released benchmark 配置不一致，复现 benchmark 应优先用配置 |
| KV cache/推理 | Algorithm 1 使用 KV cache 维护自回归历史；异步执行边执行当前动作边预测未来（论文 p.8,p.14；`sections/2_method.tex` Algorithm 1；`sections/3_experiments.tex:220-224`） | `create_empty_cache()` 按 `attn_window/2` 和 token/chunk 建 KV cache；server 初始化 latent/action token cache（`wan_va/modules/model.py:661-669`；`wan_va/wan_va_server.py:399-410`） | RoboTwin `attn_window=72`，LIBERO `attn_window=30` |
| 精度/并行 | bf16 mixed precision，FSDP，grad clip 2.0（论文 p.10；`sections/3_experiments.tex:53-54`） | FSDP fully_shard blocks/attn/ffn，param bf16 reduce fp32；activation checkpointing（`distributed/fsdp.py:8-31`） | GPU 类型/数量/训练时间未披露；脚本默认 `NGPU=8`（`script/run_va_posttrain.sh`） |
| 显存/延迟 | 异步比同步快 2x（论文 p.14；`sections/3_experiments.tex:220-224`） | README：RoboTwin 单卡 eval offload 约 24GB VRAM；i2av offload 约 18GB VRAM（`README.md:210,233`） | 精确 latency/FPS 未披露 |

**ASCII 架构图**

```text
Language instruction ── frozen T5 ───────────────┐
                                                 v
Multi-camera RGB history ─ Wan2.2 causal VAE ─ video latent z_t ─ patchify ┐
                                                                           │
Action/state history ─ 30D canonical action ─ quantile norm ─ action tokens│
                                                                           v
                Causal interleaved video-action sequence
        [z_t, a_t,1 ... a_t,tau, z_t+1, ...] + RoPE + causal/FlexAttn mask
                                                                           │
        ┌──────────────────────────────────────────────────────────────────┐
        │ LingBot-VA Transformer                                           │
        │ paper: dual-stream MoT, video d_v=3072, action d_a=768            │
        │ released code: shared 3072-dim backbone, 30 layers, 24 heads      │
        └──────────────────────────────────────────────────────────────────┘
             │                                             │
             v                                             v
  future video latent flow target              future action flow target
             │                                             │
       Wan VAE decode                              denorm + channel select
             │                                             │
 predicted future observations            executable action chunk
             └──────────── KV cache + async rollout + real observation refresh
```

## 4. 训练数据表

| 阶段/数据 | 来源、任务、embodiment | 规模 | 模态/相机/频率 | action/state 维度 | 清洗/增强/混合 | 公开性 |
|---|---|---:|---|---|---|---|
| 预训练混合语料 | Agibot：mobile manipulator 多任务；RoboMind：multi-embodiment demonstrations；InternData-A1：大规模仿真 sim-to-real；OXE：OpenVLA subset；UMI Data：UMI 人类示教，排除 DexUMI；RoboCOIN：cross-embodiment bimanual data（论文 p.10；`sections/3_experiments.tex:8-23`） | 约 16K 小时（论文 p.10；`sections/3_experiments.tex:22-23`） | 未逐数据集披露相机数、分辨率、帧率、控制频率 | 统一 30D 双臂动作（论文 p.10） | 每数据集 90% train / 10% validation；统一格式和标注质量；各来源 uniform sampling（论文 p.10；`sections/3_experiments.tex:8-10,57`） | 部分公开，且包含 internally collected demonstrations（论文 p.10）；完整混合清单/manifest 未公开 |
| RoboTwin 2.0 后训练/评测 | bimanual manipulation，50 tasks；Easy 固定初始配置，Hard 随机物体位姿/布局（论文 Table 1 p.13） | 2,500 clean demos（50/task）+25,000 randomized demos（500/task）（论文 p.13；`sections/3_experiments.tex:111-113`） | 3 相机：`cam_high`,`cam_left_wrist`,`cam_right_wrist`；配置 256x320；视频 50Hz 下采样到 12.5Hz，action 50Hz（论文 p.13；`va_robotwin_cfg.py:15-21`） | 配置 `action_dim=30`，`action_per_frame=16`；有效通道见配置（`va_robotwin_cfg.py:17-40`） | 官方发布 cleaned & augmented LeRobot 数据；动作 quantile norm（`README.md:79,268-290`；`va_robotwin_cfg.py:42-58`） | 数据集公开：`robbyant/robotwin-clean-and-aug-lerobot`（`README.md:79`） |
| LIBERO 后训练/评测 | LIBERO-Spatial/Object/Goal/Long；每 suite 10 tasks（论文 p.13-14；Table 2） | 每 suite 50 demos/task，共 500 demos/suite；评测每 suite 3 seeds x 500 trials = 1500（论文 p.13；`sections/3_experiments.tex:117-121`） | 2 相机：`agentview_rgb`,`eye_in_hand_rgb`；配置 128x128；频率未披露（`va_libero_cfg.py:15-21`） | 配置 `action_dim=30`，`action_per_frame=4`，有效 `0-6` 7D（`va_libero_cfg.py:18-38`） | 按 OpenVLA 过滤 unsuccessful demonstrations；动作 quantile norm（论文 p.13；`va_libero_cfg.py:40-59`） | README 公开 `libero-long-lerobot`；四 suite 完整训练数据公开范围未完全确认（`README.md:80`） |
| 真实机器人任务 | 6 个任务：Make Breakfast、Pick Screws、Insert Tubes、Unpack Delivery、Fold Clothes、Fold Pants；长程/精密/柔性物体（论文 Fig.5/Fig.6 p.10-12；Appendix A p.23） | 每任务 50 real-world demos 用于训练；每任务 20 trials 评测（论文 p.12,p.23；`sections/3_experiments.tex:92-93`；Appendix A） | 物理机器人平台；相机数量、分辨率、控制频率未披露 | 未披露；训练序列长度 150,000（论文 p.12） | 未披露清洗/增强；交替评测 π0.5 与本方法（Appendix A p.23） | 数据未公开 |

## 5. 训练 Recipe 和超参数表

| 阶段 | 初始化 checkpoint | 训练/冻结模块 | Loss/采样 | Optimizer 与 LR | Batch/steps/precision | 并行/硬件 | 来源与备注 |
|---|---|---|---|---|---|---|---|
| 预训练 | 视频流初始化 Wan2.2-5B；动作网络由视频权重插值并用 `alpha=sqrt(d_v/d_a)` 缩放（论文 p.6；`sections/2_method.tex:201-204`） | 论文说明训练视频/动作 world-action 模型；冻结 T5；其他冻结细节未披露 | conditional flow matching；inverse dynamics loss 权重 `lambda=1`；text dropout 0.1；uniform SNR sampler；视频/动作均使用 uniform SNR（论文 p.10；`sections/3_experiments.tex:52-60`） | AdamW，peak LR `1e-4`，weight decay `0.01`，cosine annealing + linear warmup（论文 p.10） | 1.4T tokens；bf16；grad clip 2.0；batch size/grad accumulation/dropout 细节未披露 | FSDP；GPU/TPU 类型、数量、训练时间未披露 | 论文明确说明；完整预训练脚本未开源 |
| 通用 post-training | `lingbot-va-base` 或用户指定 checkpoint | 代码 `transformer.requires_grad_(True)`，训练 transformer；VAE/text encoder 用于 latent/text 预处理，不在训练脚本中训练（`wan_va/train.py:74-104`） | released code：latent/action MSE flow target，按 scheduler training weight 加权；总 loss=`latent_loss+action_loss`；latent noisy history prob 0.5，action 0；scheduler 1000 train timesteps（`wan_va/train.py:138-141,168-217,260-329`） | AdamW fused，betas 来自配置，eps `1e-8`，weight decay 配置；released code scheduler 是 warmup-constant lambda（`wan_va/train.py:107-118`） | bf16 参数保存；grad clip 2.0；DataLoader DistributedSampler seed 42（`wan_va/train.py:123-146,319-387`） | FSDP fully_shard + activation checkpointing（`distributed/fsdp.py:8-31`）；脚本默认 `NGPU=8`（`script/run_va_posttrain.sh`） | 代码确认；与论文“cosine”仅对应预训练，不对应开源后训练代码 |
| RoboTwin post-training | 论文/README 使用 LingBot-VA；baseline WAN 对照用 Wan2.2-5B 初始化 | 训练 transformer | 配置 `snr_shift=5.0`，`action_snr_shift=1.0`；CFG prob 0.1（`va_robotwin_cfg.py:30-31`；`va_robotwin_train_cfg.py:14-17`） | LR `1e-5`；betas `(0.9,0.95)`；wd `0.1`；warmup `10`（`va_robotwin_train_cfg.py:18-22`） | 论文 RoboTwin 50K steps；代码 `batch_size=1`,`grad_accum=1`,`num_steps=50000`,`load_worker=16`（论文 p.13；`va_robotwin_train_cfg.py:23-25`） | 默认 8 GPU FSDP；具体 GPU 型号/训练时间未披露 | 论文 p.13 + 代码确认 |
| LIBERO post-training | LingBot-VA | 训练 transformer | 配置 `snr_shift=5.0`，`action_snr_shift=0.05`；CFG prob 0.1（`va_libero_cfg.py:30-31`；`va_libero_train_cfg.py:14-17`） | LR `1e-5`；betas `(0.9,0.95)`；wd `0.1`；warmup `10`（`va_libero_train_cfg.py:18-22`） | 论文 4K steps、sequence length `1e5`；代码 `batch_size=1`,`grad_accum=10`,`num_steps=5000`（论文 p.13；`va_libero_train_cfg.py:23-25`） | 默认 8 GPU FSDP；具体 GPU 型号/训练时间未披露 | 存在论文/代码步数差异，工程复现优先用实际配置并记录 |
| 真实机器人后训练 | LingBot-VA | 未披露冻结细节 | 未披露除通用 loss 外的任务特定设置 | LR `1e-4` | 每任务 50 demos；500 steps；sequence length 150,000（论文 p.12；`sections/3_experiments.tex:92-93`） | 未披露 | 数据未公开，难以复现 |

## 6. Benchmark 性能表

| Benchmark/任务 | 协议 | LingBot-VA | 主要 baseline | 公平性说明 | 来源 |
|---|---|---:|---|---|---|
| RoboTwin 2.0 Easy/Hard，50 tasks | multi-task；2,500 clean +25,000 randomized demos；Easy 固定初始，Hard 随机；成功率 | Avg 50 Tasks：Easy **92.93**，Hard **91.55**；Horizon=3：Easy 93.22，Hard 93.28 | X-VLA 72.9/72.8；π0 65.9/58.4；π0.5 82.7/76.8；Motus 88.7/87.0 | 论文称 all models 使用同一训练设置；X-VLA 结果 adopted from Motus，仍可能存在实现/输入差异 | 论文 Table 1 p.13；README `README.md:388-425` |
| LIBERO Spatial/Object/Goal/Long | 四 suite；每 suite 10 tasks、500 demos；过滤失败 demos；3 seeds，每 suite 1500 trials；成功率 | Spatial **98.5±0.3**，Object **99.6±0.3**，Goal **97.2±0.2**，Long **98.5±0.5**，Avg **98.5** | X-VLA Avg 98.1；π0 Avg 94.1；π0.5 Avg 96.9；OpenVLA Avg 76.5 | baseline 多数 adopted from X-VLA/reference，未保证相同预训练数据/输入模态；LIBERO 协议本身可比 | 论文 Table 2 p.14；README `README.md:431-469` |
| 真实机器人 6 任务 | 每任务 20 trials；与 π0.5 交替评测；PS=平均步骤得分/最大步骤，SR=成功 trial/N | Make Breakfast 97.0/75.0；Pick Screws 82.5/70.0；Insert Tube 85.8/40.0；Unpack Delivery 84.5/65.0；Fold Clothes 48.8/35.0；Fold Pants 76.7/70.0（PS/SR） | π0.5：73.0/70.0；74.0/50.0；79.2/30.0；73.0/25.0；62.9/30.0；30.0/30.0 | 真实平台和数据未公开；只可作为论文内对照，外部复现公平性低 | README `README.md:478-546`；Appendix A p.23 |
| 消融：RoboTwin Easy | 50 task Easy，成功率 | Ours 92.9；FDM-grounded Async 90.4；Naive Async 74.3；WAN init 80.6 | WAN fine-tune baseline | 同一 RoboTwin 数据、50 task-specific demos、LR `1e-5`、3K steps；只报告 Easy | 论文 Table 3 p.15；`sections/3_experiments.tex:226-256` |
| 推理资源 | RoboTwin/LIBERO server-client；i2av generation | 单卡 RoboTwin offload 约 24GB VRAM；i2av offload 约 18GB VRAM；异步比同步快 2x | 未披露 FPS/latency 数值 | 资源为 README 运行说明，不等同论文评测硬件 | `README.md:210,233`；论文 p.14 |

## 7. 消融与局限

**消融结论**

| 设计 | 结果 | 归因 | 来源 |
|---|---|---|---|
| 异步部署 | FDM-grounded Async 90.4 vs Naive Async 74.3；论文称异步相对同步任务完成 2x faster，性能相近 | 主要来自 forward dynamic prediction 用真实观测刷新 stale predictions，而非盲目异步 | Table 3 p.15；论文 p.14 |
| 预训练 | Ours 92.9 vs WAN init 80.6（RoboTwin Easy all） | joint video-action pretraining 提供 visual-motor priors | Table 3 p.15 |
| action network 初始化 | random init 梯度波动/收敛慢；share weights 稳但非最优；copy/interpolate + scaling 最好 | 来自动作分布与视频 token 分布对齐，稳定 MoT attention | Fig.7 p.15；`sections/2_method.tex:201-204` |
| 低数据样本效率 | 10 demos 下 Make Breakfast PS 比 π0.5 高 15.6%，RoboTwin Easy 高 10.3% | 世界模型的视频动力学先验降低 task adaptation 样本复杂度 | Fig.8 p.15-16；论文正文 p.15 |

**局限与复现风险**

- 完整预训练数据不可复现：约 16K 小时语料包含 internally collected demonstrations，逐数据集小时数、episode 数、采样 manifest 未披露。
- 完整预训练算力不可复现：GPU/TPU 型号、数量、训练天数、global batch、token packing 细节未披露。
- 论文架构与开源实现有差异：论文是双流 MoT `d_a=768`，当前权重/README 标为 shared backbone，代码 action 直接投到 3072 hidden。
- 真实机器人不可复现：硬件平台、相机布置、控制栈、数据和低层 controller 未公开。
- benchmark 公平性不完全统一：LIBERO/RoboTwin 部分 baseline 结果来自已有论文或 Motus/X-VLA 汇总，baseline 预训练数据和输入模态可能不同。
- 推理性能缺口：只有约 18/24GB 显存和 2x 异步加速，没有精确 latency、FPS、控制延迟。
- 复现性评级：**中**。后训练/仿真评测因代码、权重、部分数据开放可复现；完整预训练和真实机器人结果为低复现。

## 8. 最小复现步骤

1. 克隆官方仓库并安装环境。

```bash
git clone https://github.com/robbyant/lingbot-va
cd lingbot-va
conda create -n lingbot-va python=3.10.16
conda activate lingbot-va
pip install torch==2.9.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

来源：README 环境要求 `README.md:85-94`。

2. 下载模型与数据。

```bash
huggingface-cli download robbyant/lingbot-va-base --local-dir /path/to/lingbot-va-base
huggingface-cli download robbyant/lingbot-va-posttrain-robotwin --local-dir /path/to/robotwin-ckpt
huggingface-cli download --repo-type dataset robbyant/robotwin-clean-and-aug-lerobot --local-dir /path/to/robotwin-data
```

来源：README 模型/数据表 `README.md:69-80`、下载示例 `README.md:248-253`。

3. 设置 attention mode。

- 训练：把 `<model>/transformer/config.json` 的 `attn_mode` 设为 `flex`。
- 推理/评测：设为 `torch` 或 `flashattn`。

来源：README `README.md:98-109`。

4. 复现 RoboTwin 评测。

```bash
# server
NGPU=8 CONFIG_NAME=robotwin bash script/run_launch_va_server_sync.sh

# client 侧按 README 安装 RoboTwin，并 checkout 官方指定 commit
git clone https://github.com/RoboTwin-Platform/RoboTwin.git
cd RoboTwin && git checkout 2eeec322
```

来源：README `README.md:122-210`；RoboTwin 单卡 offload 显存约 24GB。

5. 复现 RoboTwin 后训练。

```bash
NGPU=8 CONFIG_NAME=robotwin_train bash script/run_va_posttrain.sh
```

配置：`lr=1e-5`、`betas=(0.9,0.95)`、`wd=0.1`、`warmup=10`、`batch=1`、`steps=50000`；来源 `va_robotwin_train_cfg.py:18-25`。

6. 复现 LIBERO 后训练/评测。

```bash
NGPU=8 CONFIG_NAME=libero_train bash script/run_va_posttrain.sh
NGPU=8 CONFIG_NAME=libero bash script/run_launch_va_server_sync.sh
```

配置：`lr=1e-5`、`grad_accum=10`、代码 `steps=5000`；论文评测写 4K steps，需记录差异。来源 `va_libero_train_cfg.py:18-25`、论文 p.13。

7. 自定义数据最小路径。

- 原始数据转 LeRobot。
- 在 `meta/episodes.jsonl` 增加 `action_config`。
- 用 Wan2.2 VAE 提取视频 latent，放入 `latents/`，结构镜像 `videos/`。
- 建议 resize 到约 256x256、采样 5-15 fps。
- 动作输出维度按 30D canonical，缺失维度补 0。

来源：README `README.md:268-365`。

## 9. 未披露信息清单

| 字段 | 状态 |
|---|---|
| 逐作者机构/affiliation | 未披露；官方材料只显示 Robbyant/Ant Group 标识 |
| 完整 16K 小时预训练 manifest、每数据集小时/episode/帧/token 分布 | 未披露 |
| 预训练 batch size、gradient accumulation、GPU/TPU 型号数量、训练时间、存储规模 | 未披露 |
| 预训练数据中内部数据的下载地址和许可证 | 未披露 |
| 真实机器人硬件、相机数量/分辨率/FPS、控制频率、action/state 维度、低层控制器 | 未披露 |
| 真实机器人训练数据与评测脚本 | 未公开 |
| T5 text encoder 具体 checkpoint | 未披露 |
| 完整 inference latency、FPS、端到端控制延迟、非 offload 显存 | 未披露 |
| EMA、dropout 除 text dropout 0.1 外的完整设置 | 未披露 |
| 论文双流 MoT 5.3B 版本的完整训练代码/配置 | 未公开；当前公开权重说明为 shared backbone |

## 10. 一手资料索引

- 论文/arXiv：[https://arxiv.org/abs/2601.21998](https://arxiv.org/abs/2601.21998)
- 项目主页：[https://technology.robbyant.com/lingbot-va](https://technology.robbyant.com/lingbot-va)
- 官方代码：[https://github.com/robbyant/lingbot-va](https://github.com/robbyant/lingbot-va)
- Hugging Face collection：[https://huggingface.co/collections/robbyant/lingbot-va](https://huggingface.co/collections/robbyant/lingbot-va)
- ModelScope collection：[https://modelscope.cn/collections/Robbyant/LingBot-VA](https://modelscope.cn/collections/Robbyant/LingBot-VA)
- 官方模型：`robbyant/lingbot-va-base`、`robbyant/lingbot-va-posttrain-robotwin`、`robbyant/lingbot-va-posttrain-libero-long`
- 官方数据：`robbyant/robotwin-clean-and-aug-lerobot`、`robbyant/libero-long-lerobot`
