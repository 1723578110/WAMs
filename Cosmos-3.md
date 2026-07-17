# Cosmos 3: Omnimodal World Models for Physical AI

面向工程复现的中文技术调研报告  
调研日期：2026-07-17  
资料范围：仅使用论文/技术报告、官方项目页、官方 GitHub 仓库、官方 Hugging Face 模型/数据集入口、模型/数据/配置文件等一手资料。

## 1. 200 字以内核心结论

Cosmos 3 是 NVIDIA 发布的 omnimodal world model 家族，用一个双塔 Mixture-of-Transformers 同时覆盖 VLM、视频/音频生成、forward dynamics、inverse dynamics 和 robot policy。论文明确开放代码、权重、合成数据和 Cosmos-HUE 评测，许可证为 OpenMDW-1.1，但完整预训练依赖大规模私有/内部数据与 1024-2048 GB200 级算力；全量复现评级为低。工程上最可复现的是基于公开 `Cosmos3-Nano` 的 DROID/LIBERO action-policy SFT，以及推理/评测流水线。关键缺口是完整训练数据清单、部分 optimizer/batch 细节、私有 egocentric/AV 数据、完整预训练脚本与权重初始化细节。

## 2. 研究对象确认

| 字段 | 结论 | 证据 |
|---|---|---|
| 准确标题 | `Cosmos 3: Omnimodal World Models for Physical AI` | 论文首页；arXiv API `2606.02800v4` |
| 作者/机构 | arXiv 作者字段为 NVIDIA，论文首页写 `NVIDIA1`，贡献者列于 Appendix G | 论文 p.1、Appendix G；arXiv API |
| 版本/日期 | arXiv 首发 2026-06-01，v4 更新 2026-06-23；PDF 首页日期为 2026-6-22；GitHub README 新闻写 2026-05-31 released | arXiv API；论文 p.1；`work/cosmos/README.md` |
| 官方论文 | https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf；https://arxiv.org/abs/2606.02800 | 论文 p.1；arXiv API |
| 官方项目页 | https://research.nvidia.com/labs/cosmos-lab/cosmos3/ | 论文 p.1；GitHub README |
| 官方代码 | https://github.com/NVIDIA/cosmos；https://github.com/NVIDIA/cosmos-framework | 论文 p.1；GitHub README |
| 本次代码快照 | `NVIDIA/cosmos` commit `9b5a6ec3c70db02bea9144364b4b33b24337cead`，2026-07-17 `Add SGLang Cosmos3 serving docs (#174)` | 本地 `git log -1` |
| 权重地址 | `nvidia/Cosmos3-Nano`、`nvidia/Cosmos3-Super`、`nvidia/Cosmos3-Super-Text2Image`、`nvidia/Cosmos3-Super-Image2Video`、`nvidia/Cosmos3-Nano-Policy-DROID` | 论文 p.1；`work/cosmos/README.md` Model Family |
| 数据集地址 | 5 个 SDG 合成数据集、`nvidia/Cosmos-HumanEval-v1`；代码还引用 `nvidia/Cosmos3-DROID`、`nvidia/LIBERO_LeRobot_v3` 用于复现 SFT | 论文 p.1、p.97 Table 22；`work/cosmos/cookbooks/cosmos3/generator/action/finetune/README.md` |
| 许可证 | OpenMDW-1.1；源代码和模型均受该许可证约束 | 论文 p.1；`work/cosmos/LICENSE`；`work/cosmos/README.md` License and Contact |
| 实际开放范围 | 开放代码、模型 checkpoint、部分 curated synthetic datasets、Cosmos-HUE；模型/HF guardrail 访问需接受 gated 条款；完整训练原始数据并未全部开放 | 论文 p.1；README quickstart/gated notes；数据章节 p.20-p.25 |
| 同名项目排除 | 目标为 NVIDIA Cosmos 3；官方 `NVIDIA/cosmos` 同仓库还含 Cosmos 1/2 相关历史与工具，需以 `Cosmos 3` README、`cookbooks/cosmos3`、`evaluation/cosmos3` 和论文 `2606.02800v4` 为准 | `work/cosmos/README.md`、`work/cosmos/cookbooks/cosmos3/*` |

## 3. 核心方法与范式

### 3.1 问题与创新

| 项 | 内容 | 证据层级 |
|---|---|---|
| 解决问题 | 将 Physical AI 中理解、未来预测、世界模拟、动作推断和策略控制统一为一个模型，而不是拼接 VLM、视频生成器、forward dynamics、VLA/WAM | 论文 p.5 Introduction |
| 范式 | 论文称为 `omnimodal world model`；可按任务表现为 VLM、world simulator、WAM、VLA-like policy，但核心是 World Model + MoT unified backbone | 论文 p.5、p.7；README Cosmos 3 |
| 核心结构 | 双塔 Mixture-of-Transformers：AR Reasoner tower + diffusion/flow Generator tower，共享 attention 交互但 LN/MLP/投影参数分开 | 论文 p.11 Fig.5、p.12 Eq.7-8 |
| 核心生成方法 | 非文本模态使用 rectified flow matching，预测 velocity；文本 Reasoner 使用 next-token prediction | 论文 p.7、p.11、p.27 |
| 核心位置编码 | 统一 3D MRoPE，语言退化为 1D RoPE；视觉有 t/h/w；音频和动作只走物理时间轴；FPS modulation 以 24 FPS / VAE temporal compression 4 得 `TPSbase=6` | 论文 p.12-p.14 |

### 3.2 输入/输出模态

| Surface | 输入 | 输出 | 代码/论文确认 |
|---|---|---|---|
| Reasoner | text、image、video | text | 论文 p.7、README Reasoner；`cookbooks/cosmos3/reasoner/README.md` |
| Generator | text、image/video conditioning、audio、action/control | image、video、audio、action | 论文 p.9-p.10；README Input and Output |
| Action policy | text + 三视角视觉 + proprioceptive state | 32-step absolute joint-position action chunk，可跳过视频 latent decoding | 论文 p.32；`action_policy_droid_repro.toml` 与 action finetune README |

### 3.3 模型架构表

| Variant | 总规模 | 底座 dense transformer | LLM layers | hidden | heads | KV heads | head dim | FFN | 初始化 | 开放状态 | 证据 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| Cosmos3-Edge | 4B | 2B | 28 | 2048 | 16 | 8 | 128 | 9216 | Edge LLM 从 scratch，设计近 Qwen3-1.7B，去 QK norm、ReLU-squared FFN | 论文称稍后发布；本次未见权重 | 论文 p.14 Table 2 |
| Cosmos3-Nano | 16B | 8B | 36 | 4096 | 32 | 8 | 128 | 12288 | Qwen3-VL-8B | 已发布 | 论文 p.14 Table 2；README Model Family |
| Cosmos3-Super | 64B | 32B | 64 | 5120 | 64 | 8 | 128 | 25600 | Qwen3-VL-32B | 已发布 | 论文 p.14 Table 2；README Model Family |
| Cosmos3-Super-Text2Image | 64B | 同 Super | 64 | 5120 | 64 | 8 | 128 | 25600 | 从 Super mid-trained checkpoint SFT | 已发布 | 论文 p.31；README Model Family |
| Cosmos3-Super-Image2Video | 64B | 同 Super | 64 | 5120 | 64 | 8 | 128 | 25600 | 从 Super mid-trained checkpoint SFT | 已发布 | 论文 p.31；README Model Family |
| Cosmos3-Nano-Policy-DROID | 16B | 同 Nano | 36 | 4096 | 32 | 8 | 128 | 12288 | 从 mid-trained Nano 继续；动作 encoder/decoder/action embeddings fresh init | 已发布 | 论文 p.32；README Model Family |

### 3.4 模块交互与信息流

```text
Language tokens ── tokenizer ─┐
Vision understanding ─ ViT ───┼─ AR subsequence ─ Reasoner tower ── next-token text
                              │        │
                              │        │ causal AR attention only
                              │        ▼
Vision generation ─ Wan2.2 VAE ─┐
Audio generation ─ audio VAE ───┼─ DM subsequence ─ Generator tower ─ rectified-flow denoise ─ image/video/audio/action
Action vector ─ domain proj. ───┘        ▲
                                         │ full attention over [AR; DM] keys/values

注意：DM tokens 可 attend AR + DM；AR tokens 不被 DM 更新，以保持因果自回归路径。
```

| 模块 | 编码/交互方式 | 训练/冻结 | 证据 |
|---|---|---|---|
| Language | tokenizer 后进入 AR subsequence；AR 使用 causal self-attention | Reasoner pretrain/SFT 中训练；Generator pretrain 时 reasoner tower 冻结 | 论文 p.7、p.11-p.12、p.29 |
| Vision understanding | ViT，16x16 patch，2-layer MLP merge 2x2 tokens；DeepStack；视频 timestamps interleaved | ViT 与 backbone 联合训练 | 论文 p.7 |
| Vision generation | Wan2.2-TI2V-5B video VAE；temporal 4x、spatial 32x32 compression；linear projection 到 hidden | VAE frozen | 论文 p.7 |
| Audio | audio VAE；48 kHz stereo，hop 1920，25 tokens/s；linear projection | audio VAE frozen | 论文 p.7 |
| Action | domain-aware input/output projections；不同 embodiment 维度不同，共享 MoT backbone | DROID posttrain fresh init action encoder/decoding MLP/action embedding，action params 5x LR | 论文 p.8、p.32 |
| Position | 3D MRoPE + absolute temporal modulation；action/audio spatial index h=w=0；FPS modulation 对齐物理时间 | 训练/推理一致 | 论文 p.12-p.14 |

### 3.5 动作表示与 action chunk

| Domain | Action dim | 表示 | 证据 |
|---|---:|---|---|
| Camera / autonomous vehicle | 9D | ego pose：3D translation + 6D rotation | 论文 p.8 Fig.3；README Input and Output |
| Single-arm robot | 10D | 9D effector pose + 1D gripper open/close；DROID/UR/Fractal/Bridge/UMI | 论文 p.8 Fig.3；README Input and Output |
| Dual-arm robot | 20D | 两个 10D arm | README Input and Output |
| Egocentric motion | 57D | head pose delta、双腕 pose delta、双手 fingertip 3D positions | 论文 p.8；README Input and Output |
| Humanoid robot | 29D | humanoid robot action | README Input and Output |
| DROID policy chunk | 32 steps @ 15 Hz | absolute joint-position actions；训练同时预测辅助 RGB video frames；推理可跳过视频 latent decoding | 论文 p.32；`action_policy_droid_repro.toml`；action finetune README |

### 3.6 推理步骤、缓存、延迟和显存

| 项 | 结论 | 证据 |
|---|---|---|
| Reasoner 推理 | AR loop；依赖 KV cache；vLLM/TensorRT-LLM 复用 Qwen3-VL optimized attention/paged KV/continuous batching | 论文 p.46、p.48 |
| Generator 推理 | diffusion loop：构造 timestep schedule，反复调用 denoiser，CFG，sampler 更新，decode/postprocess | 论文 p.46 |
| 默认采样 | Nano/Super audiovisual：50 steps、guidance 6、shift 10；T2I specialized：50 steps、guidance 4、shift 3；I2V specialized：50 steps、guidance 6、shift 5；Policy-DROID：4 steps、guidance 3、shift 5 | 论文 p.73 Table 21 |
| DROID 部署 | 4 diffusion steps，shift 5，CFG scale 3，skip video-latent decoding，2x NVIDIA RTX Pro 6000 GPUs | 论文 p.32 |
| Reasoner 显存 | Nano reasoner 单卡：Cosmos Framework full omni ~34GB；vLLM/Transformers/NIM reasoner-only ~16-17GB | `work/cosmos/cookbooks/cosmos3/reasoner/README.md` |
| Generator 延迟 | 官方 repo 有 `inference_benchmarks.md`，按 PyTorch/vLLM-Omni/Diffusers/NIM、GPU、分辨率报告；README 明确空单元表示未测 | `work/cosmos/inference_benchmarks.md`；`work/cosmos/README.md` |
| 控制频率 | Generator 支持 action native control frequency 10-30 FPS；DROID policy 15 FPS、32-step horizon | 论文 p.73；p.32 |

## 4. 训练数据表

### 4.1 Reasoner 数据

| 阶段 | 模态 | 样本数 | 数据来源/处理 | 公开性 | 证据 |
|---|---|---:|---|---|---|
| Pre-training | image-text | 18,814,952 | 19.7M 来自 Nemotron Nano 2 data collection，另 2.3M 加强 math/video/spatial grounding/instruction；semantic dedup + AI-judge filtering | Nemotron Nano 2 原始混合是否完整开放：未披露 | 论文 p.15 Table 3 |
| Pre-training | video-text | 1,016,299 | 同上 | 未披露 | 论文 p.15 Table 3 |
| Pre-training | text-only | 2,170,762 conversations | 同上 | 未披露 | 论文 p.15 Table 3 |
| SFT | image-text | 1,051,513 | Physical AI domain-specific + synthetic data | 部分未披露 | 论文 p.15 Table 3 |
| SFT | video-text | 1,079,200 | 视频占 SFT mixture 约 50%，强化 robotics/smart infrastructure/autonomous vehicle | 部分未披露 | 论文 p.15 |
| SFT | text-only | 40,960 conversations | 另混入 800K lightweight instruction-following dataset；表中 text-only 数为 media/conversation 统计，不等同 SFT 全部预算 | 部分未披露 | 论文 p.15、p.26 |

### 4.2 Generator 图像/视频/音频数据

| 阶段 | 数据 | 数量 | 关键处理 | 公开性 | 证据 |
|---|---|---:|---|---|---|
| Pre-training | images | 767M，来自 7.8B raw images | Qwen3-VL-Embedding-8B embedding，20,000 KMeans clusters，near-duplicate removal；47 类层级标签；aesthetic/photorealism/NSFW/watermark/collage filtering | 原始/完整训练集未公开 | 论文 p.20-p.21 |
| Pre-training | videos | 347.7M clips，来自 3B raw source videos | TransNetV2 scene-change，ffmpeg cropdetect，Cosmos-Embed1-448p，20,000 KMeans，DOVER/VTSS/artifact tags | 原始/完整训练集未公开 | 论文 p.20-p.21 |
| Pre-training | usable audio-video | 138.9M clips；其中 62.5M <30s 用 Qwen3-Omni-Captioner 生成 audio descriptions | 保留广泛真实视频音频 | 原始/完整训练集未公开 | 论文 p.23 |
| Mid-training | image pool | 15.6M samples | 60% real、36% synthetic、4% text-rendering | 未完整公开 | 论文 p.29、p.21 |
| Mid-training | video pool | 74.7M curated clips | 46.0% stricter filtered pretrain videos；43.9% robotics/AV/human/egocentric；10.1% capability hard cases | 未完整公开 | 论文 p.29、p.21 |
| Mid-training | audio pool | 18.8M clips；12.8M non-speech + 6M speech-synchronized | SAM-Audio separation；SyncNet lip-sync >=3.0；FireRedASR speech/music ratio；instrument detection；recaption final waveform | 未公开 | 论文 p.23-p.24 |
| Mid-training | video transfer | 3M videos + MADS 1.1M driving samples | edge/blur on-the-fly；depth via Video Depth Anything；seg via SAMv2；world-scenario-map controls | MADS 是否公开：未披露 | 论文 p.22 |

### 4.3 Action 数据

| Pillar / embodiment | 数量 | 任务/来源 | 输入/频率/维度 | 公开性 | 证据 |
|---|---:|---|---|---|---|
| 全部 action mid-training | 8.4M episodes，61.3K hours | egocentric、robotics、AV、camera motion | text-video-action | 部分私有 | 论文 p.24-p.25 Fig.9 |
| Egocentric motion | 1.7M episodes，41.3K hours，67.4% | proprietary bimanual hand manipulation，head-mounted RGB | head camera pose + hand 21-keypoint 3D pose；57D action | 私有 | 论文 p.24 |
| Autonomous vehicle | 10.0K hours，16.3% | NVIDIA Hyperion in-house driving logs | ego trajectories transformed to front-wide camera frame；9D | 私有 | 论文 p.25 |
| Robotics total | 516.7K episodes，5.36K hours，90.4K tasks | AgiBot、Franka Panda、Google Robot、WidowX-250、UMI、UR | pseudo-actions from state differences；successful + failed episodes | open-source aggregate，部分实际下载需按各源/HF 条款 | 论文 p.25 Table 4 |
| AgiBot | 338 tasks，239.4K episodes，4.37K hours | Bu et al. 2025 | 未披露 | 未披露/按源 | 论文 p.25 Table 4 |
| Franka Panda | 67.5K tasks，76.3K episodes，442h | Wu et al. 2025a；Khazatsky et al. 2024 | 未披露 | 部分公开 | 论文 p.25 Table 4 |
| Google Robot | 599 tasks，87.2K episodes，351h | Brohan et al. 2023b | 未披露 | 公开源 | 论文 p.25 Table 4 |
| WidowX-250 | 21.8K tasks，50.4K episodes，100.1h | Walke et al. 2023 | 未披露 | 公开源 | 论文 p.25 Table 4 |
| UMI | 43 tasks，38.3K episodes，67h | Lin/Ha/Liu/Chi/Wu 等 | 未披露 | 公开源 | 论文 p.25 Table 4 |
| UR | 114 tasks，25.0K episodes，35h | Wu et al. 2025a | 未披露 | 未披露 | 论文 p.25 Table 4 |
| Camera motion | 1.9M clips，4.6K hours，7.5% | 从 pre-training video mining；ViPE + DepthAnything3 | metric scale camera poses；9D | 未公开 | 论文 p.25 |
| DROID policy post-training | 76K trajectories，350h，86 tasks，564 scenes | DROID platform | 360x640 wrist + two 180x320 external views -> 540x640 canvas；15Hz；32 actions | DROID/Cosmos3-DROID success split可按 HF 获取 | 论文 p.32；action finetune README |

### 4.4 开放合成数据

| Dataset | Clips | Resolution / FPS | Modalities / annotations | 证据 |
|---|---:|---|---|---|
| SDG-PhyxSim | 76,489 | 1920x1080 / 30 | RGB、depth、seg、physics、camera、caption | 论文 p.97 Table 22 |
| SDG-RobotSim | 208,022 | varies | RGB，部分 depth/seg/physics，camera，caption | 论文 p.97 Table 22 |
| SDG-DriveSim | 264,000 | 3840x2160 / 24 | RGB、caption | 论文 p.97 Table 22 |
| SDG-SynHuman | 236,937 | 1920x1080 / 30 | RGB、depth、camera | 论文 p.97 Table 22 |
| SDG-Warehouse | 122,952 | 1920x1080 / 30 | RGB、depth、seg、BBox、camera | 论文 p.97 Table 22 |

## 5. 训练 Recipe 和超参数表

| 阶段 | 初始化 | 训练/冻结 | loss | optimizer / schedule | batch / steps / tokens | 精度/并行/硬件 | 证据 |
|---|---|---|---|---|---|---|---|
| Reasoner pre-training | Edge 用内部 Appendix D；Nano 用 Qwen3-VL-8B；Super 用 Qwen3-VL-32B | 从开始 jointly train LLM、projector、ViT；无 projector-only alignment | next-token prediction；sqrt-normalized per-token loss | AdamW；LM/projector peak LR 5e-5，ViT 5e-6；cosine decay 到 0.1x；10% warmup；betas (0.9,0.999)，wd 0.05，clip 1.0 | 2 epochs；max seq 16k；image <=2048 tokens/sample；video <=8192 tokens/sample；global batch 未披露 | GPU/TPU 数、训练时间、precision 未披露 | 论文 p.26 |
| Reasoner SFT | Reasoner pretrain checkpoint | 训练全部/冻结策略未披露 | next-token SFT；混入 pretrain:SFT budget 1:4；含 800K instruction-following | AdamW；LM/projector LR 1e-5，ViT 1e-6；cosine to 0.1x；1000 warmup；betas (0.9,0.95)，wd 0.1，clip 1.0 | 8200 iterations；global batch 512 | precision/GPU 未披露 | 论文 p.26-p.27 |
| Generator pre-training | 由 trained Reasoner 初始化 MoT；Reasoner tower frozen | 只更新 generation-specific parameters；reasoner tower frozen | rectified flow matching masked MSE；logit-normal noise for image/audio/action，mode sampling for video；text dropout 10% | FusedAdamW；LR 1e-4；betas (0.9,0.99)；wd 0.05；clip 1.0；linear decay with warmup to 0.30x floor | Nano 31.05T tokens on 1024 GB200；Super 17.86T tokens on 2048 GB200；fixed 74k token packing | GB200；precision 未披露；multi-res 256/480/720p | 论文 p.27-p.29 |
| Generator pretrain data mix | 同上 | 同上 | 同上 | 同上 | image/video 20/80；video-256 20%、video-480 40%、video-720 20%；T2I/T2V/I2V/V2V = 20/56/16/8 | 256p/480p max 400 frames，720p max 300 frames；10-30 FPS | 论文 p.28 Table 5/Fig.10 |
| Generator mid-training | Generator pretrain checkpoint | 引入 action 和 transfer；preserve visual/audio modes | per-modality velocity MSE sum；action loss scale 10x；action inherits vision noise schedule | FusedAdamW；LR 1e-4；wd 0.05；clip 1.0；loss scale 10；LambdaLinear start factor 0.4 cycle 100000 | Nano 2.4T tokens on 1024 GB200；Super 1.9T tokens on 2048 GB200 | GB200；74k context；shift 256p=3、480p=5、720p=10 | 论文 p.30 Table 6 |
| Generator mid-training mixture | pretrain -> midtrain | 同上 | 同上 | 同上 | Image 10%；Video 32%；Video+Audio 8%；Action 25%；General Transfer 20%；Driving Transfer 5% | 未披露 | 论文 p.30 Table 6 |
| T2I post-training | Cosmos3-Super mid-trained checkpoint | SFT | rectified flow，细项未披露 | Stage1 LR 1e-4，2k warmup，linear decay；其余沿用 mid-training | Stage1 20k steps；45% real、40% synthetic、15% text-rendering；Stage2 2k steps，470K ultra-HQ image-caption；70k context，>720p images | GPU/precision 未披露 | 论文 p.31 |
| I2V post-training | Cosmos3-Super mid-trained checkpoint | SFT | 未披露，I2V formulation | LR 1e-5 | 10k iter，约 50B tokens；480p，189 frames，24fps；混入 20% T2I image tokens；20K synthetic clips约占6% tokens | GPU/precision 未披露 | 论文 p.31 |
| DROID policy post-training | Cosmos3-Nano mid-trained checkpoint；action encoder/action-decoding MLP/action embedding fresh init | action params 5x LR；proprioceptive + 3-view visual | rectified-flow/action policy；代码 loss_scale 10 | 论文 LR 2e-4；代码 cosine cycle 100000；`loss_scale=10` | 代码：10000 iters，global batch 8192，max_samples_per_batch 32，grad_accum 1 | 代码：bf16；HSDP 32x8=256 ranks，64 nodes x4 GB200；VAE tokenizer compile | 论文 p.32；`work/cosmos/cookbooks/cosmos3/generator/action/finetune/toml/sft_config/action_policy_droid_repro.toml` |
| LIBERO-10 policy SFT | Cosmos3-Nano base checkpoint | 代码 recipe | 未披露 | LR 5e-5；warmup 500；cycle 16000 | 2000 iter；global batch 2048；grad_accum 1 | bf16；HSDP 2x8=16 ranks | `action_policy_libero_repro.toml`；action finetune README |
| Vision SFT Nano public recipe | `BASE_CHECKPOINT_PATH` | optimizer keys select `moe_gen,time_embedder,vae2llm,llm2vae`；skip EMA load | 未披露 | Adam/Fused；betas (0.9,0.95)，eps 1e-6，LR 2e-5，wd 0；warmup 50，cycle 1000；grad clip 0.1 | 500 iter；grad_accum 2；max seq 45056 | bf16；FSDP；EMA enabled rate 0.1；torch compile enabled | `vision_sft_nano.toml` |

未披露项：完整全量训练 batch size、gradient accumulation、EMA/dropout、完整 FSDP/HSDP/TP/CP 切分、训练 wall-clock、checkpoint save cadence、所有 post-training 数据采样比例、完整 diffusion timestep 分布参数未在论文中完全给出。

## 6. Benchmark 与性能表

### 6.1 Reasoner

| Benchmark group | 任务数 | Cosmos3-Super | Cosmos3-Nano | 主要对比 | 公平性备注 | 证据 |
|---|---:|---:|---:|---|---|---|
| General Avg. | 19 | 73.7 | 72.8 | Qwen3-VL、Gemma-4、Gemini 3.1 Pro closed 等 | 参数量、闭源模型、预训练数据不可完全对齐 | 论文 p.51 Table 10 |
| Robotics Avg. | 17 | 57.8 | 52.6 | MiMo-Embodied、Qwen3-VL、RynnBrain 等 | 部分 baseline 是专门 embodied/robotics 模型；训练数据不同 | 论文 p.51 Table 10 |
| Smart Infrastructure Avg. | 9 | 62.6 | 56.1 | VANTAGE/TARBench 等 | 自有 benchmark/数据流需关注数据重叠 | 论文 p.51 Table 10 |
| Driving Avg. | 3 | 79.3 | 40.7 | Qwen/Gemini/RynnBrain 等 | Driving 子集少，Nano/Super 差异大；baseline 输入/预训练不同 | 论文 p.51 Table 10 |

### 6.2 Generator / World Model / Action

| Benchmark | 协议 | Cosmos 3 结果 | Baseline 对比 | 公平性备注 | 证据 |
|---|---|---|---|---|---|
| UniGenBench T2I | 1170 prompts；Orig 600 + Phys 570；另 CVTG text rendering、Aesthetic v2、HPSv3 | Cosmos3-Super-Text2Image All 91.36；Super 87.33；Nano 84.61 | FLUX.2-dev 87.60；Gemini 3 Pro Image closed 90.69 | Gemini 为闭源且文字大小写问题另注；数据/agentic harness不同 | 论文 p.55 Table 11 |
| PAIBench-G T2V/I2V | 每 prompt 5 seeds，720p，16:9，189 frames；Qwen2.5-VL-72B judge | Super T2V 80.0 / I2V 82.8；Nano T2V 79.4 / I2V 82.7 | Wan2.2-A14B T2V 78.0/I2V 81.3；Veo-3.1 closed T2V 79.1/I2V 82.6 | 论文说明 public PAIBench-G I2V leaderboard judged by Qwen3-VL-235B-A22B 不可复现，内部改用 Qwen2.5-VL-72B | 论文 p.57 Table 12 |
| RBench I2V | 650 image-text cases；single seed，720p 16:9 121 frames | Nano 58.4%，Super 58.1% | Wan2.2-A14B 50.7%；closed Wan 2.6 60.7% | RBench 来自 RoVid-X，若模型训练包含相似机器人数据需谨慎 | 论文 p.57 Table 12 |
| Physics-IQ | I2V/V2V；396 scenes，五类物理；0-100 score | Super I2V 43.8；I2V + WMReward BoN 48.9；V2V 59.7；V2V + BoN 63.4 | Sora2 I2V 42.3/BoN 46.4；Magi-1 V2V 56.0/BoN 62.6 | BoN/WMReward 属推理时 rerank，不等同单次生成 | 论文 p.58 Table 13 |
| Cosmos HUE / HWB | HUE：100 PAIBench-G prompts，5 seeds=500 videos/model，二人标注+QC；HWB 180 EgoVerse samples | Super HUE T2V 89.3/I2V 89.6/HWB 71.9；Nano 87.6/88.6/66.9 | Veo-3.1 closed 91.3/89.7/67.8；Wan2.2-A14B 88.2/88.4/60.7 | HUE 由 GPT 5.2 生成问题并人工标注；Ground Truth 93.6/94.4 是上界参考 | 论文 p.59 Table 14 |
| SoundBench AVQ | AVQ/SAV/SA/AVAlign/Visual support/PQ | Nano AVQ 7.34，SAV 8.35；Super AVQ 7.31，SAV 8.34 | Seedance-1.5-Pro closed AVQ 7.64；Veo-3.1 7.45 | Cosmos 语义 AV grounding 强，低层 PQ 不如闭源 | 论文 p.62 Table 15 |
| PAIBench-C transfer | 600 clips：AgiBot/OpenDV/Ego-Exo-4D；depth/seg/blur/edge single control | Nano DOVER 10.39、Seg mIoU 0.72；Super Edge F1 0.50、Depth si-RMSE 0.58 | Cosmos-Transfer2.5 DOVER 9.49、Seg 0.68、Edge 0.45、Depth 0.68 | Cosmos 统一 backbone vs baseline per-modality ControlNet；结构不同但任务相同 | 论文 p.63 Table 16 |
| AVBench-C transfer | 486 single-view driving clips；auto + human | Super ego drift 0.003，video quality 2.86；Nano dyn obj 0.67，lane 2.50 | Cosmos-Transfer2.5-AV-Singleview ego drift 0.008，video quality 2.59 | baseline 是专用 AV transfer；Cosmos 统一模型 | 论文 p.64 Table 17 |
| Action FD/ID | AV ID、camera FD、egocentric FD、robotics FD | Super MT-init AV RRE 0.232/RTE 0.014/ATE 0.90；robotics PSNR 26.04；Nano MT-init robotics PSNR 25.52 | VGGT AV ATE 23.46；DepthAnything3 9.29；Ctrl-World robotics PSNR 22.99 | baseline 是 domain-specific，指标不同列只在对应任务比 | 论文 p.65 Table 18 |
| RoboLab-120 DROID policy | 120 tasks，10 rollouts/task；success rate by instruction specificity/difficulty | Cosmos3-Nano-Policy-DROID overall vague/default/specific = 20.6/36.8/39.7 | pi0.5 = 15.2/28.0/28.1；DreamZero = 14.9/25.7/23.9 | 均为 off-the-shelf checkpoints fine-tuned on DROID；相对公平，但模型预训练不同 | 论文 p.68 Table 19 |
| LIBERO-10 | closed-loop success；500 rollouts/checkpoint | MT-init 500/1000/1500/2000 iter = 24.6/91.4/95.8/97.4%；PT-init = 0.0/73.8/93.4/95.2% | 自身初始化消融 | 同 recipe/data/compute，较公平 | 论文 p.70 Table 20 |

### 6.3 训练/推理吞吐

| 项 | 数字 | 证据 |
|---|---:|---|
| 训练吞吐 Nano | 7.1 sec/iter，520 TFLOPS/GPU，MFU 0.23，507 iter/hr，4.56M image tok/hr/GPU，16.23M video tok/hr/GPU | 论文 p.46 Table 8 |
| 训练吞吐 Super | 19.5 sec/iter，673 TFLOPS/GPU，MFU 0.30，185 iter/hr，1.66M image tok/hr/GPU，5.91M video tok/hr/GPU | 论文 p.46 Table 8 |
| 吞吐实验硬件 | Nano 2048 GB200，Super 4096 GB200 | 论文 p.46 Table 8 |
| Batching speedup | T2V 189 frames；256p B=6，480p B=3，720p 因 74k context 只 B=1；Nano H100 8%/2%，GB200 40%/2% | 论文 p.48 Table 9 |

## 7. 消融与局限

### 7.1 关键消融

| 消融 | 设置 | 结果 | 解释 | 证据 |
|---|---|---|---|---|
| SDG 数据 | Cosmos3-Nano 分别 fine-tune 单一 SDG 或 SDG-All，PAIBench-G T2V | SDG-SynHuman overall +0.12，Robot +1.23；SDG-DriveSim overall +0.09；SDG-All overall +0.10 | 合成数据主要带来 domain-specific 小幅收益，不是唯一性能来源 | 论文 p.104 Table 26 |
| Understanding tower | Generator tower 从 scratch；Cosmos3 Reasoner vs Qwen3-VL-8B；256/480p，0-200 frames，130K iterations on 256 GPUs | Reasoner 在 T2V/I2V PAIBench domain scores 更高；T2V Robot +4.8，AV +2.3 | Reasoner 预训练对 Generator 的 Physical AI domains 有迁移 | 论文 p.107 Table 28 |
| FPS control | Base、Text Control、MRoPE FPS、Text+MRoPE；10/15/24/30 FPS，每 prompt 3 seeds，480p 5s | Composite：Base 8.51，Text 9.28，MRoPE 9.63，Text+MRoPE 9.81 | FPS modulation 是主要收益，文本控制叠加最好 | 论文 p.108 Table 29 |
| Audio pretraining | 加入 audio data vs 不加 | 具体表格在 Table 30，但文本抽取未完整保留所有数值；论文结论为报告 PAIBench T2V/I2V 分数 | 不在报告中补数 | 论文 p.109 Table 30 |
| Action-mode synergy | PushT；单 FD/ID/policy 各 2K steps vs joint FD/ID/policy 6K steps | joint checkpoint 改善 action-side metrics，同时 FD PSNR 相当 | 多 action mode 共享有益 | 论文 p.109 Table 31 |
| MT-init vs PT-init | RoboLab/LIBERO/action FD/ID | DROID policy final > PT-init；LIBERO 500 iter MT-init 24.6% vs PT-init 0.0%，2000 iter 97.4% vs 95.2% | action mid-training 是策略快速适应的核心来源 | 论文 p.68 Table 19、p.70 Table 20 |

### 7.2 局限与复现风险

| 类别 | 风险 |
|---|---|
| 数据依赖 | 图像/视频/音频完整预训练数据为数十亿 raw candidates，未开放；egocentric motion 和 AV logs 为私有/内部数据。 |
| 算力依赖 | 论文主训练使用 1024-2048 GB200；吞吐实验甚至 2048/4096 GB200；普通实验室只能做 SFT/LoRA/小规模验证。 |
| 代码完整性 | 官方 `NVIDIA/cosmos` 包含 cookbooks、评测、推理和部分 SFT 配置；完整预训练/中训练生产配置未完全在本仓库暴露。 |
| 模型访问 | 权重称 open-weight，但 HF repos/guardrail 需要接受 gated terms；自动化复现需准备 HF token。 |
| 评测公平性 | 多数 baseline 训练数据、预训练规模、闭源/开源状态不同；PAIBench-G 公榜 judge 与内部 judge 不同，论文也指出 public I2V leaderboard 不可复现。 |
| 安全与部署 | README 明示长、高分辨率、复杂物理输出会有 temporal inconsistency、object morphing、3D/physics errors；安全关键控制需额外验证。 |
| 未披露项 | 全量数据采样细节、随机种子、完整训练 wall-clock、所有 optimizer eps/dropout/EMA、完整并行拓扑、完整 checkpoint conversion 细节。 |

可复现性评级：**低（全量预训练/中训练）**；**中（公开 DROID/LIBERO policy SFT 和推理评测）**；**高（单机/多机 inference smoke test、官方 benchmark assets 的局部运行）**。

## 8. 最小复现步骤

### 8.1 推荐最小目标

不要把“从零训练 Cosmos3-Nano/Super”作为第一复现目标。工程上推荐复现：

1. Cosmos3-Nano Reasoner/Generator 推理。
2. Cosmos3-Nano-Policy-DROID 的 DROID SFT cookbook，或 LIBERO-10 SFT。
3. PAIBench/RBench/Cosmos-HUE 局部评测脚本或样例数据。

### 8.2 环境准备

```bash
git clone https://github.com/NVIDIA/cosmos.git
git clone https://github.com/NVIDIA/cosmos-framework.git packages/cosmos3
cd packages/cosmos3
uv sync --all-extras --group=cu130-train   # CUDA 13；CUDA 12.8 用 cu128-train
uvx hf@latest auth login
```

证据：`work/cosmos/cookbooks/cosmos3/generator/action/finetune/README.md`；`work/cosmos/README.md`。

### 8.3 推理 smoke test

Reasoner-only 优先使用 vLLM/Transformers/NIM，因为只加载 Reasoner tower；Nano 单卡约 16-17GB，而 Cosmos Framework full omni 约 34GB。Generator 需要 Diffusers/vLLM-Omni/SGLang/NIM 并接受 guardrail/model gated terms。

```python
from transformers import AutoProcessor, Cosmos3OmniForConditionalGeneration
import torch

model_id = "nvidia/Cosmos3-Nano"
processor = AutoProcessor.from_pretrained(model_id)
model = Cosmos3OmniForConditionalGeneration.from_pretrained(
    model_id, dtype=torch.bfloat16, device_map="auto"
)
```

证据：`work/cosmos/cookbooks/cosmos3/reasoner/README.md`。

### 8.4 DROID policy SFT 复现

```bash
cd cookbooks/cosmos3/generator/action/finetune
uvx hf@latest download nvidia/Cosmos3-DROID --repo-type dataset --local-dir data/Cosmos3-DROID
bash launch_sft_action_policy_droid.sh
```

参考配置：

| 参数 | 值 | 证据 |
|---|---:|---|
| base model | Cosmos3-Nano | action finetune README |
| dataset | `nvidia/Cosmos3-DROID` success split | action finetune README |
| action | `joint_pos` 8-D + proprioceptive state；concat_view 480p；chunk length 32 | action finetune README |
| precision | bf16 | `action_policy_droid_repro.toml` |
| LR | 2e-4 | 论文 p.32；TOML |
| loss_scale | 10 | TOML |
| max_iter | 10000 | TOML |
| global batch | 8192 | TOML/README |
| parallel | HSDP 32x8 = 256 ranks，GB200 reference 64 nodes x4 | TOML/README |
| 单 8-GPU 降配 | replicate_degree=1，grad_accum_iter=32 保持 global batch | TOML comment/README |

### 8.5 LIBERO-10 policy SFT 复现

```bash
cd cookbooks/cosmos3/generator/action/finetune
bash launch_sft_action_policy_libero.sh
```

参考配置：bf16，LR 5e-5，warmup 500，cycle 16000，global batch 2048，HSDP 2x8，2000 iter；证据见 `action_policy_libero_repro.toml` 和 action finetune README。

### 8.6 算力/时间/存储估算

| 目标 | 估算 | 依据 |
|---|---|---|
| 全量 Nano generator pretrain | 不建议；至少 1024 GB200 级，31.05T tokens | 论文 p.29 |
| 全量 Super generator pretrain | 不建议；至少 2048 GB200 级，17.86T tokens | 论文 p.29 |
| Mid-training | 不建议；Nano 2.4T tokens/1024 GB200，Super 1.9T tokens/2048 GB200 | 论文 p.30 |
| DROID policy reference SFT | 256 ranks GB200 reference；可用 8 GPU + grad accumulation 做慢速复现/烟测 | `action_policy_droid_repro.toml` |
| Reasoner Nano inference | 16-17GB reasoner-only；34GB full omni | reasoner README |
| 权重显存 | Nano 16B bf16 仅参数约 32GB，Super 64B bf16 仅参数约 128GB；实际需加 KV/cache/activations/offload | 根据参数量估算；README 提供 Nano 实测显存 |
| 数据存储 | DROID/Cosmos3-DROID、LIBERO、SDG 数据精确大小未披露；README 仅称 DROID dataset is large | 代码 README；不补具体 GB/TB |

## 9. 未披露信息清单

| 类别 | 未披露/无法一手确认内容 |
|---|---|
| 作者 | 论文正文只以 NVIDIA 和 Appendix G 贡献者呈现；arXiv API 有大量作者字段，但正式作者排序/角色需以 Appendix G 为准。 |
| 全量数据 | 原始图像/视频/音频来源列表、URL、版权状态、每源采样权重、完整过滤阈值、随机种子未披露。 |
| 私有数据 | Egocentric bimanual dataset、NVIDIA Hyperion AV logs、内部 surveillance clips 不开放。 |
| 训练 recipe | Reasoner pretrain/SFT 的完整 batch/precision/GPU 数/训练时间未披露；Generator 全量 dropout/EMA/grad accumulation/checkpoint cadence 未披露。 |
| 模型细节 | 完整 tokenizer vocab、special token id、所有 projection/embedding 维度、MoT 参数初始化映射细节未在论文表格完全展开。 |
| 评测 | 多数 baseline 是否完全相同 prompt upsampling/negative prompt/seed 未统一；部分闭源 baseline 训练数据不可查。 |
| 权重开放 | 权重是 open-weight 但 gated 访问和 guardrail 依赖需人工接受条款；实际商用/再分发需逐条审阅 OpenMDW-1.1 与第三方许可证。 |
| 数据集卡 | 本次未下载完整 HF dataset card 内容；仅确认论文和代码列出的官方地址/用途。 |

## 10. 一手资料链接

- 官方项目页：https://research.nvidia.com/labs/cosmos-lab/cosmos3/
- 官方技术报告：https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf
- arXiv：https://arxiv.org/abs/2606.02800
- 官方代码：https://github.com/NVIDIA/cosmos
- Cosmos Framework：https://github.com/NVIDIA/cosmos-framework
- Hugging Face collection：https://huggingface.co/collections/nvidia/cosmos3
- OpenMDW-1.1 license：https://openmdw.ai/license/1-1/

