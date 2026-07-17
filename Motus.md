# Motus

## 1. 核心结论（200字以内）

Motus 是一个统一的 latent-action world model：用 Wan2.2-5B、Qwen3-VL-2B、Action Expert、Understanding Expert 组成 MoT，并用 UniDiffuser/flow-matching 调度在 VLA、World Model、IDM、VGM、视频-动作联合预测之间切换。论文和模型卡公开了权重、Stage 3 微调代码和 RoboTwin2 checkpoint，但 Stage 1/2 的完整数据清洗、采样比例、集群配置和训练脚本未完全开源，工程复现评级为“中”。

## 2. 研究对象确认

| 项目 | 结论 | 证据 |
|---|---|---|
| 准确论文 | `Motus: A Unified Latent Action World Model` | arXiv 页面 title/meta；官方主页标题；README BibTeX（`README.md:189-195`）。 |
| 作者 | Hongzhe Bi, Hengkai Tan, Shenghao Xie, Zeyuan Wang, Shuhe Huang, Haitian Liu, Ruowen Zhao, Yao Feng, Chendong Xiang, Yinze Rong, Hongyan Zhao, Hanyu Liu, Zhizhong Su, Lei Ma, Hang Su, Jun Zhu | arXiv HTML `citation_author` 与页面作者列表；官方主页 citation（`motus_home.html:709-716`）。 |
| 机构 | Tsinghua University, Shengshu, Peking University, Horizon Robotics | 官方主页作者单位（`motus_home.html:80-84`）。代码 LICENSE 版权为 `TSAIL Group, Tsinghua University`（`LICENSE:189`）。 |
| 版本/日期 | arXiv v1: 2025-12-15 06:58:40 UTC；v2: 2025-12-25 08:16:05 UTC；当前版本 v2 | arXiv 页面 dateline/submission history（`arxiv_abs.html:126`, `arxiv_abs.html:171-174`）。 |
| 官方论文 | <https://arxiv.org/abs/2512.13030>，HTML: <https://arxiv.org/html/2512.13030v2> | arXiv 页面。 |
| 项目主页 | <https://motus-robotics.github.io/motus> | 官方主页按钮链接（`motus_home.html:88-100`）。 |
| 代码仓库 | <https://github.com/thu-ml/Motus.git>；本次核对 commit `f771216802b8a1601599422f12088bee3c068c14` | 官方主页按钮（`motus_home.html:94`）；本地 `git rev-parse HEAD`。 |
| 模型权重 | `motus-robotics/Motus_Wan2_2_5B_pretrain`（Stage 1 VGM）、`motus-robotics/Motus`（Stage 2）、`motus-robotics/Motus_robotwin2`（Stage 3 RoboTwin2） | README checkpoint 表（`README.md:105-107`）；HF 模型卡。 |
| 数据集地址 | 论文未提供完整混合数据下载地址；仓库提供 RoboTwin 转换工具和支持的数据格式；HF 未确认到完整 DatasetDemo 可用卡片 | README data format（`README.md:138-147`）；`DATA_FORMAT.md:9-72`。 |
| 许可证 | 代码 Apache-2.0；Motus HF 权重 Apache-2.0；依赖 backbone Qwen3-VL-2B 与 Wan2.2-TI2V-5B 也为 Apache-2.0 | `LICENSE:1-3,189-201`；HF Motus card lines 52-55；Qwen card lines 58-59；Wan card lines 58,232-234。 |
| 实际开放范围 | 开源训练/推理代码、Stage 1/2/3 权重、配置样例；未开源完整训练数据、Stage 1/2 大规模训练脚本和完整日志 | README 权重与训练说明（`README.md:99-121`, `TRAINING.md:17-130`）；附录数据表仅列 size。 |
| 同名/近名排除 | `MotuBrain: An Advanced World Action Model for Robot Control` 是 2026-04 后续 WAM，明确“building on Motus”，不是本报告对象 | arXiv 2604.27792，非 2512.13030。 |

## 3. 模型架构表

**范式判断：**论文明确称 Motus 为 `unified latent action world model`，同时支持 WMs、VLAs、IDMs、VGMs、Video-Action Joint Prediction Models（摘要 `0_abstract.tex:2`；README `README.md:41`）。工程上可归入 WAM/World-Action-Model 家族，且包含 VLA 推理模式。

| 模块 | 参数/结构 | 输入 | 输出 | 训练状态 | 证据与状态 |
|---|---:|---|---|---|---|
| VGM / Generative Expert | Wan2.2-TI2V-5B，约 5.00B；Wan2.2 VAE 压缩比来自 base card：16x16x4，TI2V-5B 总压缩到 4x32x32 | 条件帧、语言/T5 embedding、带噪未来视频 latent、动作 tokens | 未来视频 velocity / frames | Stage 1 只训练 VGM；Stage 2/3 与其他专家联合 | 论文附录表 `architecture_details`（`X_appendix.tex:350-358`）；HF Wan card lines 112-113, 215-216。 |
| VLM backbone | Qwen3-VL-2B，约 2.13B；原模型卡为 2B params、BF16 | 初始图像/多视角拼接图、语言指令 | last-layer 对应 tokens / image features | Stage 2 论文说 VLM frozen；代码中 VLM `requires_grad=False` | 论文正文（`4_methodology.tex:238`）；代码（`motus.py:523-531`）；HF Qwen card lines 391-397。 |
| Understanding Expert | hidden 512，30 layers，24 heads，FFN 2048，GELU，LayerNorm eps 1e-5，约 253.5M | VLM 最后一层 token 特征 | understanding tokens | 可训练模块 | 附录表（`X_appendix.tex:332-359`）；代码 config（`robotwin.yaml:65-69`，`und_expert.py:37-104`）。 |
| Action Expert | hidden 1024，30 layers，24 heads，FFN 4096，GELU，norm eps 1e-5（附录）/代码默认 1e-6 但 robotwin.yaml 为 1e-5，约 641.5M | state token、noisy action chunk、register tokens | action velocity / predicted action chunk | 可训练模块 | 附录表（`X_appendix.tex:325-359`）；代码（`robotwin.yaml:56-60`，`action_expert.py:60-65,250-271`）。 |
| Tri-model Joint Attention | 每层将 video/action/understanding 三路 attention 的 QKV 对齐到 Wan head space 并联合交互；FFN 各专家独立 | 三路 tokens | 三路更新后的 tokens | 可训练/部分来自 pretrained | 论文正文（`4_methodology.tex:17-18`）；代码交互循环（`motus.py:853-871,983-1003`）。 |
| Latent Action VAE | optical flow -> RGB flow -> DC-AE -> 4x512 latent tokens -> lightweight encoder -> 14D latent action；loss: recon + lambda_a action alignment + beta KL；lambda_a=1.0, beta=1e-6 | 相邻帧 optical flow，少量真实动作监督 | 14D latent action | 用于 latent action 预训练 | 论文正文（`4_methodology.tex:183-196`）；附录表（`X_appendix.tex:339-340`）。 |
| Scheduler / 生成方式 | rectified flow / flow matching；训练 timesteps 1000；推理表为 10 steps、Logit Normal；HF quickstart 示例用 20；代码默认函数参数 50 | 各模态 clean/noisy 状态和 timestep | velocity field | 训练与推理按模式切换 | 附录算法 `alg:train`、`alg:vgm`、`alg:wm`、`alg:idm`、`alg:vla`、`alg:joint_pred`（`X_appendix.tex:33-135`）；代码（`motus.py:643-667,956-963,1107-1128,1276-1280`）。 |
| Action representation | action_dim=14；state_dim=14；论文表 action chunk=48 @ 30Hz，video frames=8 @ 5Hz；HF card: bimanual 7 per arm | bimanual joint/action state | `[action_chunk_size, action_dim]` | Stage 3 用真实动作替换 latent action | 附录表（`X_appendix.tex:343-344`）；HF Motus card lines 122-127；代码（`robotwin.yaml:3-13`）。 |

**ASCII 架构图**

```text
language instruction l --------------------+
                                           v
condition image / 3-view concat --> Qwen3-VL-2B (frozen in Stage 2)
                                           |
                                           v
                                  Understanding Expert
                                           |
future video latent z_v(tau_v) --> Wan2.2 VGM blocks
                                           |
state + noisy action a(tau_a) --> Action Expert
                                           |
              +---------------- Tri-model Joint Attention ----------------+
              | shared cross-modal attention every layer, separate FFNs   |
              +-----------------------------------------------------------+
                         |                                  |
                         v                                  v
               video velocity v_o                    action velocity v_a
                         |                                  |
        VGM / World Model / joint video output       VLA / IDM / joint action output
```

### 信息流与推理模式

| 模式 | 条件 | 去噪对象 | 论文/代码证据 |
|---|---|---|---|
| VGM | `o_t, l` | future observations | `X_appendix.tex:57-75`。 |
| World Model | `o_t, a_{t+1:t+k}, l` | future observations | `X_appendix.tex:77-93`。 |
| IDM | `o_{t:t+k}, l` | actions | `X_appendix.tex:96-111`；IDM baseline MSE 见 `X_appendix.tex:193-209`。 |
| VLA | `o_t, l` | actions | `X_appendix.tex:115-130`；VLA 83.90 vs Joint 87.02 见 `X_appendix.tex:213-228`。 |
| Joint | `o_t, l` | observations + actions | `X_appendix.tex:134-150`。 |

训练与推理的信息流不同：训练同时为 observation/action 采样独立 timestep/noise 并回归 velocity；推理时通过把某些模态固定为 clean 或 noise 来切换条件分布（`X_appendix.tex:7-34,57-135`）。

## 4. 训练数据表

论文没有披露小时、总帧数、token 数、完整帧率、相机数量、分辨率、每数据集清洗规则和混合采样比例。表中 `Size` 是论文附录原列名；结合实验上下文可理解为轨迹/episode/demo 数，但论文未显式定义单位，因此不把它改写为小时或帧数。

| 数据集 | Size | embodiment | 数据级别/用途 | 模态与动作 | 公开性 | 证据 |
|---|---:|---|---|---|---|---|
| EgoDex | 230,949 | Human | Level 2: Egocentric Human Videos；Stage 1/2 | 论文只说明 egocentric human videos；动作维度未披露 | 公开数据集本身可查，但 Motus 使用子集/处理未披露 | `X_appendix.tex:368-376`；Stage 表 `4_methodology.tex:289-291`。 |
| AgiBot | 728,209 | Genie-1 Robot | Level 5: Multi-Robot Task Trajectory | 语言/图像/动作轨迹；具体相机、Hz、动作维度未披露 | 公开数据集本身可查，Motus 处理未披露 | `X_appendix.tex:376`。 |
| RDT | 6,083 | Aloha Robot | Level 5 | 未披露 | 公开/来源数据可查，Motus 处理未披露 | `X_appendix.tex:377`。 |
| RoboMind Franka | 9,589 | Franka Robot | Level 5 | 未披露 | 公开/来源数据可查，Motus 处理未披露 | `X_appendix.tex:378`。 |
| RoboMind Aloha | 7,272 | Aloha Robot | Level 5 | 未披露 | 公开/来源数据可查，Motus 处理未披露 | `X_appendix.tex:379`。 |
| RoboTwin | 27,500 | Aloha Robot | Level 3 synthetic；RoboTwin2 sim benchmark SFT | 论文评测：50/task clean + 500/task randomized；代码 action/state 14D | RoboTwin 可下载/转换；Motus 提供转换工具 | `X_appendix.tex:382`；实验协议 `5_experiments.tex:57`；代码 `robotwin.yaml:3-13`。 |
| Task-Agnostic Data | 1,000 | Aloha Robot | Level 4；latent action VAE 对齐 | image-action pairs；Curobo 随机采样目标机器人动作空间 | Motus 具体数据未公开 | `4_methodology.tex:187-189`；`X_appendix.tex:383`。 |
| In-house Data | 2,000 | Aloha Robot | Level 6 target-robot trajectory | 真实机器人目标数据 | 未公开 | `X_appendix.tex:384`。 |
| Real-world AC-One / Agilex-Aloha-2 | 每任务 100 trajectories | 双臂平台 AC-One、Agilex-Aloha-2 | Stage 3 实验 | 真实任务轨迹；代码 real-world loader 使用视频、qpos、instruction/T5 | 论文未公开完整数据 | `5_experiments.tex:66`；`DATA_FORMAT.md:31-60`。 |

**数据处理与归一化**

| 项目 | 明确内容 | 未披露/备注 |
|---|---|---|
| 视频/动作采样 | 论文附录给出 8 video frames @ 5Hz、48 action chunk @ 30Hz；正文说明 video sparse/action dense，示例视频帧率为动作帧率 1/6 | 代码 `robotwin.yaml` 当前是 `num_video_frames=8`, `video_action_freq_ratio=2`, `global_downsample_rate=3`，会得到 action chunk 16；这与论文附录 48 不一致，复现实验应以论文实际配置优先。 |
| 多视角 | 推理要求输入图像为 head + left/right wrist 三视角拼接 | 训练真实相机数量/分辨率未完整披露；RoboTwin converter 代码含 head/left_wrist/right_wrist。证据：`INFERENCE.md:30`，`robotwin_converter.py:229-317`。 |
| 归一化 | real-world loader 对 action 和 initial state 做 min-max 到 `[0,1]` | RoboTwin loader 中归一化代码被注释；真实评测是否完全相同未披露。证据：`ac_one_dataset.py:409-417`；`norm.py:14-66`；`robotwin_agilex_dataset.py:332-334`。 |
| latent action | DPFlow optical flow -> RGB flow -> DC-AE -> 4x512 -> 14D；90% unlabeled reconstruction + 10% labeled weak action supervision | DPFlow 具体版本、flow 分辨率、VAE 训练 batch/steps 未披露。证据：`4_methodology.tex:183-196`。 |
| 混合比例 | latent action VAE 训练 90%/10%；其他 Stage 1/2 数据混合比例未披露 | 不补全。 |

## 5. 训练 Recipe 和超参数表

### 论文实验配置

| 阶段 | 初始化 checkpoint | 训练模块 | 数据 | batch | LR | optimizer | WD | GPU hours | loss/目标 | 证据 |
|---|---|---|---|---:|---:|---|---:|---:|---|---|
| Foundation | off-the-shelf Wan2.2/Qwen3-VL | VGM/VLM 已预训练 | Level 1 web data | 未披露 | 未披露 | 未披露 | 未披露 | 未披露 | base 模型原始目标 | `4_methodology.tex:280-289`；README `README.md:161-166`。 |
| Stage 1 VGM Training | Wan2.2-TI2V-5B | Only VGM | Level 2/3/5 | 256 | 8e-5 | AdamW | 0.01 | ~8000 | TI2V / video generation flow objective | `4_methodology.tex:237,289`；`X_appendix.tex:397-408`；HF Stage1 card lines 119-123。 |
| Stage 2 Motus Pretraining | Stage 1 VGM + Qwen3-VL；Action/Understanding 专家初始化未完全披露 | Motus all 3 experts；正文称 VLM frozen | Level 2/3/4/5，latent actions | 256 | 5e-5 | AdamW | 0.01 | ~10000 | `l = l_action + l_obs`；latent/action/video flow matching | `4_methodology.tex:18-44,238,291`；`X_appendix.tex:397-408`。 |
| Stage 3 SFT | Stage 2 `motus-robotics/Motus` | Motus all 3 experts with real actions；代码支持 fine-tune from checkpoint | Level 6 target robot；RoboTwin2/real-world | 256（论文）；代码样例每 GPU 8 | 1e-5~5e-5（论文）；代码 robotwin 5e-5 | AdamW | 0.01 | ~400 | 同 Stage 2，但 latent action 替换为 target robot action | `4_methodology.tex:239,293`；`X_appendix.tex:397-408`；`TRAINING.md:60-66`；`robotwin.yaml:104-113`。 |

### 开源代码确认的 Stage 3 配置

| 项目 | 值 | 证据 |
|---|---|---|
| action/state dim | 14 / 14 | `configs/robotwin.yaml:3-4`。 |
| video frames | 8 | `configs/robotwin.yaml:7`。 |
| code action chunk | `num_video_frames * video_action_freq_ratio`；robotwin.yaml 为 `8 * 2 = 16` | `train.py:66`；`robotwin.yaml:12-13`。 |
| checkpoint | `finetune.checkpoint_path` 可设为 `./pretrained_models/Motus` | `TRAINING.md:60-66`。 |
| optimizer | AdamW, betas=(0.9, 0.95), weight_decay=config | `train.py:493-496`。 |
| epsilon | 未显式设置，PyTorch AdamW 默认 epsilon；论文未披露 | `train.py:493-496`。 |
| scheduler | 自定义 linear warmup/decay；robotwin warmup 200, cycle_length 5,000,000, f_max 0.99 | `utils/scheduler.py`；`robotwin.yaml:111-115`。 |
| max_steps | config 为 1,000,000；论文 RoboTwin 实验为 40k fine-tuning steps | `robotwin.yaml:105`；`5_experiments.tex:57`；HF robotwin2 lines 129-133。 |
| precision | bf16 mixed precision；DeepSpeed bf16 enabled | `train.py:603-604`；`configs/zero1.json:2-11`。 |
| gradient accumulation | config/DeepSpeed `auto` 或 `config.training.gradient_accumulation_steps`，默认 1；具体实验未披露 | `train.py:600-604`；`zero1.json:5-8`。 |
| gradient clipping | 代码有 clipping 分支，但具体 max norm 未从当前片段确认；论文未披露 | `train.py:312-320`。 |
| parallelism | 开源脚本：single-node `torchrun --nproc_per_node=8` + DeepSpeed ZeRO stage 1；SLURM 脚本支持多节点 | `scripts/train.sh:17-25`；`TRAINING.md:19-23`；`zero1.json:8-11`。 |
| EMA/dropout | 未披露 | 无一手证据。 |
| GPU 类型/数量 | 论文只给 GPU hours；README 建议训练 >80GB, A100/H100/B200；具体实验节点数未披露 | `X_appendix.tex:405`；`README.md:68-72`。 |

## 6. Benchmark 性能表

| Benchmark / setting | 训练数据与协议 | Baseline | Motus | 指标 | 公平性备注 | 证据 |
|---|---|---:|---:|---|---|---|
| RoboTwin 2.0 clean, 50+ tasks | clean 2,500 demos + randomized 25,000 demos；所有模型从 pretrained checkpoint fine-tune 40k steps；100 execution trials/task | GO-1 37.8；pi0.5 42.98；X-VLA 72.80；w/o pretrain 77.56；Stage1 82.26 | 88.66 | success rate % | 论文声称所有模型 40k fine-tune，较公平；但 baseline 预训练数据/输入差异未完全展开 | `5_experiments.tex:57-61`；`X_appendix.tex:423-494`。 |
| RoboTwin 2.0 randomized, 50+ tasks | 同上；随机背景、桌面杂物、桌高扰动、光照 | GO-1 36.24；pi0.5 43.84；X-VLA 72.84；w/o pretrain 77.00；Stage1 81.86 | 87.02 | success rate % | 同上；Motus 使用自有 Stage2 latent pretraining，baseline 是否拥有等量额外数据未披露 | `5_experiments.tex:50,57-61`；`X_appendix.tex:491`；HF robotwin2 lines 146-154。 |
| Real-world AC-One | 每任务 100 trajectories；多任务联合训练；partial success rate | pi0.5 avg 14.79；w/o pretrain avg 25.86 | avg 63.22 | partial success rate % | pi0.5 baseline；任务长程且分解，指标非严格 binary success；真实评测 episode 数未统一披露，但附录分项多为 10/20 次 | `5_experiments.tex:66-68,82-112`；`X_appendix.tex:503-575`。 |
| Real-world Agilex-Aloha-2 | 每任务 100 trajectories；多任务联合训练；partial success rate | pi0.5 avg 48.60；w/o pretrain avg 26.60 | avg 59.30 | partial success rate % | 同上；部分任务 baseline 优于 Motus，如 Put Bread into Oven | `5_experiments.tex:66-68,82-112`；`X_appendix.tex:583-640`。 |
| World Model mode on real-world robot data | real-world robot data across AC-One / Agilex-Aloha-2 | 无直接策略 baseline | Avg FID 11.209, FVD 61.20865, SSIM 0.8661, LPIPS 0.063645, PSNR 25.0700 | generative quality | 只报告生成质量，无动作闭环 baseline | `X_appendix.tex:176-187`。 |
| IDM mode | RoboTwin 2.0 randomized, Agilex-Aloha-2；100 samples；action chunk 16；MSE | ResNet18+MLP 0.044；DINOv2+MLP 0.122 | 0.014 | action MSE | baseline 是专门训练的 IDM，但是否共享同等预训练/数据未披露 | `X_appendix.tex:193-209`。 |
| VLA mode vs Joint | RoboTwin 2.0 randomized | Motus VLA 83.90 | Joint 87.02 | avg success rate % | 同模型不同推理模式；公平 | `X_appendix.tex:213-228`。 |
| LIBERO-Long | standard LIBERO-Long protocol；10 tasks | pi0 85.2；GR00T-N1 90.6；UniVLA 94.0；OpenVLA-OFT 94.5；X-VLA 97.6 | 97.6 | avg success score | 论文未披露 Motus 训练数据量/episode/seed；跨论文比较公平性有限 | `X_appendix.tex:261-273`。 |
| VLABench | single Motus model fine-tuned on multiple tasks；3 tasks x 2 tracks | pi0.5 ID avg 0.43, cross-category avg 0.22 | ID avg 0.48, cross-category avg 0.25 | success rate | pi0.5 结果来自其官方实现，训练协议细节不足 | `X_appendix.tex:276-296`。 |

推理硬件/延迟/FPS：Motus 论文未报告延迟、FPS、显存曲线；HF/README 只给显存建议：预编码 T5 约 24GB，在线 T5 约 41GB（`INFERENCE.md:32-71`；HF Motus lines 129-134）。控制频率默认 30Hz、action chunk 48 steps（HF Motus lines 122-127），但未给真实闭环执行延迟。

## 7. 消融与局限

### 消融结果

| 消融 | clean | randomized | 解读 | 证据 |
|---|---:|---:|---|---|
| w/o Pretrain | 77.56 | 77.00 | 无 Stage 1/2 预训练仍较强，说明架构/目标机器人 SFT 本身贡献较大 | `X_appendix.tex:491`。 |
| Stage1 only | 82.26 | 81.86 | 只做 VGM/视觉动态预训练比无预训练提升约 4.7/4.86 | `X_appendix.tex:491`。 |
| Full Motus / Stage2 latent action pretrain | 88.66 | 87.02 | 相比 Stage1 only 再提升约 6.40/5.16，论文归因于 latent action 和大规模 unlabeled/labeled 混合 | `5_experiments.tex:180-207`；`X_appendix.tex:491`。 |
| VLA-only inference | 83.90 randomized | - | 低于 joint 87.02，说明联合预测未来视频和动作对策略有帮助 | `X_appendix.tex:213-228`。 |

### 主要性能来源判断

- 架构：MoT + Tri-model Joint Attention 解决三专家融合，论文明确列为核心创新（`4_methodology.tex:16-18`）。
- 数据/预训练：Stage1/Stage2 消融显示预训练贡献明显，尤其 Stage2 latent action pretraining（`X_appendix.tex:491`）。
- latent action：光流显式 motion representation + 90/10 弱监督对齐是论文用于跨 embodiment/无动作视频学习的关键（`4_methodology.tex:183-196`）。
- 推理策略：Joint mode 优于 VLA-only 约 3.12 points（87.02 vs 83.90），说明视频-动作联合生成也贡献性能（`X_appendix.tex:218-228`）。

### 局限

- 完整 Stage 1/2 数据不可复刻：论文只列数据集与 size，缺少下载清单、版本、过滤规则、采样比例、相机/Hz/分辨率。
- 训练成本高：论文给 Stage1 ~8000 GPU hours，Stage2 ~10000 GPU hours，Stage3 ~400 GPU hours（`X_appendix.tex:405`）。
- 配置不完全一致：论文附录 action chunk=48@30Hz；当前 `configs/robotwin.yaml` 计算得到 16；HF robotwin2 card 又写 48。工程复现应优先论文实验配置和 checkpoint card，代码样例需手动校正。
- 推理性能不足披露：论文/代码公开了 steps 与显存建议，但未给真实机器人闭环 latency/FPS。
- 私有数据依赖：In-house Data 2,000 与真实机器人数据未公开（`X_appendix.tex:384`；`5_experiments.tex:66`）。
- baseline 公平性有限：RoboTwin 协议中 40k fine-tune 较清楚，但 pi0.5/X-VLA 的额外预训练数据、输入模态与实现差异未完全列出。

## 8. 最小复现步骤

目标是复现“可公开落地”的 Stage 3 RoboTwin2 微调/评测，而不是复刻 Stage 1/2 全量预训练。

1. 克隆官方代码并记录 commit。

```bash
git clone https://github.com/thu-ml/Motus.git
cd Motus
git rev-parse HEAD
```

2. 建环境。README 指定 Python 3.10、torch 2.7.1/cu128、flash-attn、requirements（`README.md:78-92`）。

```bash
conda create -n motus python=3.10 -y
conda activate motus
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn --no-build-isolation
pip install -r requirements.txt
```

3. 下载权重。Stage 3 可直接用 `Motus_robotwin2` 推理；微调用 `Motus` 作为 base checkpoint；Wan VAE 需另下，HF Stage1 card 明确不包含 VAE（HF Stage1 lines 154-159）。

```bash
huggingface-cli download motus-robotics/Motus --local-dir ./pretrained_models/Motus
huggingface-cli download motus-robotics/Motus_robotwin2 --local-dir ./pretrained_models/Motus_robotwin2
huggingface-cli download motus-robotics/Motus_Wan2_2_5B_pretrain --local-dir ./pretrained_models/Motus_Wan2_2_5B_pretrain
huggingface-cli download Qwen/Qwen3-VL-2B-Instruct --local-dir ./pretrained_models/Qwen3-VL-2B-Instruct
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./pretrained_models/Wan2.2-TI2V-5B
```

4. 准备 RoboTwin2。用仓库转换器把 clean/randomized 数据转成 `qpos/ videos/ umt5_wan/` 结构（`DATA_FORMAT.md:9-29`）。论文实验用 50 tasks，每任务 50 clean + 500 randomized，共 27,500 demos（`5_experiments.tex:57`）。

5. 检查配置。将 `configs/robotwin.yaml` 的 checkpoint/path、dataset_dir 指向本地；若目标是对齐论文/HF checkpoint，设置 `num_video_frames=8`、`action_chunk_size=48` 对应 30Hz 动作 chunk。当前代码 action chunk 由 `num_video_frames * video_action_freq_ratio` 计算（`train.py:66`），因此需要把 `video_action_freq_ratio` 设为 6 才能得到 48。

6. 训练 Stage 3。官方脚本为 8 GPU torchrun + DeepSpeed ZeRO-1（`scripts/train.sh:17-25`）。论文实验 fine-tune 40k steps；当前 config max_steps 是 1,000,000，需要改成 40,000。

```bash
bash scripts/train.sh
```

7. 评测 RoboTwin2。官方入口：

```bash
cd inference/robotwin/Motus
bash eval.sh <task_name>
bash auto_eval.sh
```

8. 最小算力估计。推理：预编码 T5 约 24GB，不预编码约 41GB；训练建议 >80GB GPU。复现 Stage3 40k steps 推荐 8xA100/H100；论文 Stage3 约 400 GPU hours。Stage1/2 完整复刻约 18,000 GPU hours 且缺数据处理细节，不建议作为最小复现目标。

## 9. 未披露信息清单

| 类别 | 未披露项 |
|---|---|
| 数据 | 各数据集确切版本、下载 URL 清单、许可证组合、清洗/过滤规则、指令生成规则、重复样本处理、每数据集采样比例、小时/帧数/token 数。 |
| 传感器 | 各数据集相机数量、相机外参、分辨率、原始帧率、控制频率；论文只给 Motus 训练采样率 8@5Hz/48@30Hz。 |
| 训练 | Stage1/2 训练脚本、总 steps/epochs、warmup、scheduler、betas/epsilon、gradient clipping、dropout、EMA、混合精度细节、并行策略细节、GPU 型号/数量。 |
| 模型 | Wan/Qwen 在 Motus 中的完整层级参数表；Qwen hidden size/heads 等未由 Motus 论文列出。 |
| 推理 | 真实机器人闭环控制频率、延迟、FPS、cache 策略、显存峰值曲线、batch/多任务服务吞吐。 |
| 评测 | 随机种子、所有 real-world task 的 episode 数统一说明、baseline 是否使用完全相同 observation/action interface、LIBERO/VLABench fine-tuning 数据量。 |

## 10. 可复现性评级

**中。** Stage 3 RoboTwin2 的推理/微调路径、权重、代码和主要评测协议已经公开，足以做工程级下游复现；但论文最关键的 Stage 1/2 大规模预训练依赖未完全公开的数据处理与算力，且配置文件与论文附录在 action chunk 上存在不一致，无法严格复现论文完整训练。

## 11. 一手资料来源

- arXiv: <https://arxiv.org/abs/2512.13030>
- arXiv HTML: <https://arxiv.org/html/2512.13030v2>
- 官方主页: <https://motus-robotics.github.io/motus>
- 官方代码: <https://github.com/thu-ml/Motus>
- HF Stage 1 VGM: <https://huggingface.co/motus-robotics/Motus_Wan2_2_5B_pretrain>
- HF Stage 2 Motus: <https://huggingface.co/motus-robotics/Motus>
- HF Stage 3 RoboTwin2: <https://huggingface.co/motus-robotics/Motus_robotwin2>
- Qwen3-VL-2B-Instruct model card: <https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct>
- Wan2.2-TI2V-5B model card: <https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B>
