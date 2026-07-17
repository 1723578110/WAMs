# Fast-WAM 技术复现报告

## 1. 200 字以内核心结论

Fast-WAM 是 arXiv:2603.16666v2《Fast-WAM: Do World Action Models Need Test-time Future Imagination?》，作者 Tianyuan Yuan、Zibin Dong、Yicheng Liu、Hang Zhao，机构为 IIIS, Tsinghua University 与 Galaxea AI，发布日期为 2026-03-23（论文 p.1）。它属于 World Action Model，而不是纯 VLA：训练时同时做未来视频 latent 与连续动作 flow matching，推理时取消未来视频去噪，只用首帧视频分支 KV cache 辅助动作分支去噪。论文声称 6B 总参数、1B action expert、190 ms 延迟、RoboTwin 91.8%、LIBERO 97.6%（p.1, p.6-9）。开源复现可跑 LIBERO/RoboTwin，但真实毛巾折叠数据、部分训练 wall-clock/显存未披露，整体可复现性：中。

## 2. 研究对象确认

| 项 | 结论 | 证据 |
|---|---:|---|
| 准确标题 | Fast-WAM: Do World Action Models Need Test-time Future Imagination? | 论文 p.1；README 第 3 行 |
| arXiv | 2603.16666v2, cs.CV | 论文 p.1 |
| 版本/发布日期 | v2, 2026-03-23 | 论文 p.1 |
| 作者 | Tianyuan Yuan, Zibin Dong, Yicheng Liu, Hang Zhao | 论文 p.1 |
| 机构 | IIIS, Tsinghua University; Galaxea AI | 论文 p.1 |
| 项目主页 | https://yuantianyuan01.github.io/FastWAM/ | README 第 8-12 行 |
| 官方代码 | https://github.com/yuantianyuan01/FastWAM | README 第 1-14 行；本地 clone commit `45d8e145` |
| 官方权重 | https://huggingface.co/yuanty/fastwam | README 第 151-166 行 |
| 官方数据 | `yuanty/LIBERO-fastwam`, `yuanty/robotwin2.0-fastwam` | README 第 87-119 行 |
| 代码许可证 | MIT；第三方代码/文件保留各自许可证 | `LICENSE` 第 1-16 行 |
| 实际开放范围 | 训练/评测代码、LIBERO/RoboTwin 预处理数据、LIBERO/RoboTwin release checkpoint 与 stats；真实毛巾折叠数据未开放 | README 第 151-166、252-259 行；论文 p.7 |
| 同名风险 | 本报告仅指 arXiv:2603.16666 与 `yuantianyuan01/FastWAM`；Hugging Face/GitHub 上存在派生或非官方条目，未纳入证据 | 官方 README 链接互指；搜索核对 |

证据级别说明：表中“论文明确说明”优先用于实验结果和方法定义；“代码确认”优先用于真实复现配置、数据处理、训练/推理路径；“推测”仅用于论文描述但代码没有对应配置的真实任务；没有一手证据的字段统一写“未披露”。

## 3. 模型架构表

| 模块 | 参数/结构 | 来源与可信级别 |
|---|---|---|
| 范式 | WAM；训练保留 video co-training，推理跳过 future imagination，接口类似直接策略 `p(a1:H | o,l)` | 论文明确说明，p.2-5 |
| Backbone | Wan2.2-TI2V-5B 的 video DiT、T5 text encoder、video VAE | 论文 p.4, p.6；代码 `configs/model/fastwam.yaml` 第 2-3 行 |
| 总参数量 | 6B；Action expert 1B | 论文 p.6；MoT 日志会打印各 expert 参数，`src/fastwam/models/wan22/mot.py` 第 53-56 行 |
| Video DiT | patch `[1,2,2]`, in/out dim 48, hidden 3072, FFN 14336, text dim 4096, heads 24, head dim 128, layers 30, eps 1e-6 | 代码确认，`configs/model/fastwam.yaml` 第 12-33 行 |
| Action DiT | continuous action encoder `Linear(action_dim,1024)`, hidden 1024, FFN 4096, heads 24, head dim 128, layers 30, text dim 4096, freq dim 256 | 代码确认，`configs/model/fastwam.yaml` 第 35-45 行；`action_dit.py` 第 74-99 行 |
| MoT 交互 | video/action 两个 expert layer 数、heads、head dim 必须一致；混合 self-attention，expert 自己做 cross-attn 与 FFN | 代码确认，`fastwam.py` 第 140-150 行；`mot.py` 第 32-57、77-122 行 |
| 输入模态 | RGB 多相机拼接图、语言指令、proprio/state、连续动作监督 | 论文 p.4-6；代码 `configs/data/*.yaml` |
| 输出模态 | 推理输出动作 chunk；`infer_joint` 可输出视频但不是 Fast-WAM 默认评测路径 | 代码确认，`fastwam.py` 第 905-1048 行；`eval_libero_single.py` 第 413-419 行 |
| 视觉编码 | 多相机先拼接为单图，再进 Wan VAE；LIBERO 2 cam 横向拼接为 224x448，RoboTwin 3 cam 拼成 384x320 | 论文 p.6；代码 `robot_video_dataset.py` 第 153-197 行；配置见数据表 |
| 语言编码 | Wan 内置 T5；训练数据读取预计算 embedding cache，context len 128；评测加载 text encoder 在线编码 | 代码确认，`robot_video_dataset.py` 第 216-221、236-260 行；`sim_libero.yaml` 第 11-14 行 |
| 状态编码 | proprio 追加到 language context：`Linear(proprio_dim, text_dim)` | 代码确认，`fastwam.py` 第 57-61、986-991 行 |
| 动作表示 | 连续动作向量，不是离散 token；ActionDiT 线性映射到 hidden token；action_dim LIBERO=7、RoboTwin=14 | 代码确认，`action_dit.py` 第 74-98 行；`configs/data/*.yaml` |
| Action chunk | `h=32`；数据 `num_frames=33`，32 action，视频 9 帧 | 论文 p.6；`libero_2cam.yaml` 第 28-31 行；`robotwin.yaml` 第 24-27 行 |
| 生成方式 | continuous flow matching；训练目标 `noise - sample`；推理 Euler-like step `sample + output * delta` | 论文 p.5；`scheduler_continuous.py` 第 49-61、63-88 行 |
| 训练信息流 | future video tokens 与 action tokens 都带噪；action 只看 action branch 与 clean first-frame，不看 future video，防泄漏 | 论文 p.4-5；代码 attention mask `fastwam.py` 第 385-407 行 |
| 推理信息流 | 仅首帧 VAE latent 进 video branch，预填充 video KV cache；动作 latent 反复去噪并 attend cached video K/V | 代码确认，`fastwam.py` 第 993-1048 行；`mot.py` 第 257-341、343-430 行 |
| 推理步数/CFG | 10 denoising steps，CFG scale 1.0 | 论文 p.6；配置 `train.yaml` 第 24 行，`sim_libero.yaml` 第 30-35 行 |
| 延迟/硬件 | Fast-WAM 190 ms；Fast-WAM w/o video 180 ms；Joint 190 ms；IDM 810 ms；单 NVIDIA RTX 5090D V2 32GB | 论文 p.6、Figure 4 p.9 |

ASCII 架构图：

```text
Training
-------
language -> T5/context ------------------------------+
proprio  -> Linear(proprio_dim,4096) -> context -----+
                                                     v
multi-cam RGB -> concat -> Wan VAE -> latent frames: [f0 clean, f1..fh noisy]
                                      |
                                      v
                           Video DiT expert, 30L, h=3072
                                      |
action a1..a32 + noise -> Linear -> Action DiT expert, 30L, h=1024
                                      |
                  MoT shared attention mask:
                  - video attends video/first-frame as configured
                  - action attends action + first-frame tokens
                  - action cannot attend future-video tokens
                                      |
                   losses: L = L_action + lambda * L_video

Inference (Fast-WAM default)
----------------------------
current RGB -> concat -> Wan VAE -> f0 latent -> Video DiT once -> per-layer K/V cache
language/proprio context ----------------------------------------------+
random action latent a1..a32 ------------------------------------------+
                                                                       v
                         Action DiT 10 flow steps, attending video K/V cache
                                                                       |
                                                           denormalized action chunk
```

## 4. 训练数据表

| 数据集 | 来源/任务/embodiment | 规模 | 输入与频率 | action/state | 清洗与处理 | 公开性 |
|---|---|---:|---|---|---|---|
| LIBERO | 四套：Spatial/Object/Goal/Long，每套 10 tasks、500 demonstrations；模拟 manipulation | 论文实验协议：共 40 tasks、2000 eval trials；HF 数据卡列 5,152 rows、4.69GB | 2 RGB cams：`image`, `wrist_image`，raw 512x512，resize 224x224 后横向拼接为 224x448；`num_frames=33`, stride 1, `action_video_freq_ratio=4` 得到 9 video frames/32 actions；帧率/控制频率未披露 | action 7 = eef pose 6 + gripper 1；state 8 = eef pose 6 + gripper 2 | no-noops 预处理目录；ToTensor+Resize；action/state min-max norm；delta mask 前 6 维为 delta、gripper 非 delta；高层指令默认 drop prob 1.0 | 预处理数据公开：`yuanty/LIBERO-fastwam`；配置 `configs/data/libero_2cam.yaml` 第 3-67 行 |
| RoboTwin 2.0 | 50+ 双臂任务；clean + heavy randomization；bimanual | 论文：2,500 clean demos + 25,000 randomized demos，30k steps；HF 数据卡列 27,500 episodes、6,075,103 frames、79.1GB | 3 RGB cams：top/head、left wrist、right wrist，raw 480x640；resize/拼接成 384x320；`num_frames=33`, stride 1, ratio 4；帧率/控制频率未披露 | action 14；state 14 | z-score norm；1% val split；robotwin 拼接：top 256x320，上方；left/right 128x160 并排下方 | 预处理数据公开：`yuanty/robotwin2.0-fastwam`；配置 `configs/data/robotwin.yaml` 第 3-62 行 |
| Real-world towel folding | Galaxea R1 Lite 平台，毛巾折叠长时程真实任务 | 60 hours teleoperated demos；30k training steps | 相机数、分辨率、帧率、控制频率未披露 | action/state 维度未披露 | 清洗、增强、归一化未披露 | 未公开；论文 p.7 |

未发现论文披露多数据集训练采样比例；LIBERO/RoboTwin/真实任务按各 benchmark 分别训练。

## 5. 训练 Recipe 和超参数表

| 阶段/设置 | LIBERO | RoboTwin | 真实毛巾折叠 | 证据 |
|---|---:|---:|---:|---|
| 初始化 | Wan2.2-TI2V-5B video DiT/T5/VAE；ActionDiT 由 Wan2.2 DiT 插值生成 `ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt` | 同左 | 同左推测；论文未给真实任务配置文件 | 论文 p.6；README 第 74-82 行；`preprocess_action_dit_backbone.py` 第 139-190 行 |
| 训练模块 | 训练 `model.dit` 即 MoT；VAE/text encoder 冻结；若有 proprio encoder 则训练 | 同左 | 未披露 | `trainer.py` 第 82-94、287-295 行 |
| loss | `L = L_action + lambda_video * L_video`；代码默认 `lambda_action=1`, `lambda_video=1` | 同左 | 同左推测 | 论文 Eq.5-9 p.5；`fastwam.py` 第 539-568 行；`runtime.py` 第 156-157 行 |
| optimizer | AdamW betas=(0.9,0.95), epsilon 未披露 | 同左 | 未披露 | `trainer.py` 第 89-94 行 |
| LR/scheduler | 1e-4，cosine，eta_min=1e-6 | 1e-4，cosine，eta_min=1e-6 | 论文称 1e-4/cosine | 论文 p.6；`task/libero...yaml` 第 16-27 行；`trainer.py` 第 220-252 行 |
| warmup | 5% total train steps，LinearLR | 同左 | 未披露 | `trainer.py` 第 97-104、243-252 行 |
| weight decay | 0.01 | 0.01 | 论文称 0.01 | 论文 p.6；task yaml 第 26 行 |
| batch size | per process 16；论文未披露全局 batch；README 训练命令 8 GPU | per process 16；README 称使用 64 GPU 加速训练，但示例命令 8 GPU | 未披露 | task yaml 第 8-10 行；README 第 252-259 行 |
| grad accumulation | 1 | 1 | 未披露 | task yaml 第 25 行 |
| steps/epochs | 论文：20k steps；代码 release config：10 epochs, `max_steps=null` | 论文：30k steps；代码 release config：5 epochs, `max_steps=null` | 30k steps | 论文 p.6-7；task yaml 第 18-19 行 |
| precision | bf16 mixed precision | bf16 | 未披露 | `train.yaml` 第 26-30 行 |
| grad clipping | 1.0 | 1.0 | 论文称 1.0 | 论文 p.6；`trainer.py` 第 46-47、677-682 行 |
| EMA/dropout | 未披露/代码未见 EMA | 未披露/代码未见 EMA | 未披露 | 代码检索 |
| flow schedule | shift=5.0，1000 train timesteps；代码 `_phi(u)=shift*u/(1+(shift-1)u)`；论文写 logit-normal，代码实现更像 Wan shift schedule | 同左 | 未披露 | 论文 p.6；`configs/model/fastwam.yaml` 第 47-55 行；`scheduler_continuous.py` 第 17-37 行 |
| 并行策略 | Accelerate + DeepSpeed ZeRO-1，offload none；代码也提供 ZeRO-2 配置 | README 称 64 GPU 加速；配置同 ZeRO-1 | 未披露 | `accelerate_zero1_ds.yaml` 第 1-18 行；`ds_zero1_config.json` 第 1-18 行；README 第 259 行 |
| 训练时间/存储 | 未披露；数据约 4.69GB + Wan2.2/ckpt/cache | 未披露；数据约 79.1GB + Wan2.2/ckpt/cache | 未披露 | 论文/卡片/README |

## 6. Benchmark 性能表

| Benchmark | 协议 | 方法 | Embodied PT | 指标 | 结果 | 公平性备注 |
|---|---|---|---|---|---:|---|
| RoboTwin 2.0 | 50+ tasks；100 trials/task；clean 与 randomized；训练 2,500 clean + 25,000 randomized demos；30k steps | Fast-WAM | 否 | Avg success | 91.8 | 与 Fast-WAM variants 公平：同 backbone/tokenization/recipe；与 π0/π0.5/LingBot/Motus 不完全公平，后者多有额外 embodied pretraining |
| RoboTwin 2.0 | 同上 | Fast-WAM-Joint | 否 | Avg | 90.6 | 控制对照 |
| RoboTwin 2.0 | 同上 | Fast-WAM-IDM | 否 | Avg | 91.3 | 控制对照；推理慢 |
| RoboTwin 2.0 | 同上 | Fast-WAM w/o video co-train | 否 | Avg | 83.8 | 关键消融 |
| RoboTwin 2.0 | 同上 | LingBot-VA | 是 | Avg | 92.2 | 额外预训练，协议/数据可能不等价 |
| RoboTwin 2.0 | 同上 | Motus | 是 | Avg | 87.8 | 额外预训练 |
| RoboTwin 2.0 | 同上 | π0.5 | 是 | Avg | 79.8 | VLA baseline，输入/预训练不同 |
| RoboTwin 2.0 | 同上 | π0 | 是 | Avg | 62.2 | VLA baseline，输入/预训练不同 |
| LIBERO | 4 suites, 40 tasks；2000 trials with random seeds；每 suite 500 demos；20k steps | Fast-WAM | 否 | Avg success | 97.6 | 与 Fast-WAM variants 公平；与 pretrained baselines 不完全公平 |
| LIBERO | 同上 | Fast-WAM-Joint | 否 | Avg | 98.5 | 控制对照 |
| LIBERO | 同上 | Fast-WAM-IDM | 否 | Avg | 98.0 | 控制对照 |
| LIBERO | 同上 | Fast-WAM w/o video co-train | 否 | Avg | 93.5 | 关键消融 |
| LIBERO | 同上 | LingBot-VA | 是 | Avg | 98.5 | 额外预训练 |
| LIBERO | 同上 | Motus | 是 | Avg | 97.7 | 额外预训练 |
| Real-world towel folding | Galaxea R1 Lite；60h demos；30k steps；success + completion time + latency | Fast-WAM | 否 | latency | 190 ms | 论文 Figure 4；success/time 只以图呈现，未给表格精确数 |
| Real-world towel folding | 同上 | Fast-WAM-IDM | 否 | latency | 810 ms | 比 Fast-WAM 慢约 4.3x |
| Real-world towel folding | 同上 | Fast-WAM-Joint | 否 | latency | 190 ms | 图中与 Fast-WAM 相近 |
| Real-world towel folding | 同上 | w/o video co-train | 否 | success | 10% | 论文 p.9 明确文字说明 |

RoboTwin 表格来自论文 Table 1 p.7；LIBERO 表格来自 Table 2 p.8；真实任务延迟来自 Figure 4 p.9。代码默认评测配置：LIBERO 每 task `num_trials=50`、8 GPU manager；RoboTwin `eval_num_episodes=100`、`instruction_type=unseen`、`skip_get_obs_within_replan=true`（`sim_libero.yaml` 第 16-52 行，`sim_robotwin.yaml` 第 16-44 行）。

## 7. 消融与局限

| 设计 | 结果 | 结论 | 证据 |
|---|---:|---|---|
| 去掉测试时 future video imagination：Fast-WAM vs Joint/IDM | RoboTwin 91.8 vs 90.6/91.3；LIBERO 97.6 vs 98.5/98.0 | 显式未来想象不是主要收益来源 | 论文 Table 1 p.7, Table 2 p.8 |
| 去掉 video co-training | RoboTwin 83.8，LIBERO 93.5；真实任务 10% success | 性能主要来自训练时视频建模/世界表征，而非推理时视频生成 | 论文 p.8-9 |
| IDM 推理路径 | 810 ms vs Fast-WAM 190 ms | 先生成未来视频再动作显著增加延迟 | Figure 4 p.9 |
| Action branch hidden dim 缩小到 1024 | 1B action expert，总模型 6B | 架构节省来自 action expert 降维，但消融未单独报告 hidden dim 影响 | 论文 p.6；配置第 35-45 行 |

主要局限：

- 真实任务数据未公开，且真实任务 success/completion time 未给精确表格数字，只有图示与文字说明。
- 论文未披露训练 wall-clock、显存、完整 GPU 型号/数量；README 只说 LIBERO 8 GPU、RoboTwin 64 GPU 加速。
- 论文写 t 使用 logit-normal，代码 scheduler 为 Wan shift schedule `_phi(u)`；需要以实际代码为复现优先证据。
- Wan2.2 backbone、第三方 RoboTwin/LIBERO assets 受各自许可证/下载条件约束，MIT 只覆盖 FastWAM 仓库非第三方部分。
- 与 π0/π0.5/LingBot-VA/Motus 的比较存在额外预训练、输入/评测协议差异，最公平的是 Fast-WAM family 内部对照。

## 8. 最小复现步骤

1. 环境：Python 3.10，PyTorch 2.7.1+cu128，`pip install -e .`；见 README environment。
2. 准备 Wan2.2：设置 `DIFFSYNTH_MODEL_BASE_PATH=./checkpoints`，下载/缓存 `Wan-AI/Wan2.2-TI2V-5B` 和 tokenizer `Wan-AI/Wan2.1-T2V-1.3B`。
3. 生成 ActionDiT 初始化：
   ```bash
   python scripts/preprocess_action_dit_backbone.py \
     --model-config configs/model/fastwam.yaml \
     --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
     --device cuda --dtype bfloat16
   ```
4. 下载数据：LIBERO 解压到 `data/libero_mujoco3.3.2/*_no_noops_lerobot`；RoboTwin 解压到 `data/robotwin2.0/robotwin2.0`，并准备 `dataset_stats.json`。
5. 预计算文本 embedding：
   ```bash
   python scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4
   python scripts/precompute_text_embeds.py task=robotwin_uncond_3cam_384_1e-4
   ```
6. 训练：
   ```bash
   bash scripts/train_zero1.sh 8 task=libero_uncond_2cam224_1e-4
   bash scripts/train_zero1.sh 8 task=robotwin_uncond_3cam_384_1e-4
   ```
   论文目标是 LIBERO 20k steps、RoboTwin 30k steps；release 配置是 epochs 驱动，若数据量/卡数不同，需用 `max_steps` 覆盖到论文步数。
7. 评测 release checkpoint：
   ```bash
   python experiments/libero/run_libero_manager.py \
     task=libero_uncond_2cam224_1e-4 \
     ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
     EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
     MULTIRUN.num_gpus=8

   python experiments/robotwin/run_robotwin_manager.py \
     task=robotwin_uncond_3cam_384_1e-4 \
     ckpt=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
     EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
     MULTIRUN.num_gpus=8
   ```

算力/存储估算：最小下载需 Wan2.2-5B、FastWAM checkpoints、LIBERO 约 4.69GB 或 RoboTwin 约 79.1GB，再加文本 embedding cache 和训练 checkpoint；单卡可评测，训练按 README 推荐 LIBERO 8 GPU、RoboTwin 64 GPU。训练时间、峰值显存未披露。

## 9. 未披露信息清单

- 训练 wall-clock、GPU 型号、RoboTwin 64 GPU 的节点拓扑、峰值显存。
- optimizer epsilon。
- dropout、EMA 是否使用；代码未见 EMA，但论文未明确。
- 数据帧率、控制频率、episode 平均长度、token 数。
- LIBERO/RoboTwin 具体多任务采样比例；论文只说明数据混合规模。
- 真实毛巾折叠数据的相机、分辨率、动作/state 维度、清洗规则、公开地址。
- 真实任务 success rate/average completion time 的精确表格值。
- 模型/数据 Hugging Face 条目的独立 license 字段；代码仓库 MIT 不自动覆盖 Wan2.2 与第三方 assets。

## 10. 一手资料链接

- 论文：https://arxiv.org/abs/2603.16666
- PDF：https://arxiv.org/pdf/2603.16666
- 项目页：https://yuantianyuan01.github.io/FastWAM/
- 代码：https://github.com/yuantianyuan01/FastWAM
- 权重：https://huggingface.co/yuanty/fastwam
- LIBERO 数据：https://huggingface.co/datasets/yuanty/LIBERO-fastwam
- RoboTwin 数据：https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam
