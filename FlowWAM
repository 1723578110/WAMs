# FlowWAM 工程复现深度调研报告

> 调研日期：2026-07-17。证据范围严格限制为一手资料：arXiv 论文/API/源码包、官方项目页、官方 GitHub 代码、Hugging Face 模型卡/数据集卡/API。标注规则：“论文明确说明”优先于“代码确认”；未找到的一律写“未披露”，不按经验补全。

## 1. 200 字以内核心结论

FlowWAM 是一个 World Action Model（WAM），不是传统 VLA：它把 optical flow 编码成 RGB 视频流，和 RGB 视频在同一个 Wan2.2-TI2V-5B DiT 中双流联合建模；policy 模式生成 flow 再由 780M action expert 解码 14D 动作，world-model 模式则把目标 flow 作为条件生成未来 RGB。论文与代码公开了 RoboTwin/WorldArena 权重、RoboTwin 训练集和主要训练脚本，但 Stage 1 EgoDex 原始数据、总 step、训练时长、显存/延迟和部分评测细节未披露；整体可复现性：中。

## 2. 研究对象确认

| 项目 | 结论 | 来源 |
|---|---|---|
| 准确标题 | FlowWAM: Optical Flow as a Unified Action Representation for World Action Models | arXiv API `2607.13017v1`; 论文源码 `neurips_2026.tex:71` |
| 作者与机构 | Yixiang Chen, Peiyan Li, Yuan Xu, Qisen Ma, Jiabing Yang, Kai Wang, Jianhua Yang, Dong An, He Guan, Gaoteng Liu, Jianlou Si, Jun Huang, Jing Liu, Nianfeng Liu, Yan Huang, Liang Wang；NLPR/中科院自动化所、UCAS、FiveAges、MBZUAI、Alibaba Group | 论文源码 `neurips_2026.tex:83-108` |
| 版本/发布日期 | arXiv v1；published/updated 2026-07-14 17:57:12 UTC | arXiv API `work/arxiv_api.xml` |
| 官方入口 | 论文：https://arxiv.org/abs/2607.13017；项目页：https://flow-wam.github.io/；RoboTwin 代码：https://github.com/YixiangChen515/FlowWAM；WorldArena 代码：https://github.com/YixiangChen515/FlowWAM_WorldArena；模型：https://huggingface.co/YixiangChen/FlowWAM；RoboTwin 数据：https://huggingface.co/datasets/YixiangChen/FlowWAM_RoboTwin；WorldArena 数据：https://huggingface.co/datasets/YixiangChen/FlowWAM_WorldArena | README/HF 模型卡 |
| 开源许可证 | 代码仓与 HF 模型/数据卡均标注 Apache-2.0；SeedVR refiner 为第三方组件，WorldArena README 要求看 `inference/refiner/LICENSE_SeedVR` 和 `NOTICE` | `FlowWAM/LICENSE`; `FlowWAM_WorldArena/LICENSE`; HF model card; `FlowWAM_WorldArena/README.md:226-227` |
| 实际开放范围 | 开放 RoboTwin policy 权重 `flowwam_robotwin.safetensors`、action norm、WorldArena checkpoint `flowwam_worldarena_stage1.safetensors`；开放 FlowWAM_RoboTwin raw dataset；开放 FlowWAM_WorldArena 320/640 zip 数据；基础 Wan 和 SeedVR 需从各自 HF 仓下载 | HF model card “Files”; `FlowWAM/README.md:99-102`; `FlowWAM_WorldArena/README.md:78-84`; HF dataset API |
| 同名项目排查 | 检索到的 FlowWAM 结果均指向同一 arXiv/HF/GitHub/项目页；alphaXiv、Hugging Face Papers 等为转载/聚合，不纳入证据 | Web 检索 |

## 3. 核心方法与架构

### 3.1 范式与信息流

FlowWAM 属于 WAM/World Model 范式，并带有 policy action-decoding 头；论文没有把它定义为 VLA。核心问题是：预训练视频生成器需要“视频原生”的动作条件，而低维数值 action 与视频先验有模态鸿沟。FlowWAM 用 optical flow 作为统一动作表示：flow 可由视频无动作标签提取，也可作为动作条件驱动未来视频。

```text
Policy mode
I0 + instruction + qpos
   -> Wan VAE/T5 (frozen)
   -> Dual-stream DiT: RGB tokens <concat attention> Flow tokens
   -> generated RGB rollout + generated flow rollout
   -> Action Expert (30-layer AdaLN DiT, per-layer cross-attn to RGB+flow hidden states)
   -> 14D normalized action chunk -> denorm -> robot execution

World-model mode
I0 + instruction + target flow sequence
   -> VAE encodes RGB first frame and clean flow latents
   -> Dual-stream DiT denoises RGB while flow latents are held fixed
   -> generated future RGB video
```

### 3.2 模型架构表

| 模块 | 参数/结构 | 状态 | 来源与证据 |
|---|---:|---|---|
| Backbone | Wan2.2-TI2V-5B，约 5B image-to-video DiT；UMT5-XXL text encoder；Wan2.2 causal VAE | 论文明确说明；VAE/Text frozen，DiT 按阶段训练 | Appendix A.1；Table 3；`FlowWAM/README.md:78` |
| Wan2.2-TI2V-5B DiT config | patch `[1,2,2]`, `in_dim=48`, `dim=3072`, `ffn_dim=14336`, `text_dim=4096`, `out_dim=48`, `num_heads=24`, `num_layers=30` | 代码确认 | `diffsynth/models/wan_video_dit.py:704-715` |
| Dual-stream flow path | flow patch embedding + flow output head，均从 RGB 对应层 deep copy；learnable `stream_embed`；RGB/flow tokens concat self-attn 后再 split；RoPE 独立应用 | 论文+代码确认 | Appendix A.1；Method 3.2；`wan_video_dit_dual_stream.py:18-33` |
| VAE | 同一 frozen VAE 编码 RGB frame 与 flow RGB；VAE latent `z_dim` 实现默认 16 | 论文+代码确认 | Appendix A.1；`flow_action_train.py:252` |
| Text encoder | UMT5-XXL/T5 dim 4096，bf16 frozen | 论文+代码确认 | Table 3；`wan_video_text_encoder.py:211-217`; `training/train.sh:75` |
| Action expert | AdaLN diffusion transformer，约 780M；30 layers / hidden 1024 / heads 16 / FFN 4096；每层 action block self-attn + cross-attn video features + cross-attn T5 tokens | 论文明确说明，代码确认默认值 | Appendix A.1；Table 3；`flow_action_train.py:228-230`; `action_dit.py:220-268` |
| Proprio/state | normalized 14D qpos token，投影到 instruction embedding dim 并 append 到 T5 context；默认 `proprio_mode=text` | 论文+代码确认 | Appendix A.1；`training/train.sh:72,85`; `action_dit.py:13-14` |
| Action 表示 | 14D absolute qpos：left_arm 6 + left_gripper 1 + right_arm 6 + right_gripper 1；全局 z-score 归一化 | HF 数据卡 + 代码确认 | HF `FlowWAM_RoboTwin` card；`dataset_action_robotwin.py:163-191,144-152` |
| Action chunk | 论文 Stage 2：9 pixel frames -> 32-step action chunk，pixel frame temporal stride 4；代码默认 `NUM_FRAMES=33`（1 anchor + 32）、`NUM_VIDEO_FRAMES=9`、`VISUAL_STRIDE=4` | 论文明确说明 + 代码确认 | Appendix A.2/Table 3；`training/train.sh:51-53` |
| Flow 表示 | HSV/RGB encoding；direction->hue, magnitude->saturation, value=1/255；25 px magnitude cap，0.5 px threshold | 论文明确说明；代码有 codec | Method Eq.1；Appendix A.3/Table 3；`reversible_flow_codec.py:59-153` |
| 训练目标 | Video latent flow matching：`L_video=(1-lambda_f)L_RGB+lambda_f L_flow`；labeled data 加 `lambda_a L_action`；flow timestep shared，action timestep independent | 论文明确说明；代码确认 MSE/flow scheduler | Method Eq.3-5；`flow_action_train.py:671-849` |
| 推理 steps | RoboTwin eval：video/action denoising steps 25/50；execute-and-replan window 25（论文 Table 3）。代码公开 server 默认 action chunk 49，client 用 anchor+future chunk；WorldArena inference 默认 50 steps/121 frames | 论文+代码确认 | Table 3；`flow_action_server.py:486`; `deploy_policy.yml:17`; `world_model_inference.py:437-438,505` |
| 缓存/显存/延迟 | 支持 latent cache/precompute、gradient checkpointing；具体延迟、FPS、显存未披露 | 代码确认/未披露 | `training/precompute.sh`; `training/train.sh:178`; 论文未披露 |

## 4. 训练数据表

| 数据集 | 用途 | 规模 | 模态/分辨率/频率 | 动作/状态 | 处理 | 公开性 | 来源 |
|---|---|---:|---|---|---|---|---|
| EgoDex | Stage 1 action-free motion pretraining | 论文未披露 hours/episodes/frames/token | 30 fps 视频 subsample 到 15 fps；resize 320x256；clip buckets `{17,33,49,65,81}` | 无 robot action | RAFT 提取相邻帧 optical flow；无额外 task-level filtering | FlowWAM 未开放其处理后 EgoDex 数据；EgoDex 本身为外部数据 | Appendix A.2/A.3/Table 3 |
| RoboTwin 2.0 aloha-agilex | Stage 2 policy 联合训练 | 50 tasks；Clean 50 demos/task=2,500；Random 500 demos/task=25,000；合计 27,500 episodes | head/left/right camera RGB；每相机 320x256；T-shape tile 320x384；9 pixel frames span 32 actions | 14D absolute qpos；joint_action vector | head flow 来自 robot-only replay + RAFT；wrist-flow placeholder；action 全局 z-score；instruction 随机取 `seen` | HF `YixiangChen/FlowWAM_RoboTwin` 开放 raw dataset | Appendix A.3-A.5/Table 3；HF dataset card；`dataset_action_robotwin.py` |
| FlowWAM_WorldArena RoboTwin episodes | WorldArena 训练/复现仓使用 | HF API：50 task zip × 2 resolution roots（320/640），`size_categories:10K<n<100K`，存储约 128.83GB；README 说明 single clean variant/task | 640 high-res supervision+robot-only；320 low-res conditioning；训练脚本用 head_camera 640x480，flow 320x240->upscale | 读取 action HDF5 生成 robot-only flow；不训练 action expert | RAFT flow，mask robot，max_stride=3，max_rollouts=2，121-frame chunks | HF `YixiangChen/FlowWAM_WorldArena` 开放 zip 文件；无 README card，但官方仓 README 与 HF API 可确认 | `FlowWAM_WorldArena/README.md:128-179`; HF API; `world_model_train.sh:78-109`; `dataset.py:1-12` |
| Real-world teleop demos | 真实机器人评测训练 | 7 tasks；每 task 100 demonstrations；10 eval trials/task | 单臂 Franka + 双臂 ARX；相机/分辨率/频率未披露 | 未披露 | 未披露 | 未公开 | Section 4.3；Appendix real-world details |

## 5. Training Recipe 和超参数表

| 阶段 | 配置 | 值 | 来源 |
|---|---|---|---|
| Stage 1 motion pretraining | 初始化 | Wan2.2-TI2V-5B；flow stream from RGB patch/head copy | Appendix A.1-A.2/Table 3 |
| Stage 1 | 数据 | EgoDex，action-unlabeled；15 fps，320x256，buckets `{17,33,49,65,81}` | Appendix A.3/Table 3 |
| Stage 1 | 训练模块 | dual-stream DiT；VAE/Text frozen；无 action expert | Appendix A.1-A.2 |
| Stage 1 | loss | `L_video` flow matching；`lambda_f=0.1`（Table 3 全局列出） | Method Eq.3；Table 3 |
| Stage 1 | optimizer | AdamW, weight decay `1e-2`; betas/eps 未在论文 Table 3 披露 | Table 3；betas/eps 未披露 |
| Stage 1 | LR/batch | LR `5e-5`; per-GPU batch `1`; 32 NVIDIA H100 across 4 nodes | Appendix Table 3 |
| Stage 1 | steps/epochs/time | 未披露 | 论文/代码未披露 |
| Stage 2 policy | 初始化 | Stage 1 checkpoint + fresh action expert（论文）；公开脚本可设置 `RESUME_CHECKPOINT`，action expert 默认 fresh | Appendix A.2；`training/train.sh:96-100` |
| Stage 2 | 数据 | RoboTwin 50 tasks；Clean 50 + Random 500 demos/task；head/left/right | Appendix A.3-A.5；HF card |
| Stage 2 | 训练模块 | full DiT, flow stream, action expert；VAE/Text frozen | Appendix A.2；`training/train.sh:69` |
| Stage 2 | loss | `L=L_video+lambda_a L_action`; `lambda_f=0.1`; `lambda_a=1.0`; motion boost `alpha=2.0`; cond noise prob `0.5` | Method Eq.2-5；Table 3；`training/train.sh:70-81` |
| Stage 2 | optimizer/scheduler | AdamW betas `[0.9,0.95]`, weight decay from parser/default `1e-2`；linear warmup + cosine；grad clip 1.0 | `flow_action_train.py:908-927,1011,1075`; `training/train.sh:89-90` |
| Stage 2 | LR/batch/epochs | LR `1e-4`; per-GPU batch `16`; default epochs `5`; grad accumulation `1`; warmup `1000` | Table 3；`training/train.sh:65-68,89-90` |
| Stage 2 | precision/parallel | bf16 pipeline; Accelerate/DDP; gradient checkpointing; 32 H100 across 4 nodes in paper | Table 3；`flow_action_train.py:242`; `training/train.sh:178` |
| Stage 2 | diffusion/flow timesteps | video scheduler samples random train timestep from 1000; action FlowMatchScheduler set 1000 training timesteps; action SNR shift 5.0 | `flow_action_train.py:312-315,671-674,751-754`; `training/train.sh:74` |
| WorldArena public training script | 用途 | 单独 world-model 仓复现；不是论文 Stage 1 EgoDex recipe | `FlowWAM_WorldArena` |
| WorldArena script | 配置 | Wan2.2-TI2V-5B, 121 frames, 640x480, LR `5e-5`, epochs `500`, batch `1`, max_stride `3`, max_rollouts `2`, flow max magnitude `20`, grad clip 1.0, ConstantLR | `world_model_train.sh:89-109`; `world_model_train.py:51-54,108` |

## 6. Benchmark 性能表

| Benchmark/任务 | 协议 | FlowWAM 结果 | Baseline 对比 | 公平性备注 | 来源 |
|---|---|---:|---|---|---|
| RoboTwin 2.0 policy | 50 bimanual tasks；Clean/Random；训练 50 clean + 500 random demos/task；100 rollouts/task | w/ PT: Clean `92.94%`, Random `92.14%`; w/o PT: `82.40%/80.80%` | π0.5 `42.98/43.84`; X-VLA `72.88/72.84`; Motus `88.66/87.02`; GigaWorld `86.36/85.04`; X-WAM `89.76/90.68`; Fast-WAM `91.88/91.78` | 论文称 WAM baselines 按各自大规模预训练协议评测；仍存在不同预训练数据/协议差异，非严格同数据公平比较 | Section 4.1；Table 1/Table 4 |
| WorldArena | 121-frame rollouts at 24 fps；initial frame + instruction + recorded joint-action trajectory；报告 6 representative metrics + EWMScore，附录 16 metrics | EWMScore `63.71`; JEPA `97.14`; Subj `82.46`; Bg `89.97`; Traj Acc `64.26`; Depth `98.97`; Sem Align `89.93` | ABot-PhysWorld text EWM `62.63`; GigaWorld-1 `62.34`; Ctrl-World `59.98`; IRASim `56.15` | 官方远程评测；baseline 输入条件不同（Text/Action/Img-Act/Flow），因此可比较 benchmark 分数，但不完全同输入公平 | Section 4.2；Table 2/Table 5 |
| Real-world manipulation | 7 tasks；Franka single-arm 4 tasks + ARX dual-arm 3 tasks；100 teleop demos/task；10 randomized trials/task | Average `75.7%` | π0.5 `61.4%`; Motus `57.1%` | 硬件/相机/详细训练超参未披露，外部复现实验难度高 | Section 4.3；Figure 3 |
| 延迟/FPS/显存/硬件推理 | 未披露；公开推理脚本为 server-client，多 GPU 可起多个 server | 未披露 | 未披露 | 只可由代码本地测量，论文无官方值 | `inference/start_server.sh`; Table 3 只给 denoising steps |

## 7. 消融与局限

| 消融 | 结果 | 解释 | 来源 |
|---|---:|---|---|
| Policy: numerical actions | `69.8%` success | 低维动作无法贴合视频生成器视觉先验 | Figure 4(a) |
| Policy: raw `(u,v)` flow | `72.3%` | 破坏预训练 VAE/DiT 期望的 RGB 格式 | Figure 4(a) |
| Policy: w/o flow-loss reweighting | `83.9%` | 静态背景主导 flow loss | Figure 4(a); Eq.4 |
| Policy: w/o stochastic AE conditioning | `82.1%` | action expert 对 inference 生成 latent 残差不鲁棒 | Figure 4(a); Eq.2 |
| Policy: full | `89.8%` | 注意：消融值与主表 92.94 不同，可能为验证设置 | Figure 4(a) |
| World: text only | EWM `49.31` | motion under-specified | Figure 4(b) |
| World: numerical actions | `54.18` | action outside visual space | Figure 4(b) |
| World: flow actions `(u,v)` | `56.72` | 有稠密位移但非视频原生 RGB | Figure 4(b) |
| World: image masks | `57.84` | spatial cue 静态，缺少跨帧方向/幅值 | Figure 4(b) |
| World: full RGB flow | `65.23` | flow 同时提供 per-pixel motion 和 RGB-compatible 表示 | Figure 4(b); footnote: custom validation split |
| Flow quality vs success | Pearson `r=-0.81` | flow error 越低成功率越高，支持 decodability | Figure 4(c) |

主要提升来源：架构/表示（RGB optical flow 双流）是最大来源；action-unlabeled EgoDex 预训练进一步放大，尤其 Random setting。限制：依赖 Wan2.2-TI2V-5B、RAFT、SAPIEN/RoboTwin、H100 级训练资源；EgoDex 预处理数据与完整训练时长未公开；真实机器人细节不足；WorldArena ablation 用 custom validation split，和官方表不可直接等同。

## 8. 最小复现步骤

1. 克隆官方仓库：`git clone https://github.com/YixiangChen515/FlowWAM.git`；WorldArena 另克隆 `FlowWAM_WorldArena`（README）。
2. 下载基础模型：`Wan-AI/Wan2.2-TI2V-5B`，另按 README 下载 `Wan2.1-T2V-1.3B` tokenizer/T5 相关文件（`FlowWAM/README.md:84-87`）。
3. 下载公开权重：`hf download YixiangChen/FlowWAM flowwam_robotwin.safetensors flowwam_robotwin_action_norm_stats.npz`；WorldArena 用 `flowwam_worldarena_stage1.safetensors`（HF model card）。
4. 下载 RoboTwin 数据：`hf download YixiangChen/FlowWAM_RoboTwin --repo-type dataset`；解压到 `DATASET_BASE_PATH`。全量约 27,500 episodes，HF card 标注 10K<n<100K。
5. 可选预计算 latent：运行 `training/precompute.sh`，减少在线 RAFT/VAE 成本；代码支持 cache mode（`training/train.sh:101-109,180`）。
6. Stage 2 policy 训练：设置 `DATASET_BASE_PATH`，运行 `bash training/train.sh`。默认：LR `1e-4`、batch/GPU `16`、5 epochs、flow loss `0.1`、action loss `1.0`、gradient checkpointing。
7. RoboTwin 评测：启动 inference server：`CHECKPOINT=... ACTION_NORM_PATH=... bash inference/start_server.sh`；在 RoboTwin 环境用 `inference/robotwin_policy/eval.sh` 跑 `demo_clean`/`demo_randomized`。
8. WorldArena 复现：按 `FlowWAM_WorldArena/README.md` 下载 640/320 数据、Wan/SeedVR/embodiment；运行 `inference/world_model_inference.sh <test_dataset_dir>`。
9. 算力估算：论文训练为 32×H100/4 nodes；公开脚本可单/多节点运行，但若全量从头训练，至少需要多 GPU 大显存。精确训练时间、存储除 WorldArena HF API 约 128.83GB 外未披露。

## 9. 未披露信息清单

- EgoDex 在 FlowWAM 中实际使用的 hours、clip 数、episode 数、token/frame 总量：未披露。
- Stage 1/Stage 2 总 optimizer steps、wall-clock training time、随机种子、EMA、dropout、optimizer epsilon：未披露/未在主训练脚本显式确认。
- FSDP/DeepSpeed/ZeRO：论文未披露；公开代码使用 Accelerate/DDP，未确认作者实验是否使用其他并行。
- 推理延迟、FPS、显存、控制频率的实测官方值：未披露。
- 真实机器人相机数量/分辨率/帧率/控制频率、action/state 精确定义与公开数据：未披露。
- baseline 是否完全同训练数据：论文只说明 WAM baseline 使用各自预训练协议；细粒度数据公平性未披露。
- WorldArena 官方测试集 episode 数、随机种子、远程评测可重复性细节：论文未披露。

## 10. 一手资料链接

- Paper/API: https://arxiv.org/abs/2607.13017
- Official project: https://flow-wam.github.io/
- RoboTwin code: https://github.com/YixiangChen515/FlowWAM
- WorldArena code: https://github.com/YixiangChen515/FlowWAM_WorldArena
- Model card: https://huggingface.co/YixiangChen/FlowWAM
- RoboTwin dataset card: https://huggingface.co/datasets/YixiangChen/FlowWAM_RoboTwin
- WorldArena dataset repo: https://huggingface.co/datasets/YixiangChen/FlowWAM_WorldArena

