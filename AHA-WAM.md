# AHA-WAM

## 1. 核心结论（200字以内）

AHA-WAM 是一个面向机器人操作的 World-Action Model（WAM），不是传统 VLA：它把低频 Video-DiT 作为世界规划器、把高频 Action-DiT 作为闭环执行器，并用 rolling K/V memory 与 Observation-Guided Video-Context Routing（OVCR）复用长时域视频上下文。论文报告 RoboTwin 2.0 平均成功率 92.80%、真实机器人 4 任务 78.33%、单 RTX 5090D 10-step 推理 24.17 Hz（论文 P1、P9 Table 1、P10 Table 3）。工程复现中，RoboTwin checkpoint、Flash checkpoint、代码和评测管线已公开；真实机器人 checkpoint 与真实任务数据未公开，训练硬件/时间未披露，且论文 Table 4 的 LR=1e-4 与 released checkpoint 模型卡/任务配置 LR=5e-5 不一致。综合可复现性：中。

## 0. 研究对象确认

| 项目 | 结论 | 证据 |
|---|---|---|
| 准确标题 | **AHA-WAM: Asynchronous Horizon-Adaptive World-Action Modeling with Observation-Guided Context Routing** | 论文 P1；README citation `README.md:531-536` |
| 作者 | Jisong Cai, Long Ling, Shiwei Chu, Zhongshan Liu, Jiayue Kang, Zhixuan Liang, Wenjie Xu, Yinan Mao, Weinan Zhang, Xiaokang Yang, Ru Ying, Ran Zheng, Yao Mu | 论文 P1；README `README.md:531-536` |
| 机构 | Shanghai Jiao Tong University；Shanghai AI Laboratory；Baidu AI Cloud；The University of Hong Kong | 论文 P1 |
| 版本/发布日期 | arXiv:2606.09811v1，2026-06-08 | 论文 P1；[arXiv](https://arxiv.org/abs/2606.09811) |
| 官方论文 | [arXiv abs](https://arxiv.org/abs/2606.09811)，[PDF](https://arxiv.org/pdf/2606.09811) | 论文 P1 |
| 项目主页 | [https://serene-sivy.github.io/aha-wam/](https://serene-sivy.github.io/aha-wam/) | 论文 P1 |
| 代码仓库 | [https://github.com/serene-sivy/AHA-WAM](https://github.com/serene-sivy/AHA-WAM)，本次检查 commit `69076f6206518554477be5639deb016eee03a9d6`（2026-07-06） | `git rev-parse HEAD`；README |
| 模型权重 | [SereneC/AHA-WAM-RoboTwin2.0](https://huggingface.co/SereneC/AHA-WAM-RoboTwin2.0)：`robotwin_ahawam.pt`、`robotwin_ahawam-flash.pt`、`dataset_stats.json` | 模型卡 `work/hf_modelcard.md:35-48`；HF API files |
| 数据集入口 | AHA-WAM README 指向 LeRobot 格式 RoboTwin2.0 预处理数据：[yuanty/robotwin2.0-fastwam](https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam)；也支持自行转换上游 RoboTwin | README `README.md:240-304`；数据集卡 `work/hf_datasetcard.md:13-40` |
| 开源许可证 | AHA-WAM 仓库 MIT；第三方代码/资产保留各自许可证；HF checkpoint card 标 MIT | `LICENSE:1-16`；README `README.md:550-553`；模型卡 `work/hf_modelcard.md:1-4` |
| 实际开放范围 | 已开放训练代码、RoboTwin 评测、ODE distillation、真实机器人部署示例、RoboTwin AHA-WAM/AHA-WAM-Flash checkpoint；真实机器人预训练 checkpoint 未开放 | README `README.md:68-74` |
| 同名排查 | 精确检索标题和 “AHA-WAM” 未发现另一个同名开源模型/论文；相关结果均回指官方 arXiv、项目页、GitHub、HF。 | 检索查询：`"AHA-WAM" "Asynchronous Horizon-Adaptive..."` |

## 2. 模型架构表

### 范式与信息流

- **范式**：论文明确称为 World-Action Model（WAM），基于 dual Diffusion Transformer / flow-matching；不是以大 VLM 为主体的 VLA（论文 P1-P3）。
- **解决问题**：既有 WAM 把视频世界预测和动作执行绑定到同一短时域/同一频率，视频分支被迫建模近邻帧冗余变化，闭环动作推理也慢。AHA-WAM 将慢速长时域世界规划与快速短 chunk 动作执行解耦（论文 P2-P3）。
- **训练 vs 推理差异**：训练时 video branch 预测未来 video latents、action branch 预测 action flow，并 joint 训练；推理时 Video-DiT 主要做异步 prefill/rolling K/V，上下文被 Action-DiT 查询，不在每个动作更新重新跑完整 Video-DiT（论文 P5-P8；代码 `src/ahawam/models/wan22/ahawam.py:372-490`, `src/ahawam/models/wan22/ahawam_chunk_base.py:1235-1379`）。

### 模块参数

| 模块 | 参数/结构 | 数值 | 类型 | 证据 |
|---|---:|---:|---|---|
| 总模型 | instantiated size | 约 7.23B | 论文明确说明 | 论文 P8、P15 Table 4 |
| Video-DiT planner | 参数量 | 4.99B | 论文明确说明 | 论文 P8、P15 Table 4 |
| Video-DiT planner | backbone | Wan2.2-TI2V-5B video expert | 论文/代码确认 | 论文 P15 Table 4；`configs/model/ahawam.yaml:1-3` |
| Video-DiT planner | hidden / FFN | 3072 / 14336 | 代码确认 | `configs/model/ahawam.yaml:27-28` |
| Video-DiT planner | layers / heads / head dim | 30 / 24 / 128 | 论文/代码确认 | 论文 P15 Table 4；`configs/model/ahawam.yaml:32-34` |
| Video-DiT planner | latent in/out dim, patch | 48 / 48, patch `[1,2,2]` | 代码确认 | `configs/model/ahawam.yaml:25-31` |
| Video-DiT planner | history frames / video-action ratio / RoPE stride | 6 / 8 / 8 | 论文/代码确认 | 论文 P15 Table 4；`configs/task/robotwin_ahawam.yaml:17-19` |
| Action-DiT executor | 参数量 | 1.02B | 论文明确说明 | 论文 P8、P15 |
| Action-DiT executor | hidden / FFN | 1024 / 4096 | 论文/代码确认 | 论文 P8；`configs/model/ahawam.yaml:48-49` |
| Action-DiT executor | layers / heads / head dim | 30 / 24 / 128 | 代码确认 | `configs/model/ahawam.yaml:50-52` |
| Action-DiT executor | action horizon / chunk size / chunks | 64 / 16 / 4 | 论文/代码确认 | 论文 P15 Table 4；`configs/model/ahawam.yaml:12-13` |
| Action-DiT executor | action/state dim | 14 / 14 | 论文/代码/HF 确认 | 论文 P15 Table 4；`configs/data/robotwin.yaml:16-23`；模型卡 `work/hf_modelcard.md:56-58` |
| Action-DiT executor | 生成方式 | flow denoising；chunkwise teacher forcing；inference 顺序生成 4 个 action chunks | 论文/代码确认 | 论文 P9、P15；`configs/model/ahawam.yaml:55-58`；`src/ahawam/models/wan22/ahawam_chunk_base.py:1235-1379` |
| OVCR + memory/context routing | 参数量 | 1.22B | 论文明确说明 | 论文 P8、P15 |
| OVCR | queries per chunk | 32 | 论文/代码确认 | 论文 P15 Table 4；`configs/model/ahawam.yaml:17` |
| OVCR | granularity | per action chunk, per transformer layer | 论文明确说明 | 论文 P15 Table 4 |
| Text context | text dim / context length | 4096 / 128 | 代码确认 | `configs/model/ahawam.yaml:30,53`；`configs/model/ahawam.yaml:4` |
| Scheduler | train timesteps / shift | 1000 / 5.0 | 论文/代码确认 | 论文 P15 Table 4；`configs/model/ahawam.yaml:61-69` |

### 模态与编码

| 模态 | 编码方式 | 证据等级 | 证据 |
|---|---|---|---|
| 视觉输入 | RoboTwin 3 cameras：`cam_high`, `cam_left_wrist`, `cam_right_wrist`；原始 480×640，单相机预处理 240×320，RoboTwin 拼接后最终 video_size 384×320；VAE 编码为 video latents | 代码确认 | `configs/data/robotwin.yaml:6-15,27`；`src/ahawam/datasets/lerobot/robot_video_dataset.py:316-340,354-357` |
| 语言指令 | 默认 prompt 模板包裹 task；T5/UMT5 context cache，长度 128，mask 后置零；在线编码要求 load_text_encoder 或 precomputed context | 代码确认 | `src/ahawam/datasets/lerobot/robot_video_dataset.py:22,584-594,632-649`；`src/ahawam/models/wan22/base_wam.py:300-314` |
| 状态/proprio | 14-D state，经 `nn.Linear(14, text_dim)` 作为 proprio token 追加到 context | 代码确认 | `configs/data/robotwin.yaml:20-23,39-40`；`src/ahawam/models/wan22/base_wam.py:139-144,316-342` |
| 动作 | 14-D action，经 ActionDiT `nn.Linear(action_dim, hidden_dim)` 编码；输出 head 还原到 14-D | 代码确认 | `src/ahawam/models/wan22/action_dit.py:97-123` |
| 未来表示 | 训练时 video branch 对未来 video latents 做 flow matching；action branch 对 action tokens 做 flow matching；推理复用 Video-DiT K/V cache | 论文/代码确认 | 论文 P5-P6、P16；`src/ahawam/models/wan22/ahawam_chunk_base.py:920-1021,1235-1379` |

### ASCII 架构图

```text
Inputs at time t
  RGB obs (3 cams) + instruction + proprio(14)
        │
        ├─ Image preprocessing: 3 views -> RoboTwin mosaic 384x320 -> Wan VAE latents
        │
        ├─ Text: task -> prompt -> cached UMT5/T5 context (L=128, D=4096)
        │
        └─ Proprio: 14-D -> linear token -> context

Slow stream: Video-DiT world planner (Wan2.2, 30L, hidden 3072)
        │  history=6, video/action ratio=8, causal video mask
        ├─ rolling layerwise video K/V memory
        └─ planner context for future scene evolution
                │
                ▼
OVCR: Observation-Guided Video-Context Routing
  latest obs -> visual context -> 32 queries/chunk -> edit/query causal video K/V
                │
                ▼
Fast stream: Action-DiT executor (30L, hidden 1024)
  64-action horizon = 4 chunks x 16 actions
  flow denoising, default 10 steps, CFG=1.0
                │
                ▼
Output: current executable 16-step action chunk (14-D per step)
```

## 3. 训练数据表

| 数据集/来源 | 用途 | 任务/embodiment | 规模 | 模态与频率 | action/state | 预处理/增强/归一化 | 公开性 | 证据 |
|---|---|---|---|---|---|---|---|---|
| RoboTwin 2.0 simulation | 主训练与 RoboTwin benchmark | 50 dual-arm manipulation tasks；AgileX embodiment | 论文：每任务 50 clean demos + 500 randomized demos，共 2,500 clean + 25,000 randomized；每任务 clean/random 各 100 eval trials | 3 cameras；65-frame trajectories；`action_video_freq_ratio=8`；预处理数据卡：FPS=50，27,500 episodes，6,075,103 frames | 14-D action，14-D state/proprio | 3 camera resize/拼接到 384×320；z-score action/state normalization；prompt cache；val split 0.01；多相机 `robotwin` 拼接 | 上游 RoboTwin/预处理数据公开；AHA-WAM README 指向 Fast-WAM 预处理 LeRobot release | 论文 P9、P16；`configs/data/robotwin.yaml:6-61`；README `README.md:240-304`；数据集卡 `work/hf_datasetcard.md:32-40` |
| Selected RoboCOIN subset | 真实机器人前置预训练，仅用于 real-world 实验中 Fast-WAM/AHA-WAM 稳定部署比较 | 论文未给完整任务列表；用于 bimanual AgileX Piper real-world transfer | 24,600 trajectories，约 165 hours | 未披露相机数量/分辨率/频率；真实实验 setup 使用 ego-view RGB + proprio + language | 未披露；AHA-WAM real-world policy 输入 proprio + RGB + language | 未披露清洗/采样/归一化；论文只说明二者使用同一 subset | 未披露是否公开，AHA-WAM 仓库未发布该 checkpoint | 论文 P10 |
| Real-world task-specific demonstrations | 真实机器人 4 任务 finetune/eval | AgileX Piper；Fold Towel, Organize Desktop, Prepare Soy Milk, Store Plate | 每任务约 120 episodes；每任务/模型 30 independent trials eval | ego-view RGB，proprio，language；控制频率与采样细节未披露 | 未披露；按代码/模型 RoboTwin 配置为 14-D 不可直接假设 real-world 一致 | 未披露；评分为 0-3 subtask progress，成功=score 3 | 数据与 checkpoint 未公开（README only notes real-world checkpoints todo） | 论文 P10-P11、P16-P17 Table 6；README `README.md:68-74` |

未披露项：RoboTwin 原始小时数、每条 trajectory 平均长度、训练 token 数、多数据集采样概率、真实机器人数据的相机分辨率/帧率/控制频率、RoboCOIN subset 抽样规则、真实任务数据清洗策略。

## 4. 训练 Recipe 和超参数表

| 阶段 | 项目 | 配置 | 证据等级 | 证据 |
|---|---|---|---|---|
| AHA-WAM RoboTwin | 初始化 checkpoint | Video-DiT/Wan VAE/text encoder/tokenizer 来自 Wan2.2/Wan converted assets；ActionDiT backbone 由 Wan video DiT 预处理生成 | README/代码确认 | README `README.md:127-190`；`configs/model/ahawam.yaml:1-10` |
| AHA-WAM RoboTwin | 训练模块 | 默认冻结非 DiT/非附加模块；训练 `model.dit`（MoT 内 video/action 专家）和额外模块（proprio/OVCR 等）；`freeze_video_dit=false` | 代码确认 | `src/ahawam/trainer.py:113-143,688-718`；`configs/train.yaml:31` |
| AHA-WAM RoboTwin | loss | video loss + action loss，论文称 equal weight；代码 total=`lambda_video*loss_video + lambda_action*loss_action`，默认 lambda_video=1、lambda_action=1；OVCR action_prior lambda=1 | 论文/代码确认 | 论文 P16；`src/ahawam/models/wan22/ahawam_chunk_base.py:920-1021`；`configs/model/ahawam.yaml:71-73` |
| AHA-WAM RoboTwin | optimizer | AdamW；代码 betas=(0.9,0.95)；eps 未显式传入 | 论文/代码确认 | 论文 P9、P15 Table 4；`src/ahawam/trainer.py:149-154` |
| AHA-WAM RoboTwin | LR | **冲突**：论文 Table 4 写 1e-4；released checkpoint 模型卡与 task config 写 5e-5 | 论文/代码/HF 不一致 | 论文 P15 Table 4；模型卡 `work/hf_modelcard.md:207-225`；`configs/task/robotwin_ahawam.yaml:45-47` |
| AHA-WAM RoboTwin | weight decay | 1e-2 / 0.01 | 论文/代码确认 | 论文 P9、P15 Table 4；`configs/task/robotwin_ahawam.yaml:53-54` |
| AHA-WAM RoboTwin | scheduler/warmup | cosine，warmup first 5%；代码 `warmup_ratio=0.05`，eta_min=LR*0.01 | 论文/代码确认 | 论文 P15 Table 4；`src/ahawam/trainer.py:159-176,355-392` |
| AHA-WAM RoboTwin | batch/epochs | 论文 global batch 512、epochs 5；released task config per-process batch 6、grad_accum 1、epochs 5 | 论文/代码确认且需按 GPU 数换算 | 论文 P15 Table 4；`configs/task/robotwin_ahawam.yaml:8,45-54` |
| AHA-WAM RoboTwin | precision/grad clip | bf16 mixed precision；max grad norm 1.0 | 代码确认；论文称 mixed precision + clipping | 论文 P9；`configs/train.yaml:27-31`；`src/ahawam/trainer.py:1096-1105` |
| AHA-WAM RoboTwin | flow timestep | video/action train timesteps=1000，shift=5.0；noise times logit-normal | 论文/代码确认 | 论文 P9、P15-P16；`configs/model/ahawam.yaml:61-69` |
| AHA-WAM RoboTwin | inference | default action denoising steps=10，CFG=1.0；sim eval `num_inference_steps=10` | 论文/代码确认 | 论文 P9、P15 Table 4；`configs/sim_robotwin.yaml:25-31` |
| AHA-WAM RoboTwin | parallelism | repo supports Accelerate + DeepSpeed ZeRO-1/ZeRO-2 launchers；actual paper training GPU type/count/time 未披露 | 代码确认/论文未披露 | `scripts/accelerate_configs/accelerate_zero2_ds.yaml`；`scripts/ds_configs/ds_zero2_config.json` |
| AHA-WAM-Flash ODE distill | teacher/student | teacher 16 denoising steps；student 2 steps；distilled timesteps 5,000；anchors 0,1,2,4,8,12,16；prediction parameterization Flow | 论文明确说明 | 论文 P15 Table 5、P18-P19 |
| AHA-WAM-Flash ODE distill | optimizer | AdamW，LR=2e-5，weight decay=1e-2，cosine，epochs=5，global batch=512，local batch/grad accum=16/4，workers=16 | 论文明确说明 | 论文 P15 Table 5 |
| AHA-WAM-Flash ODE distill | frozen modules | freeze Video-DiT，distill action denoising path only | 论文明确说明 | 论文 P18 |

未披露：训练总 step 数、训练 wall-clock time、训练 GPU/TPU 型号与数量、EMA、dropout、随机种子、数据采样比例、真实机器人 finetune 细节。

## 5. Benchmark 性能表

### RoboTwin 2.0

| Method | Robo. P.T. | Clean | Randomized | Avg | 公平性备注 | 证据 |
|---|---:|---:|---:|---:|---|---|
| π0 | 是 | 65.92 | 58.40 | 62.16 | 使用 embodied robot-data pretraining；与无 robot-data pretraining 的 AHA-WAM 不完全公平 | 论文 P9 Table 1；README `README.md:480-494` |
| π0.5 | 是 | 82.74 | 76.76 | 79.75 | 使用 embodied PT；输入/预训练不同 | 同上 |
| ABot-M0 | 是 | 81.20 | 80.40 | 80.80 | 使用 embodied PT；输入/预训练不同 | 同上 |
| Motus from Wan2.2 | 否（Wan2.2 初始化） | 77.56 | 77.00 | 77.28 | 更接近架构初始化公平性，但仍为不同模型 | 论文 P9 Table 1、P16 baseline discussion |
| Motus | 是 | 88.66 | 87.02 | 87.84 | 使用 robot-data pretraining版本，非完全公平 | 论文 P9 Table 1、P16 |
| LingBot-VA | 是 | 92.90 | 91.50 | 92.20 | 强 baseline，但有大规模 embodied PT；与 AHA-WAM 数据条件不同 | 论文 P9 Table 1、P16 |
| Fast-WAM | 否 | 91.88 | 91.78 | 91.83 | 最接近公平 baseline：同为 WAM，论文称用于隔离架构效果 | 论文 P9 Table 1、P16 |
| AHA-WAM-Flash | 否 | 90.48 | 89.92 | 90.20 | 同架构 distill 后速度优先 | 论文 P9 Table 1 |
| AHA-WAM | 否 | **93.40** | **92.20** | **92.80** | 与 Fast-WAM 最可比；高于 LingBot-VA 但后者有不同预训练 | 论文 P9 Table 1 |

协议：50 dual-arm tasks；每任务 50 clean + 500 randomized training demonstrations；clean/randomized 每任务各 100 eval trials；task-averaged success rate（论文 P9、P16）。

### 真实机器人

| 设置 | 任务/平台 | 数据/协议 | 结果 | 公平性备注 | 证据 |
|---|---|---|---|---|---|
| Original settings | AgileX Piper；Fold Towel / Organize Desktop / Prepare Soy Milk / Store Plate | 每任务约 120 demos；每模型每任务 30 trials；success=score 3 | AHA-WAM 78.33%；Motus 21.67%；Fast-WAM 68.33%；π0.5 76.67% | Fast-WAM/AHA-WAM 都先在 selected RoboCOIN subset 上预训练再同任务 finetune；π0.5 是强 generalist VLA，预训练来源不同 | 论文 P10-P11 Figure 4、P16-P17 Table 6 |
| Generalization shifts | lighting/object config/texture/novel environment | 30 trials；同时报告 0-3 progress score | AHA-WAM success 53.3%、score 35；π0.5 success 55.0%、score 33.25；Fast-WAM success 46.7%、score 31.5；Motus success 16.7%、score 19 | π0.5 generalization success 最高；AHA-WAM progress score 最高；预训练协议不同 | 论文 P11 Figure 4 |

### 延迟/频率

| Method | Latency Lchunk | Frequency | Speedup | 硬件/协议 | 证据 |
|---|---:|---:|---:|---|---|
| Motus | 1866.10 ms | 0.54 Hz | 0.10x | 除 Fast-WAM official latency 外，其余单 NVIDIA RTX 5090D；bf16；丢弃首个 warmup episode | 论文 P10 Table 3、P17 |
| Fast-WAM | 190.00 ms | 5.26 Hz | 1.00x | Fast-WAM official latency | 论文 P10 Table 3 |
| AHA-WAM | 41.37 ms | 24.17 Hz | 4.59x | RTX 5090D；10-step；CUDA/TRT/CG optimizations | 论文 P10 Table 3、P18 Table 8 |
| AHA-WAM-Flash | 17.56 ms | 56.95 Hz | 10.82x | RTX 5090D；2-step distilled sampler | 论文 P10 Table 3、P19 Table 11 |

## 6. 消融与局限

### 消融结果

| 设计 | Clean | Rand | Avg | 解释 | 证据 |
|---|---:|---:|---:|---|---|
| Fast-WAM | 91.88 | 91.78 | 91.83 | closest WAM baseline | 论文 P9 Table 2 |
| Naive-Async | 88.64 | 88.56 | 88.60 | 仅异步解耦会因 stale/phase-misaligned planner context 掉点 | 论文 P9-P10 Table 2 |
| + KV Memory | 91.40 | 90.62 | 91.01 | rolling K/V memory 恢复部分性能 | 论文 P9-P10 |
| + OVCR | 91.52 | 91.42 | 91.47 | observation-conditioned context adaptation 更直接缓解异步错位 | 论文 P10 |
| Full AHA-WAM | 93.40 | 92.20 | 92.80 | memory 与 routing 互补 | 论文 P10 |

### 延迟优化消融

| Stage | 优化 | Lchunk | Lprefill | 证据 |
|---|---|---:|---:|---|
| 0 | PyTorch eager baseline | 415.77 ms | 61.15 ms | 论文 P18 Table 8 |
| 1 | + Action DiT TRT + CUDA Graph | 83.87 ms | 未报告 | 论文 P18 Table 8 |
| 2 | + Memory/context TRT + CUDA Graph | 71.37 ms | 未报告 | 论文 P18 Table 8 |
| 3 | + Video-DiT prefill compile | 71.45 ms | 34.59 ms | 论文 P18 Table 8/9 |
| 4 | + Hot-path redundancy elimination | 50.37 ms | 未报告 | 论文 P18 |
| 7 | + VAE encoder TRT | 41.37 ms | 未报告 | 论文 P18 Table 8 |
| Flash 2-step | ODE distilled sampler | 17.56 ms | 未报告 | 论文 P19 Table 11 |

### 性能来源判断

- **主要来自架构**：Table 2 显示 Naive-Async 不够，KV memory + OVCR 才恢复并超过 Fast-WAM；这是核心架构贡献（论文 P9-P10）。
- **来自推理策略/系统优化**：24.17 Hz 主要来自把 Video-DiT 移出 action critical path 以及 TRT/CUDA Graph/hot-path 优化；Flash 56.95 Hz 来自 ODE distillation（论文 P8、P18-P19）。
- **来自数据/预训练**：RoboTwin 主结果声称无 robot-data pretraining；真实机器人实验为了公平稳定，对 Fast-WAM/AHA-WAM 都用 RoboCOIN subset 预训练，因此真实部署结果不能外推为“零 robot-data pretraining”（论文 P1、P10）。

### 局限

- 论文明确局限：planner update frequency、video horizon、action chunk size 是时间超参，最优分配依赖任务动态和 embodiment；缺少专门 long-horizon benchmark 系统评估（论文 P11）。
- 工程局限：真实机器人 checkpoint 未发布；真实机器人数据、RoboCOIN subset 抽样配置、训练硬件/时间未披露（README `README.md:68-74`；论文 P10）。
- 复现差异：论文 Table 4 LR=1e-4，但 released checkpoint config/model card 为 5e-5；复现实验需优先使用 checkpoint 对应配置，论文数字需单独标注（论文 P15；模型卡 `work/hf_modelcard.md:207-225`；`configs/task/robotwin_ahawam.yaml:45-47`）。

## 7. 最小复现步骤

1. 克隆官方仓库并记录 commit。

```bash
git clone https://github.com/serene-sivy/AHA-WAM.git
cd AHA-WAM
git rev-parse HEAD
```

2. 建环境。模型卡指定 Python 3.10、PyTorch 2.7.1+cu128、`pip install -e .`（模型卡 `work/hf_modelcard.md:64-80`）。

```bash
conda create -n ahawam python=3.10 -y
conda activate ahawam
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
pip install huggingface_hub
```

3. 准备 Wan assets。默认查找 Wan2.2-TI2V-5B DiT、Wan VAE、T5 encoder、Wan2.1 tokenizer；可用 ModelScope 或 HF mirror（README `README.md:127-176`）。

4. 准备 ActionDiT backbone（从 scratch 训练需要）。

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/ahawam.yaml \
  --output /path/to/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

5. 下载数据。最小路径使用 README 指向的预处理 RoboTwin2.0 derivative；严格复现需确认与论文 RoboTwin 2.0 版本/资产一致。

```bash
huggingface-cli download yuanty/robotwin2.0-fastwam \
  --repo-type dataset \
  --local-dir data/robotwin2.0-fastwam
cd data/robotwin2.0-fastwam
cat robotwin2.0.tar.gz.part-* | tar -xzf -
```

6. 预计算文本 embedding。

```bash
python scripts/precompute_text_embeds.py task=robotwin_ahawam \
  data.train.dataset_dirs=[/path/to/robotwin2.0] \
  data.train.text_embedding_cache_dir=/path/to/text_embeds_cache/robotwin
```

7. 训练 AHA-WAM。若目标是复现 released checkpoint，优先用模型卡/任务配置 LR=5e-5；若目标是论文 Table 4，另跑 LR=1e-4 对照。

```bash
bash scripts/train_zero1.sh 8 task=robotwin_ahawam \
  model.action_dit_pretrained_path=/path/to/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  data.train.dataset_dirs=[/path/to/robotwin2.0] \
  data.val.dataset_dirs=[/path/to/robotwin2.0] \
  data.train.pretrained_norm_stats=/path/to/dataset_stats.json \
  data.val.pretrained_norm_stats=/path/to/dataset_stats.json
```

8. 或直接评测 released checkpoint。

```bash
huggingface-cli download SereneC/AHA-WAM-RoboTwin2.0 \
  robotwin_ahawam.pt dataset_stats.json \
  --local-dir checkpoints/AHA-WAM-RoboTwin2.0

python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_ahawam \
  ckpt=checkpoints/AHA-WAM-RoboTwin2.0/robotwin_ahawam.pt \
  EVALUATION.dataset_stats_path=checkpoints/AHA-WAM-RoboTwin2.0/dataset_stats.json \
  EVALUATION.robotwin_root=/path/to/RoboTwin \
  MULTIRUN.num_gpus=8
```

### 复现资源估算

- **存储下限**：预处理数据 HF repo used storage 约 84.09GB；AHA-WAM checkpoint repo 约 29.40GB；合计约 113.5GB，不含 Wan base assets、RoboTwin simulator assets、训练中间 checkpoint/cache（HF API；数据集卡 `work/hf_datasetcard.md:21-40`）。
- **推理硬件**：论文 latency 在单 RTX 5090D 上报告；评测管理器默认 8 GPUs、每 GPU 2 tasks 并行，但可调（论文 P10/P17；`configs/sim_robotwin.yaml:40-43`）。
- **训练算力/时间**：论文未披露。代码示例是 `bash scripts/train_zero1.sh 8`，但论文 global batch=512；若使用 per-process batch=6，需要通过更多 GPU/梯度累积/配置改动达到 global batch 512。训练时间不可从一手资料可靠估算。

## 8. 未披露信息清单

| 类别 | 未披露/无法确认项 |
|---|---|
| 论文版本 | v1 之后是否有修订版：截至本次检索只确认 arXiv v1。 |
| 数据 | RoboTwin 原始小时数、token 数、清洗过滤细节、多数据集采样比例、真实机器人数据集文件、RoboCOIN subset 精确抽样规则、真实任务完整数据。 |
| 训练 | 实际训练 GPU 型号/数量、训练 wall-clock time、总 optimizer steps、随机种子、EMA、dropout、AdamW eps、是否使用论文 LR=1e-4 或 released checkpoint LR=5e-5 的最终实验配置。 |
| 架构 | 1.22B memory/context-routing 参数的逐子模块拆分未给；Flash student 完整权重差异未在论文表中逐层列出。 |
| 评测 | RoboTwin simulator 具体 commit/assets hash 未披露；真实机器人控制频率、相机分辨率/帧率、网络延迟、低层控制器细节未完整披露。 |
| 开放范围 | 真实机器人 pretrained checkpoint 和真实任务 finetune checkpoint 未发布；真实机器人数据未发布。 |

## 资料来源（仅一手资料）

- 论文：[arXiv:2606.09811](https://arxiv.org/abs/2606.09811) / [PDF](https://arxiv.org/pdf/2606.09811)
- 项目页：[https://serene-sivy.github.io/aha-wam/](https://serene-sivy.github.io/aha-wam/)
- 代码：[https://github.com/serene-sivy/AHA-WAM](https://github.com/serene-sivy/AHA-WAM)
- 模型卡：[SereneC/AHA-WAM-RoboTwin2.0](https://huggingface.co/SereneC/AHA-WAM-RoboTwin2.0)
- 数据集卡（README 指向的预处理 release）：[yuanty/robotwin2.0-fastwam](https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam)
