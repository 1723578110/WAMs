# RynnWorld-4D 工程复现技术报告

> 调研日期：2026-07-17  
> 证据范围：仅使用 arXiv 论文源码/PDF、官方 GitHub 仓库、官方 README/配置/训练脚本、官方 Hugging Face 模型卡/API 与官方模型文件列表。搜索结果只用于确认是否存在同名对象，不用于技术事实补全。

## 0. 研究对象确认

| 项目 | 结论 | 证据 |
|---|---|---|
| 准确标题 | **RynnWorld-4D: 4D Embodied World Models for Robotic Manipulation** | 论文明确说明：arXiv API `work/rynnworld_arxiv_api.xml`；论文源码 `work/rynn_arxiv_src/paper.tex:130`；README `work/RynnWorld-4D/README.md:3` |
| arXiv 版本/发布日期 | arXiv `2607.06559v1`，published/updated `2026-07-07T17:58:15Z` | 论文明确说明：`work/rynnworld_arxiv_api.xml` |
| 作者 | Haoyu Zhao, Xingyue Zhao, Siteng Huang, Xin Li, Deli Zhao, Zhongyu Li | 论文明确说明：`work/rynn_arxiv_src/paper.tex:132-137`；arXiv API |
| 机构 | DAMO Academy, Alibaba Group；Hong Kong Embodied AI Lab；CUHK；Hupan Lab | 论文明确说明：`work/rynn_arxiv_src/paper.tex:140-143` |
| 贡献/通信 | Haoyu Zhao、Xingyue Zhao equal contribution；Siteng Huang、Deli Zhao、Zhongyu Li corresponding authors | 论文明确说明：`work/rynn_arxiv_src/paper.tex:146` |
| 官方论文/项目/代码/模型 | arXiv: <https://arxiv.org/abs/2607.06559>；项目页：<https://alibaba-damo-academy.github.io/RynnWorld-4D.github.io>；GitHub：<https://github.com/alibaba-damo-academy/RynnWorld-4D>；Hugging Face：<https://huggingface.co/Alibaba-DAMO-Academy/RynnWorld-4D>；ModelScope：<https://www.modelscope.cn/models/DAMO_Academy/RynnWorld-4D> | 论文明确说明：`work/rynn_arxiv_src/paper.tex:159-162`；README `work/RynnWorld-4D/README.md:9`；arXiv comment |
| 模型权重开放范围 | HF 模型仓库公开、非 gated；文件含 `pytorch_model/mp_rank_00_model_states.pt` 与 `ema_weights.pt`，总存储约 `59,283,855,054` bytes；模型卡称发布 Stage-3 checkpoint | 代码/模型卡确认：`work/RynnWorld-4D_HF_API.json`；`work/RynnWorld-4D_HF_README.md:34,60-64,92-94` |
| 代码开放范围 | GitHub 发布训练、推理、策略头训练/服务脚本；README 称 2026-07-07 release code and checkpoints | 代码确认：`work/RynnWorld-4D/README.md:33,88-164` |
| 数据集开放范围 | 论文提出 Rynn4DDataset 1.0，但未发现官方独立数据集下载页/数据集卡；GitHub 仅随附 `data/sample.json` 与 3 episodes Tianji Pick-Place 样例用于 smoke test | 论文明确说明：`work/rynn_arxiv_src/Sec/3-method.tex:5-15`；代码确认：README `work/RynnWorld-4D/README.md:88,143` |
| 开源许可证 | Apache License 2.0；vendored third-party code follows original upstream licenses | 代码/模型卡确认：`work/RynnWorld-4D/LICENSE:1`；README `work/RynnWorld-4D/README.md:194-196`；HF `work/RynnWorld-4D_HF_README.md:201-203` |
| 同名项目核对 | 当前检索到的 `RynnWorld-4D` 均指向 Alibaba DAMO 官方 arXiv/项目页/GitHub/HF/ModelScope；未发现另一个同名开源模型/论文对象 | 检索确认；技术内容仍只引用上述一手资料 |

## 1. 200 字以内核心结论

RynnWorld-4D 是 Alibaba DAMO 发布的 4D embodied world model，不是 VLA 本体。它在 Wan2.2-TI2V-5B 上扩展 RGB/Depth/Optical Flow 三分支，通过每 3 层插入 Joint Cross-Modal Attention 同步生成 RGB-DF 未来视频，并用冻结世界模型特征训练 RynnWorld-4D-Policy 输出 54 维双臂双手 action chunk。权重和代码已开源，许可证 Apache-2.0；但核心 Rynn4DDataset 1.0 未公开下载，训练 GPU 数、耗时、完整 step 数和部分数据细节未披露，整体可复现性评级为 **中偏低**：可复现推理/小样例训练，难以完全复现实验结果。

## 2. 模型架构表

### 2.1 范式与信息流

```text
Training / inference input:
  RGB-D first observation + language instruction
        |
        v
  Wan2.2 I2V latent/VAE + text condition
        |
        v
+--------------------------------------------------------------------+
| RynnWorld-4D tri-branch diffusion transformer                      |
|                                                                    |
|  RGB branch      Depth branch        Flow branch                    |
|  self-attn/FFN   self-attn/FFN       self-attn/FFN                  |
|      |               |                  |                           |
|      +---- every 3 blocks: Joint Cross-Modal Attention ------------+
|      |     frame-wise cross-modal K/V + 3D RoPE + gated residual   |
|      v               v                  v                           |
|  future RGB      future depth       future optical flow             |
+--------------------------------------------------------------------+
        |
        +--> world-model output: synchronized RGB-DF video
        |
        +--> policy path: take hidden states from block 15 at t=500
             concat RGB/depth/flow features
                    |
                    v
              Flow Former
       frame-wise spatial cross-attn + temporal self-attn
                    |
                    v
        flow-matching action head, 4-step Euler ODE
                    |
                    v
           10-step action chunk, each action 54D
```

| 模块 | 结构/参数 | 输入/输出 | 证据与口径 |
|---|---|---|---|
| 总体范式 | 4D embodied world model / generative world model；下游策略为 inverse dynamics/action-from-latent policy；不是 VLA 主模型 | 输入单帧 RGB-D 观测 + instruction；输出同步 RGB、depth、optical flow 未来序列 | 论文明确说明：`Sec/3-method.tex:1-3,92`；README `work/RynnWorld-4D/README.md:17,27` |
| 主干 | Wan 2.2-TI2V-5B diffusion transformer；约 5B params；30-layer DiT；hidden size `d=3072`；FFN dim `14,336`；attention heads 未披露/未能从已下载一手材料确认 | 单分支 RGB I2V backbone 扩展为三分支 | 论文明确说明：`Sec/4-exp.tex:7`；模型卡确认约 5B：`work/RynnWorld-4D_HF_README.md:47` |
| 三分支 | RGB、Depth、Flow 三个分支；depth/flow branch 由预训练组件复制初始化，包括 patch embedding、self-attention、normalization、FFN | 每个分支处理对应模态 latent `z_t^m ∈ R^{T×C×H×W}` | 论文明确说明：`Sec/3-method.tex:96-107`；`Sec/4-exp.tex:7` |
| 文本条件 | text cross-attention K/V projections 在三分支共享 | language instruction 作为 modality-agnostic semantic signal | 论文明确说明：`Sec/4-exp.tex:9` |
| Joint Cross-Modal Attention, JA | 每 3 个 transformer blocks 插入一次，层号 `0,3,6,...,27`，共 10 个 JA；每分支有 zero-init modality embedding `e^m ∈ R^{1×1×d}`、per-modality LayerNorm、Q/KV projections、RMSNorm、zero-init OutProj、gate `g=1`；参数成本从 `18d^2` 降到 `12d^2`/block | query 来自本分支，K/V 来自另外两个分支；tokens reshape `[B,T*S,d]→[B*T,S,d]`，跨模态 attention 限定同一时间帧，并使用 3D RoPE | 论文明确说明：`Sec/3-method.tex:110-132`；实现位置：`work/RynnWorld-4D/core/finetune/models/wan_i2v/module_joint.py:485-519` |
| 训练目标 | Flow matching：`z_t^m=(1-t)z_0^m+t eps^m`；target velocity `eps^m-z_0^m`；首帧 conditioning latent 不监督；三模态共享同一个 Gaussian noise；`λ_rgb=λ_depth=1`，`λ_flow=0.5` stage1，stage2/3 为 `1.0` | 输出三模态 velocity，loss 为三模态 MSE 加权和 | 论文明确说明：`Sec/3-method.tex:146-159`；代码确认：`rynnworld4d_trainer.py:831-844,890-908` |
| Branch Dropout | Stage2/3 以 `p_drop` 随机选择 depth 或 flow，将 noisy latent frames `[1:]` 替换为 pure Gaussian noise；RGB never dropped | 强迫 JA 从可见模态重建被 drop 模态 | 论文明确说明：`Sec/3-method.tex:138-144`；代码确认：`rynnworld4d_trainer.py:854-858` |
| 世界模型推理 | 官方推理默认 `num_inference_steps=50`；HF 建议 `guidance_scale=1.0`，使用 EMA 权重 | 输出 RGB/depth/flow 视频 | 代码/模型卡确认：`work/RynnWorld-4D/inference-sft.py:143-144,298-299`；HF `work/RynnWorld-4D_HF_README.md:101-126` |
| RynnWorld-4D-Policy | 冻结 RynnWorld-4D；在 diffusion timestep `t=500` 做单步 feature extraction；抽取 transformer block 15；concat RGB/depth/flow 的 `3072`-dim features；Flow Former 压缩；flow matching action head | 输入当前 RGB 观测、instruction、proprioception；输出 action chunk length `10`，每个 action `54D` | 论文明确说明：`Sec/4-exp.tex:78-93`；配置确认：`rynnworld4d_policy/policy_conf/train_config.yaml:46-65,75-101` |
| Flow Former | 论文描述为 learnable queries + frame-wise spatial cross-attn + temporal self-attn；代码配置 `Former_depth=6`、`Former_heads=8`、`Former_dim_head=64`、`Former_num_time_embeds=21`、`num_latents=336` | 输入 `F_p ∈ R^{B×T×3C×H×W}`，输出固定大小策略表示 | 论文明确说明：`Sec/3-method.tex:164-178`；代码确认：`policy_conf/train_config.yaml:61-65` |
| Action 表示 | Tianji dual-arm + dual-hand 共 `54D = 7+7+20+20`；action_seq_len/act_window_size `10`；policy inference `4` step Euler ODE | 4-step ODE 生成 10 个未来动作，部署 open-loop 执行后再 re-query | 论文明确说明：`Sec/4-exp.tex:87-93`；代码确认：`policy_conf/train_config.yaml:21,47-55`；policy README `rynnworld4d_policy/README.md:68` |
| 推理延迟/控制频率 | RTX 5090 + FP8 + FA3：DA3 `85 ms`，VAE prep `18 ms`，RynnWorld-4D `990 ms`，Flow Former `4 ms`，action head `8 ms`，总 `1106 ms`；planning `≈0.9 Hz`，chunk 后有效控制 `≈9 Hz`，cached lookup 50 Hz | 真实机器人控制 | 论文明确说明：`Sec/3-method.tex:186-222`；Table 1 |
| 缓存/早退 | policy 配置 `wan_num_inference_layers=20`，注释称 features from block 15，跳过更后层以省算力；服务端缓存上一个 action chunk | 代码路径用于部署工程优化 | 代码确认：`policy_conf/train_config.yaml:94-101`；论文 Table 1/延迟段落 |

## 3. 训练数据表

| 数据/来源 | 任务与 embodiment | 数量 | 模态/相机/频率 | 清洗、标注与归一化 | 公开性 | 证据 |
|---|---|---:|---|---|---|---|
| Rynn4DDataset 1.0：Epic-Kitchens、EgoVid、RoboMIND、RDT-1B、Galaxea、RoboCoin、AgiBot | human egocentric + robotic manipulation；用于训练 RynnWorld-4D 世界模型 | over `254.4M` video frames；每数据源 hours/episodes/frames 未披露 | RGB 视频；论文未披露相机数量、原始分辨率、控制频率、state/action 维度；伪标注生成 depth/flow/instruction | Qwen3-VL caption：视频 `1 FPS` 采样，分为 `5 s` segments，max output `512` tokens，temperature `0.7`，保存 JSON；DPFlow：frame pairs native resolution，flow video `25 FPS`；Depth Anything 3 `DA3NESTED-GIANT-LARGE-1.1`，视频 `30 FPS`，短边 working resolution `392`，depth clip `[0,5]m`，`floor(d/dmax*255)` 8-bit grayscale，再保存 RGB video | 数据集本体未找到官方下载/数据集卡；论文只给组成与标注流程 | 论文明确说明：`Sec/3-method.tex:5-63`；Figure 2/3 |
| 真实机器人数据：Tianji M6 + Wuji Hand，六任务 | Dual Picking、Block Pushing、Hand-over、Bimanual Lifting、Lid Placement、Bowl Stacking；双臂双灵巧手 | 世界模型训练 episodes：`500/500/300/500/300/300`，总 `2400`；policy：每任务 `200`，总 `1200` | RealSense D435i FPV；双 Tianji 7-DOF arm + 双 Wuji 20-DOF hand，共 `54 DOF`；原始帧率/图像分辨率未披露；policy 使用 `480×640` crop | 初始对象 6-DoF pose 大幅随机化；teleoperation：5 个 HTC Vive trackers，wrist-to-chest transforms `100-120 Hz`；Pinocchio IK，Ruckig 平滑；arm commands `200 Hz`；Manus glove -> 21-point MediaPipe skeleton -> 20-DOF hand，EMA smoothing | 完整真实数据未公开；GitHub 只带 3 episodes Tianji Pick-Place 样例 | 论文明确说明：`Sec/4-exp.tex:111-154`，Table 3；Appendix `Sec/Appendix.tex:1-21`；README `work/RynnWorld-4D/README.md:143` |
| GitHub sample data | smoke-test world/policy pipeline | `data/sample.json`；policy sample 3 episodes | head video + parquet + metadata + depth videos | 仅用于跑通脚本，不等价论文训练集 | 公开随代码 | 代码确认：README `work/RynnWorld-4D/README.md:88,143` |

未披露：Rynn4DDataset 1.0 各源数据混合比例、每源 hours/episodes/trajectories/token 数、完整过滤规则、数据增强、训练采样权重、公开数据与私有数据的精确边界、action/state normalization 统计量。

## 4. 训练 Recipe 和超参数表

### 4.1 论文实验配置

| 阶段 | 初始化 checkpoint | 训练/冻结模块 | Loss/权重 | Optimizer/LR/Scheduler | Batch/Steps/Precision/Parallel | 证据 |
|---|---|---|---|---|---|---|
| Stage 1: Modality Adaptation | Wan2.2-TI2V-5B；depth/flow 由 RGB 预训练组件复制初始化 | 禁用 JA；三分支独立训练；训练 all branches | Flow matching；`λ_flow=0.5`，`λ_rgb=λ_depth=1`；首帧不监督 | AdamW `β1=0.9, β2=0.95`，weight decay `1e-4`；LR `2e-5`；linear warmup `500`；cosine schedule；EMA decay `0.9999` CPU shadow | 分辨率 `81×480×640`，latent `T=21`；per-GPU batch `1`，grad accumulation `2-4`，effective batch `1×accum×N_GPU`；bf16；gradient checkpointing；GPU 数、steps、epochs、训练时间未披露 | 论文明确说明：`Sec/4-exp.tex:16-66`，Table 2 |
| Stage 2: Frozen-Backbone JA | Stage1 model-only checkpoint；optimizer/scheduler reset | 冻结 backbone 与 per-branch self-attn/FFN；只训练 JA projections、RMSNorm、per-modality LayerNorm、tanh gates、modality embeddings | Flow matching；`λ_flow=1.0`；Branch Dropout `p=0.2` on depth/flow | AdamW 同上；LR `5e-5`；warmup `200`；cosine；EMA `0.9999` | 同上；DeepSpeed ZeRO-2 optimizer offload | 论文明确说明：`Sec/3-method.tex:134-144`；`Sec/4-exp.tex:16-66`，Table 2 |
| Stage 3: Full-Parameter Joint SFT | Stage2 model-only checkpoint；optimizer/scheduler reset | unfreeze entire model；joint full SFT on Rynn4DDataset 1.0 | Flow matching；`λ_flow=1.0`；Branch Dropout `p=0.1` | AdamW 同上；LR `1e-5`；warmup `500`；cosine；EMA `0.9999` | 同上；DeepSpeed ZeRO-2 optimizer offload | 论文明确说明：`Sec/4-exp.tex:47-66`，Table 2 |
| RynnWorld-4D-Policy | Frozen RynnWorld-4D backbone | 只训练 Flow Former + flow matching action head；RynnWorld-4D frozen | action-space flow matching；inference 4-step Euler ODE；具体 loss 权重未披露 | AdamW LR `1e-4`，`β=(0.9,0.9)`，weight decay `0.05`；tri-stage LR：2% warmup、8% hold、90% cosine to `1e-6×peak` | mixed precision；batch size `1/GPU`；`100 epochs`；GPU 数、训练时间未披露 | 论文明确说明：`Sec/4-exp.tex:78-93` |

### 4.2 官方代码默认值与论文表 2 的差异

| 脚本/配置 | 代码确认的默认值 | 与论文配置关系 |
|---|---|---|
| `scripts/rynnworld4d-stage1.sh` | `train_epochs=1`，batch `1`，grad accum `4`，bf16，LR `1.5e-5`，warmup `300`，resolution `25x480x832`，`fusion_mode none`，`share_ffn False`，`use_ema True`，`loss_weight_flow 0.5` | 与论文 Stage1 LR `2e-5`、warmup `500`、resolution `81×480×640` 不一致；应视为发布脚本默认/样例或当前工程配置 |
| `scripts/rynnworld4d-stage2.sh` | batch `1`，grad accum `2`，LR `5e-5`，`joint_out_lr=2.5e-4`，warmup `200`，resolution `25x480x832`，JA layers `0-30 every 3`，frame-wise+RoPE，`joint_unidirectional True`，`freeze_non_joint True`，branch dropout `0.2`，ZeRO-2 offload | 大部分与论文 Stage2 一致；resolution 与论文不一致；代码额外给出 joint_out_lr |
| `scripts/rynnworld4d-stage3.sh` | batch `1`，grad accum `2`，LR `5e-6`，`joint_out_lr=1e-5`，warmup `500`，resolution `25x480x832`，`ema_decay=0.999`，branch dropout `0.05`，`freeze_non_joint False`；注释称数据 `rynnworld4d-all-with-tianji-add.json (3.17M samples)` | 与论文 Stage3 LR `1e-5`、EMA `0.9999`、dropout `0.1`、resolution `81×480×640` 不一致；报告优先记录论文实验值，代码值用于复现脚本 |
| `finetune_rynnworld4d.py` / trainer | parser defaults：AdamW、`β1=0.9, β2=0.95`，epsilon `1e-8`，weight_decay `1e-4`，max_grad_norm `1.0`，`cosine_with_warmup`；timestep 由 `torch.randint` 均匀采样后使用 scheduler `flow_shift=5.0` 变换 | 代码补充了 epsilon、grad clipping、timestep 实现；论文未披露 epsilon/max_grad_norm/flow_shift |
| `rynnworld4d_policy/policy_conf/train_config.yaml` | `max_epochs=50`，trainer `precision=16`，`total_steps=250000`，`limit_train_batches=3000/5000`，Flow Former depth `6` heads `8` dim_head `64` num_latents `336`，`num_inference_steps=4`，`action_dim=54` | 与论文 policy `100 epochs` 不一致；配置文件更像可运行默认/样例配置 |

## 5. Benchmark 性能表

### 5.1 World model benchmark

评测协议：held-out test set `50` video sequences，随机采自 RoboMIND/RDT-1B/Galaxea；评估 visual synthesis quality、pixel fidelity、depth accuracy、flow accuracy。随机种子只在部分 baseline 实现细节中披露，RynnWorld 自身评测 seed 未披露。

| Method | IQ ↑ | MS ↑ | SC ↑ | Subj. ↑ | SSIM ↑ | PSNR ↑ | LPIPS ↓ | AbsRel ↓ | δ1 ↑ | AEPE ↓ | 公平性备注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CogVideoX | 0.604 | 0.976 | 0.866 | 0.917 | 0.534 | 12.17 | 0.577 | N/A | N/A | N/A | 2D video generation baseline，不能原生输出 depth/flow |
| Wan-2.2-TI2V-5B | 0.555 | 0.970 | 0.886 | 0.909 | 0.593 | 14.54 | 0.489 | N/A | N/A | N/A | RynnWorld 主干相关；但 baseline 非 4D 输出 |
| Wan-2.1-I2V-14B | **0.684** | 0.988 | 0.891 | 0.956 | 0.536 | 12.72 | 0.568 | N/A | N/A | N/A | 参数量更大，但非 4D 输出 |
| Free4D | 0.354 | 0.993 | 0.787 | 0.848 | 0.492 | 12.40 | 0.597 | 0.804 | 0.179 | N/A | 4DGS pipeline；depth 任意尺度需 median scaling；无 flow |
| TesserAct | 0.608 | 0.992 | 0.904 | 0.956 | 0.693 | 16.91 | 0.335 | 0.699 | 0.279 | N/A | 需额外 depth/normal conditioning，输入协议不同 |
| 4DNeX | 0.637 | 0.994 | 0.917 | 0.986 | 0.649 | 14.47 | 0.404 | 0.423 | 0.327 | N/A | 使用 Wan2.1-I2V-14B + LoRA/domain embeddings；输出 pointmap depth |
| **RynnWorld-4D** | 0.635 | **0.995** | **0.957** | **0.992** | **0.754** | **17.85** | **0.269** | **0.310** | **0.610** | **0.170** | 唯一原生同步 RGB/depth/flow；与不能输出 flow 的 baseline 在 AEPE 上不可直接比较 |

证据：论文明确说明，Table 4：`work/rynn_arxiv_src/Sec/4-exp.tex:160-205`；评测协议：`Sec/4-exp.tex:95-109`；baseline 实现：`Sec/Appendix.tex:127-136`。

### 5.2 真实机器人策略 benchmark

评测协议：六个真实任务，每项 `35` consecutive trials；任务在 `120 s` 内完成算成功；初始物体 6-DoF pose 随机化；硬件为 TIANJI M6 + WUJI HAND + RealSense D435i FPV。

| Method | Dual Picking | Block Pushing | Hand-over | Bimanual Lifting | Lid Placement | Bowl Stacking | 备注 |
|---|---:|---:|---:|---:|---:|---:|---|
| DP | 77.14 | 85.71 | 17.14 | 88.57 | 57.14 | 57.14 | policy baseline；额外预训练/输入细节未完整披露 |
| π0 | 88.57 | 94.29 | 2.86 | 91.43 | 34.29 | 51.43 | foundation policy；论文指出预训练偏 parallel-jaw grippers，和 dexterous hand embodiment 存在差异 |
| π0.5 | **94.29** | **100.00** | 0.00 | 94.29 | 37.14 | 42.86 | foundation policy；数据/预训练协议与 RynnWorld-4D-Policy 不同 |
| **RynnWorld-4D-Policy** | **94.29** | 97.14 | **28.57** | **97.14** | **65.71** | **65.71** | 使用冻结 RynnWorld-4D 4D latent；同一真实任务平台评测 |

证据：论文明确说明，Table 5：`work/rynn_arxiv_src/Sec/4-exp.tex:224-253`；协议：`Sec/4-exp.tex:102-154`。

### 5.3 推理性能

| 阶段 | 延迟 | 占比 | 硬件/设置 |
|---|---:|---:|---|
| DA3 depth estimation | 85 ms | 7.7% | NVIDIA RTX 5090，FP8 + FlashAttention 3 |
| VAE encoding & latent prep | 18 ms | 1.6% | 同上 |
| RynnWorld-4D | 990 ms | 89.5% | 同上 |
| Feature reshape & concat | 1 ms | 0.1% | 同上 |
| Flow Former | 4 ms | 0.4% | 同上 |
| Action Flow Matching Head | 8 ms | 0.7% | 同上 |
| Total forward | 1106 ms | 100% | planning ≈0.9 Hz；K=10 action chunk 后 effective control ≈9 Hz |

证据：论文明确说明，Table 1：`work/rynn_arxiv_src/Sec/3-method.tex:186-222`。

## 6. 消融与局限

### 6.1 消融结果

| 消融 | 关键结果 | 结论 | 证据 |
|---|---|---|---|
| Independent Branches | AbsRel `0.737` vs full `0.310`；AEPE `0.247` vs full `0.170` | 主要收益来自同步三模态融合/JA，而不是三个独立生成器 | 论文明确说明：Table 4；`Sec/4-exp.tex:266-269` |
| w/o MA | δ1 `0.479` vs full `0.610`；AEPE `0.231` vs `0.170` | Stage1 modality adaptation 对 depth/flow 从 RGB prior 迁移很重要 | 论文明确说明：`Sec/4-exp.tex:271-274` |
| w/o 4D Pre-training | AEPE `0.729` vs `0.170`；AbsRel `0.797` vs `0.310` | 性能高度依赖 Rynn4DDataset 1.0 大规模预训练；任务数据 alone 不足 | 论文明确说明：`Sec/4-exp.tex:276-279` |
| w/o RoPE in JA | δ1 `0.450` vs `0.610`；AEPE `0.210` vs `0.170` | 3D RoPE 提供跨模态空间对齐 | 论文明确说明：`Sec/4-exp.tex:289-293` |
| shared FFN | AbsRel `0.580`，δ1 `0.380`，AEPE `0.280` | 模态特异 FFN 对 RGB/depth/flow 异构 latent 很关键 | 论文明确说明：`Sec/4-exp.tex:295-302` |
| Policy w/o RynnWorld-4D | Dual Picking `71.43` vs full `94.29`；各任务下降 | predictive 4D latent 强于静态 2D feature | 论文明确说明：Table 5；`Sec/4-exp.tex:281-286` |
| Policy RGB/RGB+Depth/RGB+Flow | full RGB+Depth+Flow 在 6 任务总体最好；RGB+Depth 对空间任务收益明显，RGB+Flow 对运动任务有帮助 | 策略收益来自 4D 模态互补 | 论文明确说明：Table 5；`Sec/4-exp.tex:284-287` |

### 6.2 局限与复现风险

- **数据依赖大**：Rynn4DDataset 1.0 是 `254.4M` frames 级别，且未找到官方数据集卡/下载页；完整预训练不可直接复现。
- **算力与硬件依赖大**：世界模型基于约 5B Wan2.2 主干并扩展三分支，Stage2/3 需要 DeepSpeed ZeRO-2 offload；论文未披露 GPU 数与训练时间。
- **真实机器人复现门槛高**：需要 TIANJI M6、WUJI Hand、RealSense D435i、HTC Vive trackers、Manus gloves、Pinocchio/Ruckig/LCM 控制链。
- **推理延迟仍高**：RTX 5090 FP8 + FA3 下世界模型仍占 `990/1106 ms`，实际靠 action chunk 达到约 `9 Hz`，对高频接触控制仍是瓶颈。
- **代码/论文配置不完全一致**：官方训练脚本默认分辨率 `25x480x832`，而论文实验为 `81×480×640`；Stage1/Stage3 LR、EMA、branch dropout 也不同。
- **模型卡局限**：HF 模型卡称长序列可能 drift，bidirectional joint attention 仍可能把 depth/flow artifacts 传到 RGB；guidance scale >1 需要 null-prompt embedding 文件。

整体可复现性评级：**中偏低**。理由：推理和样例训练可运行，世界模型/策略代码与权重已开源；但核心数据集、完整实验训练资源、真实机器人硬件和若干关键实验配置未公开。

## 7. 最小复现步骤

1. 获取代码与依赖  
   ```bash
   git clone https://github.com/Alibaba-DAMO-Academy/RynnWorld-4D
   cd RynnWorld-4D
   pip install -r requirements.txt
   pip install -e . --no-build-isolation
   ```
   证据：README `work/RynnWorld-4D/README.md:39-49`。

2. 下载 backbone 与 RynnWorld-4D Stage-3 权重  
   ```bash
   huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
     --local-dir ./pretrained/Wan2.2-TI2V-5B-Diffusers
   huggingface-cli download Alibaba-DAMO-Academy/RynnWorld-4D \
     --local-dir ./pretrained/RynnWorld-4D
   ```
   证据：README `work/RynnWorld-4D/README.md:57-79`；HF model card `work/RynnWorld-4D_HF_README.md:85-94`。

3. 跑世界模型推理  
   ```bash
   python inference-sft.py \
     --model_path ./pretrained/Wan2.2-TI2V-5B-Diffusers \
     --checkpoint_path ./pretrained/RynnWorld-4D \
     --fusion_mode joint \
     --share_ffn False \
     --joint_start_layer 0 \
     --joint_end_layer 30 \
     --joint_every_n_layers 3 \
     --joint_frame_wise True \
     --joint_use_rope True \
     --joint_unidirectional False \
     --use_ema True \
     --num_inference_steps 50 \
     --guidance_scale 1.0
   ```
   证据：README/HF inference flags：`work/RynnWorld-4D_HF_README.md:101-126`；`work/RynnWorld-4D/inference-sft.py:294-314`。

4. 若只做工程 smoke test，运行官方样例训练脚本  
   - Stage1：`bash scripts/rynnworld4d-stage1.sh`
   - Stage2：`bash scripts/rynnworld4d-stage2.sh`
   - Stage3：`bash scripts/rynnworld4d-stage3.sh`
   注意这些脚本需要 `WORLD_SIZE/RANK/MASTER_ADDR/MASTER_PORT/NPROC_PER_NODE` 等分布式环境变量；脚本默认值不是论文完整实验配置。

5. 训练/服务 policy  
   - 下载 DA3：`huggingface-cli download depth-anything/Depth-Anything-3 --local-dir ./pretrained/da3`
   - 修改 `rynnworld4d_policy/policy_conf/train_config.yaml` 中 `pretrained_model_path` 与 `rynnworld4d_ckpt`
   - 运行：`bash scripts/rynnworld4d-policy.sh`
   - 服务：`python serve_rynnworld4d_policy.py --port 8099 --checkpoint <policy_ckpt>`
   证据：README `work/RynnWorld-4D/README.md:138-162`；policy config。

6. 若尝试完整训练复现，需要自行重建数据管线  
   - 收集论文列出的 human/robotic video sources。
   - Qwen3-VL caption：1 FPS，5 秒分段，max 512 tokens，temperature 0.7。
   - DPFlow 生成 25 FPS flow videos。
   - DA3 生成 depth，30 FPS，短边 392，clip `[0,5]m`，8-bit quantization。
   - 按论文三阶段 recipe 训练；但 GPU 数、训练时长、完整 step 数和数据混合比例未披露。

复现算力/存储估算：官方 HF RynnWorld 权重约 `59.3GB`，另需 Wan2.2-TI2V-5B、DA3 与训练缓存；全量 Rynn4DDataset 1.0 为 `254.4M` frames，视频+depth+flow+latent cache 存储至少 TB 级，论文未给精确大小。训练需要多 GPU + ZeRO-2 offload；精确 GPU 型号/数量/时间未披露，不能可靠估算。

## 8. 未披露信息清单

| 类别 | 未披露项 |
|---|---|
| 数据 | Rynn4DDataset 1.0 官方下载地址/数据集卡；各数据源小时数、episodes、轨迹数、帧数分布；多数据集采样比例；完整清洗/过滤规则；数据增强；公开/私有边界；训练 captions/token 总数 |
| 模态/控制 | 各原始数据集相机数量、原始分辨率、原始帧率、控制频率、action/state 维度；真实数据 action/state normalization 统计量 |
| 模型 | Wan2.2 backbone attention heads 在 RynnWorld 论文/仓库中未披露；RynnWorld 总参数量未明确给出；JA 新增参数量未披露；显存峰值未披露 |
| 训练 | GPU/TPU 类型、数量、训练总时长、实际 optimizer steps、epochs、样本/token 数；dropout 除 branch dropout 外的细节；EMA 是否用于全部阶段最终评测；完整 DeepSpeed/FSDP 拓扑 |
| 评测 | RynnWorld 自身评测随机种子；policy baseline 的完整训练数据、额外预训练与输入协议；真实机器人评测每任务随机种子/场景分布 |
| 推理 | 世界模型单次推理显存；非 RTX 5090 延迟；policy 端到端部署中的网络/相机/控制栈同步细节；缓存策略实现细节 |

## 主要一手来源

- arXiv: <https://arxiv.org/abs/2607.06559>
- Project page: <https://alibaba-damo-academy.github.io/RynnWorld-4D.github.io>
- GitHub: <https://github.com/alibaba-damo-academy/RynnWorld-4D>
- Hugging Face model: <https://huggingface.co/Alibaba-DAMO-Academy/RynnWorld-4D>
- ModelScope: <https://www.modelscope.cn/models/DAMO_Academy/RynnWorld-4D>
