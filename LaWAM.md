# LaWAM: Latent World Action Models for Efficient Dynamics-Aware Robot Policies

## 1. 核心结论（200 字以内）

LaWAM 是一个把 VLA 与 Latent World Model 结合的 World-Action Model：先用冻结 DINOv3 特征训练 LaWM，再让 Qwen3-VL/GR00T 风格策略预测 latent action，经 LaWM 单次前向得到 latent visual subgoal，并用 flow-matching action expert 输出动作 chunk。论文报告 LIBERO 平均 98.6%、RoboTwin 平均 91.22%、真实任务平均 90.0%，单次 action-chunk 延迟 187 ms。工程复现的主要门槛是未完全公开的预训练混合数据、64 H100 级 policy pretrain，以及 LaWM/LAM 数据混合细节。

## 2. 研究对象确认

| 项目 | 结论 | 证据与可复现备注 |
|---|---|---|
| 准确标题 | `LaWAM: Latent World Action Models for Efficient Dynamics-Aware Robot Policies` | 论文标题：arXiv API `2606.15768v1`；TeX 源 `example.tex:37`。 |
| 作者 | Jialei Chen, Kai Wang, Kang Chen, Shuaihang Chen, Feng Gao, Wenhao Tang, Zhiyuan Li, Weilin Liu, Zhuyu Yao, Boxun Li, Yuanbo Xu, Chao Yu | arXiv API；TeX 源 `example.tex:51-76`。 |
| 机构 | Tsinghua University, Jilin University, Nankai University, Peking University, Harbin Institute of Technology, Zhongguancun Academy, Striding.AI, Infinigence AI | TeX 源 `example.tex:69-76`。 |
| 版本与发布日期 | arXiv `v1`，published/updated `2026-06-14 12:06:58 UTC` | arXiv API `work/arxiv_2606.15768.xml`。 |
| 官方论文 | [arXiv:2606.15768](https://arxiv.org/abs/2606.15768) / PDF | arXiv API 与 PDF。 |
| 官方项目页 | [rlinf.github.io/LaWAM](https://rlinf.github.io/LaWAM/) | README badge 与项目页一致：`README.md:4-8`。 |
| 官方代码仓库 | [github.com/RLinf/LaWAM](https://github.com/RLinf/LaWAM) | 本地克隆 HEAD `1add20a376126eacab02f19a62d726072a322cae`，2026-07-15。 |
| 权重 | `Qwen/Qwen3-VL-2B-Instruct`, `facebook/dinov3-vitb16-pretrain-lvd1689m`, `jialei02/lawam_lam`, `jialei02/lawam_pretrain`, `jialei02/lawam_libero_sft_release`, `jialei02/lawam_robotwin_sft_release` | README 模型准备表：`README.md:104-116`。 |
| 数据集 | `jialei02/libero_merged_no_noops_20hz`, `jialei02/robotwin_merged` | README：`README.md:337-352`, `README.md:387-403`；训练配置 `train_libero.yaml:55-68`, `train_robotwin.yaml:53-64`。 |
| 开源许可证 | LaWAM Python package 标为 MIT；README 说明代码基于 StarVLA 并保留 MIT；部分 vendored GR00T/LeRobot 文件有 Apache-2.0 SPDX | `pyproject.toml:33-38`, `README.md:489-491`，代码许可证搜索。DINOv3 为 gated model，另有官方 license；Qwen3-VL HF 卡为 Apache-2.0。 |
| 实际开放范围 | 训练/评测代码、SFT 数据、LaWM/LAM、pretrain、LIBERO/RoboTwin SFT checkpoint 均给出下载地址；LaWM 预训练原始 3,000h robot + 1,500h egocentric 数据混合的逐数据源清单、采样比例与清洗规则未在论文正式内容中披露 | README 模型/数据表；论文 p.4, `example.tex:295-296`；TeX 末尾有注释掉的 ratio 表，未进入正文，本文不按其补数。 |
| 同名项目排除 | 官方对象由 arXiv、项目页、GitHub README、HF checkpoint/dataset 相互指向同一标题；检索未发现另一个官方同名 LaWAM 机器人论文/项目与之冲突 | 以官方互链为准；非官方搜索结果未作为技术证据。 |

## 3. 模型架构表

### 3.1 方法定位

LaWAM 属于 World-Action Model（WAM）/VLA hybrid，而不是单纯 VLA。它保留 VLA 的视觉-语言语义 backbone，同时把未来预测从 pixel/video generation 换成 frozen visual latent space 中的 `latent visual subgoal`。论文把核心模块称为 LaWM：一个 latent-action-conditioned Latent World Model（论文 p.1-4；`example.tex:96-103`, `137-139`, `162-165`）。

### 3.2 模块参数

| 模块 | 结构与参数 | 输入 | 输出 | 证据等级 |
|---|---:|---|---|---|
| 视觉 latent encoder | DINOv3 ViT-B/16，冻结；LaWM 在其 patch feature space 工作 | RGB image，训练/评测统一 resize 到 `256x256` | 当前/未来视觉 patch features `u`, `u_T` | 论文明确：p.14, `example.tex:420`, `424`；代码确认：`dino_large_vae.yaml:11`, `25-26`。DINO 原生 hidden/layers 由外部 official HF config 决定，LaWAM 仓库不 vendored。 |
| LaWM/LAM encoder | inverse-dynamics encoder，Transformer；论文写 encoder/decoder 都是 24-layer transformer | 当前/未来 observation features `(u,u_T)`，endpoint state 可用于 auxiliary | latent action posterior `z` | 论文明确：p.14, `example.tex:420`。代码 release YAML：`enc_layers:24`, `dim:1024`, `num_heads:16`, `ffn_expansion_factor:4`, `code_dim:32`, `num_queries:1`, `dropout:0.0`，见 `latent_action_model/config/dino_large_vae.yaml:14-30`。注意 release YAML 中 `dec_layers:12` 与论文“decoder 24-layer”不一致；以实际 release config 复现应按 YAML。 |
| LaWM decoder | LAM forward decoder，policy-facing latent world model；latent action 通过 adaptive layer norm 条件化 | 当前 feature `u` + latent action `z` | 预测未来 latent feature `\hat u_T` | 论文明确：p.14, `example.tex:420`; 代码 release config `dec_layers:12`：`dino_large_vae.yaml:29`。 |
| Auxiliary state predictor | lightweight MLP heads，训练 LaWM 时预测 horizon state，训练后丢弃 | current EEF state `s` + latent action `z` | horizon state `s_T` | 论文明确：p.4, `example.tex:243-254`；代码：`latent_action_model/core/lam_model.py` state decoder 路径。 |
| VLM backbone | Qwen3-VL-2B-Instruct；LaWAM 使用前 16 个 Qwen3-VL transformer layers | primary-view image, task instruction, optional aux-view images, query tokens | VLM hidden states | 论文明确：p.14, `example.tex:424`；代码 `keep_llm_first_n_layers:16`：`train_libero.yaml:92-100`, `train_robotwin.yaml:88-96`。Qwen3-VL official config text hidden size `2048`, heads `16`, layers `28`, FFN/intermediate `6144`；LaWAM flow config也固定 `vlm_dim:2048`：`train_libero.yaml:37`。 |
| Latent-action query mapper | 8 latent-action query tokens + 1-layer QFormer-style aggregation；cross-attn heads 8, FFN expansion 4, dropout 0 | VLM hidden states | predicted latent action `\hat z` | 代码确认：`lawam.py:73-74`, `282-292`。 |
| Action expert | Alternate-DiT / conditional flow matching；4 Alternate-DiT blocks = 16 transformer layers，hidden 1024；code config：`num_layers=16`, `attention_heads=16`, `action_dim=32`, `state_dim=32`, `use_state=false` | VLM context, current latent feature `u`, predicted subgoal `\hat u_T`, masked/padded actions during training | action chunk | 论文明确：p.14, `example.tex:424`; 代码确认：`train_libero.yaml:33-54`, `train_robotwin.yaml:33-52`, `flowmatching_expert.py:156-193`, `242-256`。FFN size 未披露。 |
| Action chunk / horizon | policy config `action_horizon=50` as padded max；LIBERO `horizon_sec=0.4`, RoboTwin `horizon_sec=1.2`; data config action indices LIBERO `range(8)`, RoboTwin `range(16)` | valid actions are masked by native action Hz | padded action sequence, valid chunk length by physical-time horizon | 代码确认：`train_libero.yaml:14-16`, `49`, `data_config.py:541-542`; `train_robotwin.yaml:14-16`, `48`, `data_config.py:1067-1068`; runtime horizon check `runner.py:54-61`。 |
| 参数量 | full LaWAM `2.3B`，LaWM `230M`，pixel WAM baseline WAN backbone `5B`；论文报告 LaWM 参数少约 `95%` | - | - | 论文明确：p.2-5, `example.tex:167-169`, `305`；LIBERO Table 1 `tab_libero_comparison.tex:48`。 |

### 3.3 架构图

```text
Stage 1: LaWM / latent action model pretraining

 RGB o_t --------> frozen DINOv3 ViT-B/16 ----> u_t -----------+
                                                               |
 RGB o_{t+tau} -> frozen DINOv3 ViT-B/16 ----> u_T ----+       |
                                                       |       v
                                                inverse-dynamics encoder
                                                       |  q_phi(z | u_t,u_T)
                                                       v
                                                latent action z (32-d, 1 query)
                                                       |
 u_t --------------------------------------------------+----> LaWM decoder
                                                            -> predicted future latent feature
 auxiliary: EEF state s_t + z -> MLP -> s_T

Stage 2: LaWAM policy training / inference

 RGB obs + instruction + query tokens
          |
          v
 Qwen3-VL first 16 layers
          |
          +--> latent-action query mapper -> z_hat
                                            |
 current DINO feature u_t ------------------+--> LaWM decoder -> u_hat_T
          |
          +------------------------------+
                                         v
             Alternate-DiT flow action expert
             context stream: VLM hidden states
             dynamics stream: (u_t, u_hat_T)
                                         |
                                         v
                         action chunk, padded max 50, valid by horizon_sec * Hz
```

### 3.4 训练/推理信息流差异

| 阶段 | 信息流 | 差异点 |
|---|---|---|
| LaWM training | 有未来 observation `o_T`；encoder 用 `(u,u_T)` 得到 teacher latent action `z`；decoder 学 `u,z -> u_T` | 未来特征可见；辅助 state loss 可用。论文 p.4, `example.tex:239-254`。 |
| LaWAM policy training | policy 从当前 observation + language 预测 `\hat z`；LaWM 解码 `\hat u_T`；同时用 teacher `z` 做 distillation，用 GT `u_T` 做 subgoal loss，用 flow matching 学 action | `L = L_act + lambda_distill L_distill + lambda_wm L_wm`；论文 p.4-5, `example.tex:273-284`；代码总 loss `lawam.py:777-780`。 |
| Inference | 无未来 observation；policy 预测 `\hat z`，LaWM 单次前向得到 `\hat u_T`，flow head 用 10 denoising/ODE steps 输出 action chunk | 不做 pixel future generation；论文 p.3-5, `example.tex:165`, `305`; 代码 `predict_action`：`lawam.py:808-833`。 |

## 4. 训练数据表

| 阶段/数据 | 来源、任务、embodiment | 数量 | 模态与频率 | action/state | 清洗/增强/处理 | 公开性与证据 |
|---|---|---:|---|---|---|---|
| LaWM pretraining robot videos | open-source robot data，论文引用 EgoDex/Ego4D/LEGO/AgiBot/RoboMIND/RoboCOIN/Open X-Embodiment/DROID 等，但未逐项列明实际纳入子集与比例 | roughly `3,000h` robot videos | `num_frames=2`, image `256x256`; robot teleoperation horizon `1.2s` | `max_state_dim=32`; latent action `code_dim=32` | DINO 编码前对 encoder/decoder views 使用不同 random crops/color augmentations，encoder clip 内 temporally consistent；代码配置 `image_aug=true`, `dual_view_aug=true` | 论文 p.4, p.15：`example.tex:295-296`, `458`; release YAML `dino_large_vae.yaml:21-35`, `63-72`。逐数据源小时/episode/帧数/采样比例：未披露。 |
| LaWM pretraining egocentric human videos | egocentric human videos；不用于 policy integration，因为通常缺少机器人任务语言描述 | roughly `1,500h` | human horizon `0.4s`; image `256x256` | 无机器人 action/state labels；作为 dynamics prior | 同上 | 论文 p.4, p.15：`example.tex:295-296`, `458`, `461`。逐数据源细节未披露。 |
| LaWAM policy-integration pretraining | robot trajectories with explicit language instructions | 样本/小时/episode 未披露；训练 `200k steps`, global batch `1024` | RGB-only, `256x256`; mixed-frequency data 保持 native control frequency，使用 physical-time encoding | EEF action labels；policy/eval 不提供 proprioceptive state | latent-action distillation + subgoal supervision + action flow matching；human videos 不参与此阶段 | 论文 p.14-16：`example.tex:424`, `426-438`, `461`。数据混合比例未披露。 |
| LIBERO SFT | four suites: LIBERO-Spatial/Object/Goal/Long；Franka | 训练 demo 数未披露；评测 `2,000 trials` across `40 tasks` | official release `libero_merged_no_noops_20hz`; code image `256`, `num_frames=2`, `sec_chunk=0.4`, `pyav`; data name indicates `20Hz` | code keys: primary/wrist RGB; state/action `eef_position`, `eef_orientation`, `gripper`; action indices `0..7`; model action padded to `32` | remove failed demonstrations following OpenVLA setup；no primary video aug in release config；LeRobot 3.0 conversion; val_tail_ratio `0.01` | 论文 p.16：`example.tex:479`; README `337-352`; code `train_libero.yaml:55-68`, `data_config.py:521-542`, `571-595`。raw state/action exact dims come from dataset metadata，未在论文卡片披露。 |
| RoboTwin SFT | RoboTwin 2.0, coordinated bimanual manipulation, over `50 tasks` | `2,500` clean demos + `25,000` randomized demos | `robotwin_merged`; code image `256`, `num_frames=2`, `sec_chunk=1.2`, `pyav` | code keys: `cam_high`, left/right wrist cameras; left/right EEF position, quaternion orientation, gripper; action indices `0..15`; model action padded to `32` | derived from lingbot-va / robbyant release; converted to LeRobot 3.0; no primary video aug in release config | 论文 p.16：`example.tex:483`; README `387-403`; code `train_robotwin.yaml:53-64`, `data_config.py:1042-1068`, `1095-1125`。 |
| Real-world SFT | Franka pick-and-place, Franka drawer opening, Quanta X1 towel folding | `150` demos for each Franka task; `280` demos for towel folding; eval `30` trials per task | external RGB camera for Franka; Quanta X1 bimanual robot; detailed camera resolution/fps/control Hz 未披露 | action/state dimension 未披露 | fixed initial configurations across methods; seen + unseen conditions | 论文 p.16-17：`example.tex:499-505`, `507-517`。 |

## 5. 训练 Recipe 和超参数表

| 阶段 | 初始化与训练模块 | Loss | Optimizer / LR / scheduler | batch/steps/precision/并行 | 证据 |
|---|---|---|---|---|---|
| LaWM / LAM pretraining | Frozen DINOv3 ViT-B/16 feature encoder；训练 LAM encoder/decoder + auxiliary state predictor；训练后保留 decoder 为 LaWM，posterior encoder 只用于 teacher latent action | `L_lam = L_wm + beta L_KL + lambda_aux L_aux`；`L_wm=||u~_T-u_T||_2^2`, `L_aux=||g(s,z)-s_T||_2^2`; paper `beta=1e-5`; release YAML `lambda_aux=1.0`, `loss_type=smooth_l1`, `state_loss_type=l2` | AdamW; paper LR `3e-4`, WD `1e-2`; release YAML same；warmup `10000` code | paper: `16 H100`, `100k steps`, global batch `1024`; release YAML batch `64` per process, `bf16-mixed`, grad clip `1.0`, DDP false-unused, val interval `10000` | 论文 p.4, p.15：`example.tex:245-254`, `458`; YAML `dino_large_vae.yaml:14-35`, `53-58`, `72-93`。 |
| LaWAM policy-integration pretraining | init from Qwen3-VL + LaWM/LAM; robot trajectories with language only；human video only through LaWM prior | `L = L_act + lambda_distill L_distill + lambda_wm L_wm`; paper `lambda_distill=lambda_wm=0.1`; KI prevents action-expert gradients overwriting LaWM | action expert LR `1e-4`; all other modules LR `3e-5` | paper: `200k steps`, `64 H100`, global batch `1024`; precision not explicitly in paper; code training launcher uses Accelerate bf16 DDP | 论文 p.4-5, p.15-16：`example.tex:273-284`, `461`; Accelerate config `ddp_bf16.yaml`。optimizer betas/eps/WD for this pretrain未披露。 |
| LIBERO SFT | init `results/Checkpoints/pretrain/lawam_pretrain/final_model/pytorch_model.pt`; load pretrained policy flow；Qwen3-VL base path + LaWM checkpoint path in config；freeze policy keeps first 16 LLM layers, freezes embedding and last LLM layer, unfreezes vision merger and LAM decoder | code total loss = flow loss + `0.1 * loss_perceptual + 0.1 * loss_distill`; flow matching MSE on velocity | AdamW betas `(0.9,0.95)`, eps `1e-8`, WD `1e-8`; LR `1e-4` for flow/LaWM decoder/VLM; cosine_with_min_lr, min LR `5e-7`, warmup `1500` | `25k steps`; paper global batch `256`; config per-device batch `32`, grad accumulation `1`，因此 8 GPUs 与 Accelerate config 可得 global `256`; bf16 DDP; grad clip `1.0`; eval every `500` steps | 论文 p.16：`example.tex:479`; code `train_libero.yaml:69-112`, `ddp_bf16.yaml`; loss code `lawam.py:721-780`; flow loss `flowmatching_expert.py:558-567`。 |
| RoboTwin SFT | init same pretrain checkpoint; load pretrained policy flow | same; `perceptual_weight=0.1`, `lam_encoder_distill_weight=0.1` | AdamW betas `(0.9,0.95)`, eps `1e-8`, WD `1e-8`; flow LR `1e-4`; LaWM decoder LR `3e-4`; VLM LR `1e-4`; cosine_with_min_lr min `5e-7`, warmup `2000` | paper: `100k steps`, global batch `1024`, `64 H100`, about `20h`; config per-device `16`, grad accumulation `2` => with 32 GPUs gives 1024, but paper says 64 H100 so per-device effective setup may differ from release config; bf16 DDP; grad clip `1.0` | 论文 p.16：`example.tex:483`; code `train_robotwin.yaml:65-109`; README notes global batch `1024` required：`README.md:420-428`。 |
| Inference | released LIBERO/RoboTwin SFT checkpoints | no training loss | flow CFG scale `1.0`, `num_inference_steps=10`; Beta time schedule only training-side | latency measured with `1,000` repeated action-chunk predictions on A100, model-only; reported `187 ms` | 论文 p.1, p.16：`example.tex:102-103`, `465`; configs `train_libero.yaml:45-52`, `train_robotwin.yaml:44-51`。 |

未披露或无：EMA 未披露；dropout 多处代码为 `0.0` 但全模型 dropout 未系统披露；FSDP/DeepSpeed/ZeRO 未使用或未披露，release launcher 为 Accelerate DDP bf16。

## 6. Benchmark 性能表

| Benchmark / task | 协议 | LaWAM | 主要 baseline | 公平性备注 |
|---|---|---:|---|---|
| LIBERO four suites | 4 suites, `40 tasks`, `50 trials/task`, total `2,000 trials`; train on Spatial/Object/Goal/Long; remove failed demos | Long `97.0`, Goal `98.4`, Object `99.6`, Spatial `99.4`, Avg `98.6`; model size `2.3B`; latency `187 ms` | OpenVLA-OFT avg `97.1`; pi0.5 avg `96.9`, `220 ms`; GR00T-N1.6 avg `97.0`, `259 ms`; Cosmos-Policy avg `98.5`, `1413 ms`; LingBot-VA avg `98.5`, `4482 ms`; Fast-WAM avg `97.6`, `486 ms` | Table 1 mixes original-paper numbers and strongest reproduced numbers；paper excludes video-diffusion VAE/text encoder from WAM parameter counts，可能低估 pixel-WAM full deploy footprint。证据：Table 1 `tab_libero_comparison.tex:29-48`, p.16 `example.tex:465`, `479`。 |
| RoboTwin 2.0 | `50 tasks`; average over `100 trials/task`; clean and randomized settings | Clean `92.64`, Rand `89.80`, overall abstract reports `91.22` | Fast-WAM `91.98/90.52`; GigaWorld `86.36/85.04`; LingBot-VA `91.50/90.92`; pi0.5 `82.74/76.76`; Motus `88.66/87.02` | Fast-WAM/LingBot-VA re-evaluated from open weights on H100；其他 baseline 取自原论文，数据/协议一致性依赖原文。证据：Table 2 `tab_robotwin_lawam_sota.tex`; p.6 `example.tex:318`; p.16 `example.tex:483`。 |
| Real-world pick/open/fold | 3 tasks, `30 trials/task`; Franka for pick/open, Quanta X1 for towel; fixed initial configurations across methods | Pick `93.3`, Open Drawer `86.7`, Fold Towel `90.0`, Avg `90.0` | pi0.5 avg `83.3`; GR00T-N1.6 `68.9`; Fast-WAM `63.3`; LingBot-VA `53.3` | 同一物理初始配置，较公平；但真实训练数据量、采集细节、控制频率、硬件延迟除论文描述外未完全公开。证据：Table 3 `example.tex:345-358`; protocol `example.tex:499-517`。 |
| LaWM dynamics analysis | 500 LIBERO trajectories for feature cosine rollout analysis | qualitative/curve evidence：rollout remains close to true future and moves away from initial | no numeric benchmark table | 证明 dynamics behavior，但不是 task success metric。证据：p.17 `example.tex:522-530`。 |

## 7. 消融与局限

### 7.1 消融

| 消融 | 结果 | 数值可读性 | 证据 |
|---|---|---|---|
| w/o WM / remove LaWM | 最大下降，尤其 LIBERO-Long；说明 explicit latent subgoal conditioning 是主要收益来源 | Figure 5 only; 未给精确表格数值。渲染图可读大致 Avg 从 98 左右降到 95.5 左右，但报告不把近似值当精确数字 | 论文 p.8, `example.tex:383-388`; `picture/ab.pdf`。 |
| w/o distill | 明显下降；说明 policy 需要 LAM posterior supervision 才能稳定驱动 LaWM | 精确数值未披露 | 同上。 |
| w/o KI & distill | 比单独 w/o distill 更差；说明 LaWM 既需要 aligned latent actions，也需要 Knowledge Insulation 保护 | 精确数值未披露 | 同上。 |
| w/o pretrain | 小但一致下降；benchmark-specific finetune 可恢复部分但不是全部 latent-action alignment | 该句在 TeX 注释中，正文主要图示；精确数值未披露 | 图 `picture/ab.pdf`; TeX 注释 `example.tex:389` 不作为正文主张。 |
| mixed-frequency physical-time encoding | 5/10/20 Hz LIBERO controlled co-training 中，w/o pos 明显低于 w/ pos；w/ pos 接近 only 20Hz upper-bound | Figure only；精确表格数值未披露 | 论文 p.15, `example.tex:442-446`, `picture/mix_hz.pdf`。 |

### 7.2 提升来源判断

论文与消融共同指向：主要提升来自架构接口（LaWM 将 latent action 扩展成 spatial latent visual subgoal，并直接条件化 action expert），其次是 latent-action distillation / KI / physical-time encoding。数据与预训练也重要，但论文没有提供足够 ablation 把 `3,000h+1,500h` 预训练数据量、Qwen3-VL 初始化、DINO latent space 分别拆开量化。

### 7.3 局限与依赖

| 类型 | 内容 |
|---|---|
| 数据依赖 | LaWM pretraining mixture 的逐源数据、采样比例、过滤规则未正式披露；真实任务 demos 公开性未披露。 |
| 算力依赖 | LaWM paper config 需要 `16 H100 * 100k steps`; policy integration pretrain 需要 `64 H100 * 200k steps`; RoboTwin SFT 需要 `64 H100` 约 `20h`。 |
| 硬件/仿真依赖 | LIBERO、RoboTwin simulator 环境需单独安装；真实任务依赖 Franka/Quanta X1，无法仅凭开源仓库完整复现。 |
| 方法局限 | camera motion dominant 的 egocentric videos 中 LaWM 可能难以学 coherent latent action space；fine-grained cloth deformation 数据稀缺导致 deformable manipulation 仍有限。论文 p.8, p.17：`example.tex:392`, `517`。 |
| 代码/论文不一致 | 论文 p.14 写 LaWM encoder/decoder 都是 24-layer；release `dino_large_vae.yaml` 为 `enc_layers=24`, `dec_layers=12`。工程复现应按 release checkpoint/config。 |

整体可复现性评级：**中**。理由是 SFT/eval 路径、权重、LIBERO/RoboTwin 预处理数据、训练脚本已开源；但从零复现论文主结果依赖未完整披露的 LaWM/policy pretraining 数据混合与 64 H100 级算力。

## 8. 最小复现步骤

下面是面向工程 smoke test / SFT 复现的最短路径，而不是从零重训 LaWM 的完整论文级复现。

1. 准备环境  
   ```bash
   git clone https://github.com/RLinf/LaWAM.git LaWAM
   cd LaWAM
   conda create -n lawam python=3.10 -y
   conda activate lawam
   pip install -r requirements.txt
   pip install flash-attn==2.8.3 --no-build-isolation
   pip install -e .
   ```
   证据：README `Environment Setup`，`README.md:55-83`。

2. 下载基础模型与 LaWM/LAM  
   ```bash
   hf download Qwen/Qwen3-VL-2B-Instruct --local-dir results/Checkpoints/qwen3_weights
   hf download facebook/dinov3-vitb16-pretrain-lvd1689m --local-dir weights/dinov3-vitb16-pretrain-lvd1689m
   hf download jialei02/lawam_lam --local-dir latent_action_model/logs/dino_large_vae/lam_release
   ```
   证据：README `README.md:104-145`。

3. LIBERO 推理 smoke test  
   下载 `jialei02/lawam_libero_sft_release`，安装 LIBERO simulator，运行：
   ```bash
   SUITES="libero_10 libero_goal libero_object libero_spatial" \
   NUM_TRIALS_PER_TASK=50 \
   NUM_WORKERS=4 \
   GPU_IDS="0 1 2 3" \
   bash examples/LIBERO/eval_files/auto_eval_scripts/run_libero_benchmark.sh "$CKPT_PATH"
   ```
   证据：README `README.md:171-226`。

4. LIBERO SFT 复现  
   ```bash
   hf download jialei02/libero_merged_no_noops_20hz \
     --repo-type dataset --local-dir dataset/libero_merged_no_noops_20hz
   hf download jialei02/lawam_pretrain --local-dir results/Checkpoints/pretrain/lawam_pretrain
   bash train_lawam.sh --run_id libero_sft_from_pretrain
   ```
   关键配置：25k steps、global batch 256、cosine LR、bf16 DDP；证据：`train_libero.yaml:55-112`，论文 p.16。

5. RoboTwin SFT / eval  
   ```bash
   hf download jialei02/robotwin_merged \
     --repo-type dataset --local-dir dataset/robotwin_merged
   bash train_lawam.sh starVLA/config/training/train_robotwin.yaml \
     --run_id robotwin_sft_from_pretrain
   ```
   关键配置：100k steps、paper global batch 1024、64 H100 about 20h；证据：`train_robotwin.yaml:53-109`, `README.md:420-452`, 论文 p.16。

6. 从零复现 LaWM + policy pretrain  
   需要重建未完全公开的 open-source robot + egocentric human video mixture；LaWM paper config 为 16 H100/100k/global 1024；policy integration 为 64 H100/200k/global 1024。由于逐数据源 recipe 未披露，此路线无法严格复现论文 pretrain，只能近似复现。

## 9. 未披露信息清单

| 类别 | 未披露项 |
|---|---|
| 数据 | LaWM 预训练逐数据源小时/episode/轨迹/帧/token 数、采样比例、过滤规则、每源相机/分辨率/fps/control Hz、公开/未公开边界；policy-integration pretraining 的机器人数据清单和比例。 |
| 数据卡 | HF dataset cards 在 README 中只说明来源转换到 LeRobot 3.0；episode 数、文件大小、per-task 分布、dataset license 细节未在 LaWAM README 中系统列出。 |
| 模型 | Qwen3-VL 截断/冻结的精确所有参数组数量；Action expert FFN/intermediate size；完整 parameter breakdown；显存占用。 |
| 训练 | policy-integration pretrain 的 AdamW betas/eps/WD、warmup、scheduler、gradient clip、precision；LaWM 实际 100k steps 与 release YAML `max_epochs=10` 的完整换算；EMA 未披露。 |
| 推理 | 控制频率/robot execution latency、缓存策略、端到端系统延迟、显存、FPS；paper 仅披露 A100 上 model-only 187 ms/chunk。 |
| 评测 | LIBERO/RoboTwin 随机种子；baseline 是否全部同输入/同数据/同预训练未完全统一；真实任务采集设备、控制接口、失败判据细节未完全披露。 |
| 真实硬件 | Franka/Quanta X1 控制频率、动作空间精确定义、相机分辨率与标定、真实数据是否公开。 |

## 10. 主要一手资料

- Paper: [arXiv:2606.15768](https://arxiv.org/abs/2606.15768), PDF/source v1.
- Project: [https://rlinf.github.io/LaWAM/](https://rlinf.github.io/LaWAM/)
- Code: [https://github.com/RLinf/LaWAM](https://github.com/RLinf/LaWAM), inspected commit `1add20a376126eacab02f19a62d726072a322cae`.
- Checkpoints/data listed in official README: `Qwen/Qwen3-VL-2B-Instruct`, `facebook/dinov3-vitb16-pretrain-lvd1689m`, `jialei02/lawam_lam`, `jialei02/lawam_pretrain`, `jialei02/lawam_libero_sft_release`, `jialei02/lawam_robotwin_sft_release`, `jialei02/libero_merged_no_noops_20hz`, `jialei02/robotwin_merged`.

