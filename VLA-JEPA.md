# VLA-JEPA

## 1. 核心结论（200字以内）

VLA-JEPA 是一个面向 VLA 预训练的 latent world model / JEPA 框架，核心是“无泄漏未来状态预测”：未来帧只经冻结 V-JEPA2 target encoder 产生 latent target，不作为 VLM 输入；Qwen3-VL-2B 通过 latent action tokens 条件化自回归 world predictor，并用 flow-matching DiT 动作头输出连续动作。论文在 LIBERO、LIBERO-Plus、SimplerEnv 和 Franka 真实任务上显示鲁棒性收益，尤其 LIBERO-Plus 平均成功率 79.5，高于无 human video 的 62.9（论文表3，第8页）。但公开仓库只标注“Partial training code”，未公开完整训练数据、真实数据、checkpoint 训练日志和若干精确协议；工程复现评级：中偏低。

## 2. 研究对象确认

| 项目 | 结论 | 证据 |
|---|---|---|
| 准确标题 | *VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model* | arXiv 页面；PDF 第1页 |
| 作者 | Jingwen Sun, Wenyao Zhang, Zekun Qi, Shaojie Ren, Zezhi Liu, Hanxin Zhu, Guangzhong Sun, Xin Jin, Zhibo Chen | PDF 第1页；arXiv 页面 |
| 机构 | USTC；Zhongguancun Academy；Shanghai Jiao Tong University；Tsinghua University；Eastern Institute of Technology, Ningbo；UCAS；Nankai University | PDF 第1页 |
| 版本/日期 | arXiv 页面显示提交日期 2026-02-10；本地下载 PDF 页眉为 `arXiv:2602.10098v2 [cs.RO] 14 Feb 2026`；PDF 封面日期为 2026-02-17 | arXiv 页面；PDF 第1页 |
| 论文 | https://arxiv.org/abs/2602.10098 | PDF 第1页 |
| 项目主页 | https://ginwind.github.io/VLA-JEPA/ | PDF 第1页；项目页 |
| 官方代码 | https://github.com/ginwind/VLA-JEPA | PDF 第1页；README |
| 权重 | https://huggingface.co/ginwind/VLA-JEPA；目录含 `LIBERO`、`SimplerEnv` checkpoint/config/statistics | HF 模型页；README 评测部分 |
| 数据集入口 | README 列出 ssv2、Droid、LIBERO、BridgeV2、Fractal 下载链接 | `README.md` 第88-96行 |
| 许可证 | 仓库 `pyproject.toml` 声明 MIT classifier 且 `license={file="LICENSE"}`，但 GitHub tree 未包含 LICENSE 文件；HF 页面正文未显示可核验 license 字段。结论：许可证状态不完整，代码/权重实际授权范围需向作者确认 | `pyproject.toml`；GitHub tree；HF 页面 |
| 同名/近名风险 | 存在近名论文 *JEPA-VLA: Video Predictive Embedding is Needed for VLA Models*，arXiv:2602.11832；不是本报告对象 | arXiv 搜索结果 |

## 3. 核心方法与范式

**论文明确说明。** VLA-JEPA 属于 VLA 框架，同时引入 latent world model / JEPA-style latent predictive alignment。它要解决 latent-action pretraining 中未来信息泄漏、pixel-level 目标偏向外观、真实视频 nuisance motion、以及多阶段训练复杂的问题（论文第1-3页）。关键做法是：target encoder 用未来帧产生未来 latent state；student path 只看当前观测/历史 latent state 和 VLM 产生的 latent action tokens；world predictor 在 latent space 对齐未来状态（论文式4-5，第4-5页）。

**输入/输出模态。**

| 模态 | 训练输入 | 推理输入 | 输出 | 证据 |
|---|---|---|---|---|
| 视觉 | 多视角图像/视频；VLM 输入 resize 224x224，world-state encoder 视频 resize 256x256 | `batch_images: List[List[PIL.Image]]`，多视角 | latent action tokens、embodied action token hidden states | 论文附录 A.2 第16-17页；`VLA_JEPA.py` 第278-335行 |
| 语言 | instruction / video text description | instruction | 参与 Qwen3-VL token 序列 | `README.md` 第117-121行；`VLA_JEPA.py` 第307-311行 |
| 状态 | robot state，可选；公开配置 `state_dim=8` | 可选 `state` | action head conditioning | `scripts/config/vlajepa_cotrain.yaml`；`LayerwiseFM_ActionHeader.py` 第233-237行 |
| 动作 | robot action sequence；`action_dim=7` | 无 ground truth | normalized continuous actions `[B,T,7]` | 论文表6 第16页；`LayerwiseFM_ActionHeader.py` 第329-385行 |
| 未来表示 | 训练时未来帧经冻结 V-JEPA2 encoder 产生 target state | 推理时不使用未来帧 | world-model loss only | 论文第3-5页；`VLA_JEPA.py` 第220-244行 |

### ASCII 架构图

```text
Training on video / robot demos

current frames + instruction
        |
        v
  Qwen3-VL-2B VLM  -- emits <latent_i> tokens and <embodied_action> tokens
        |                              |
        | latent action tokens          | embodied action token sequence x32
        v                              v
 V-JEPA2 latent world predictor       Flow-matching DiT-B action head
        |                              |
 predict future latent states          predict velocity field over action chunk
        |                              |
 L1/LJEPA vs frozen V-JEPA2 targets    MSE flow-matching loss
        |                              |
 future frames -> frozen V-JEPA2 encoder (target only, never VLM input)

Inference

images + instruction + optional state -> Qwen3-VL -> embodied action tokens
                                      -> Flow head Euler steps x4
                                      -> 7-step normalized continuous actions
```

## 4. 模型架构表

| 模块 | 配置/参数 | 来源与状态 |
|---|---:|---|
| VLM backbone | Qwen3-VL-2B，dense Transformer；VLA-JEPA config 使用 hidden size 2048 | 论文附录 A.1 第16页；`scripts/config/vlajepa_cotrain.yaml` 第11-14行 |
| Qwen3-VL text config | 28 layers；hidden 2048；FFN/intermediate 6144；16 attention heads；8 KV heads；head dim 128；max position 262144；dtype bf16 | Qwen3-VL-2B-Instruct `config.json` 第366-395行 |
| Qwen3-VL vision config | ViT + 3D conv；24 depth；hidden 1024；out hidden 2048；FFN 4096；16 heads；patch 16；temporal patch 2 | 论文附录 A.1；Qwen3-VL config 第400-418行 |
| Latent world model | 自回归 Transformer；12 layers；8 heads；image token dim 2048；每时刻 256 image tokens；action token dim 2048；每时刻 3 action tokens；2 views；future video horizon 8 | 论文表5，第16页 |
| 公开代码 world predictor | `VisionTransformerPredictorAC`；默认 predictor dim 1024、MLP ratio 4；构造时 `depth=cfg.depth`、`num_heads=cfg.num_heads`、`embed_dim=V-JEPA hidden*2`、`num_add_tokens=cfg.num_action_tokens_per_timestep` | `vj2_predictor.py` 第17-119行；`VLA_JEPA.py` 第81-94行 |
| Latent/action special tokens | `<|action_{}|>`、`<|embodied_action|>`；embodied action token repeated 32；代码中 `max_action_tokens=action_horizon*4` | 论文附录 A.1 第16页；`VLA_JEPA.py` 第62-100行 |
| 公开 YAML latent tokens | `num_action_tokens_per_timestep=8`，`num_embodied_action_tokens_per_instruction=32`，`num_frames=8` | `scripts/config/vlajepa_cotrain.yaml` 第43-49行 |
| Action head | Flow matching DiT-B；16 layers；12 heads；token dim 1024；state dim 8；action dim 7；future action horizon 7；learnable pos encoding；denoising timesteps 4 | 论文表6，第16页 |
| 公开代码 action head | `FlowmatchingActionHead`：Beta(1.5,1.0) timestep；noisy trajectory `(1-t)*noise+t*actions`；target velocity `actions-noise`；loss 为 MSE；推理 Euler integration 4 steps | `LayerwiseFM_ActionHeader.py` 第255-261、267-325、329-385行 |
| 动作表示 | VLA-JEPA 使用 end-effector delta position + delta axis-angle；gripper 二值化；代码 config `action_dim=7` | 论文附录 A.2 第16页；`scripts/config/vlajepa_cotrain.yaml` 第21-24行 |
| 归一化 | 连续动作 min-max normalize 到 `[0,1]`；gripper `{0,1}` | 论文附录 A.2 第16页 |
| 预训练/冻结 | 论文：训练所有参数，除 world state encoder；V-JEPA2 encoder checkpoint + randomly initialized predictor | 论文第6页、附录 A.1 |
| 并行/精度 | 8 GPU；DeepSpeed ZeRO-2；bf16；gradient clipping 1.0；gradient checkpointing true | 论文第7页；`deepspeed_zero2.yaml`；`ds_config.yaml`；`scripts/config/vlajepa_cotrain.yaml` |
| 参数量 | VLA-JEPA 总参数量未披露；backbone 名称为 2B；world/action head 参数量未披露 | 论文/代码未给出总量；禁止补算 |

**代码差异提示。** 论文表5写“每时刻 3 action tokens”，公开 YAML 写 `num_action_tokens_per_timestep=8`；论文附录还写 `K=24/T` 且 T=8 时 K=3。工程复现应优先遵循论文表5/附录，或明确记录使用公开 YAML 会改变 latent token 数。

## 5. 训练数据表

| 阶段 | 数据集 | 来源/任务/embodiment | 规模 | 模态/视角/分辨率 | action/state | 处理与采样 | 公开性 |
|---|---|---|---:|---|---|---|---|
| Human-video pretraining | Something-Something-v2 / ssv2 | 人类动作视频；action-free | 220K human videos | 训练 VLM image 224x224；world-state video 256x256；fps 未披露 | 无真实 robot action | 论文 Eq.5；README 配置 `video_dir` + headerless CSV text file；公开 YAML video batch 16/GPU | 公开数据；论文实际 subset/过滤未披露 |
| Robot pretraining / cotrain | DROID | action-labeled robot demos | 76K high-quality demonstration trajectories | 论文未披露相机数/分辨率/fps；代码 modality 2 views: exterior + wrist | state 8；action 7 in modality；论文说 VLA-JEPA 用 delta EE + axis-angle | Eq.9；LeRobot v2.1；`modality.json` under meta | DROID 公开；具体清洗未披露 |
| Simulation finetune | LIBERO | Franka Emika Panda；4 suites | approximately 2K expert demonstrations | 论文未披露相机数/fps；VLM 224，world 256 | LIBERO config 可用 `delta_qpos`；论文称 VLA-JEPA end-effector control | 从 pretrained checkpoint 继续 30K steps；不使用 LIBERO-Plus augmented data | 公开 |
| SimplerEnv post-training | Fractal | Google Robot embodiment | 论文未披露本实验使用规模 | 代码 modality 单 `observation.images.image` | state 8；action 7 | 对应 SimplerEnv Google Robot；README 使用 LeRobot | 公开 |
| SimplerEnv post-training | BridgeV2 | WidowX embodiment | 论文未披露本实验使用规模 | 论文未披露 | action/state 维度按 modality/config；未披露更多 | 对应 SimplerEnv WidowX | 公开 |
| Real-world finetune | 自采 Franka 数据 | Franka Research 3 + Robotiq 2F-85；水果 pick/place | 100 demonstrations，3 tasks | 3 Intel RealSense D435：2 third-person + 1 wrist | 论文未披露维度；推测同 action head 7D，但未明确 | 从 pretrained checkpoint 继续 20K steps | 未公开 |

未披露项包括：每个公开数据集在本实验中的小时数、episode 过滤规则、帧率、控制频率、多数据集混合比例、采样权重、真实数据 raw logs、数据增强细节、DROID/Fractal/BridgeV2 实际训练子集。

## 6. 训练 Recipe 和超参数表

| 阶段 | 初始化 | 训练/冻结模块 | Loss | Optimizer/LR | Batch/steps | 精度/并行 | 证据 |
|---|---|---|---|---|---|---|---|
| JEPA human-video pretraining | Qwen3-VL-2B；V-JEPA2 encoder；world predictor 随机初始化 | 训练所有参数，除了 world state encoder | Eq.5 latent world prediction；代码 `F.l1_loss(predicted_states, gt_states)` | AdamW；betas `[0.9,0.95]`；eps `1e-8`；代码 weight decay `1e-8`；base LR `3e-5`，VLM/world `1e-5`，action head `1e-4`；cosine + warmup 5000 | 论文：batch 32/GPU x8=256；SSV2+DROID pretrain 50K steps；代码 cotrain VLA batch 16/GPU + video batch 16/GPU | bf16；DeepSpeed ZeRO-2；grad clip 1.0；gradient accumulation 1；checkpointing true | 论文第6-7页、附录 A.2；`vlajepa_cotrain.yaml`；`train_vlajepa_cotrain.py` 第93-119、172-178、284-323、380-411行 |
| Robot cotrain / pretraining | 上述 checkpoint | 同上；可同时 robot action + human video | `L = LFM + beta LWM`；论文 beta tunable；代码返回 `action_loss` + `wm_loss*0.1`，trainer 直接求和 | 同上 | 与 pretraining 合计 50K steps | 同上 | 论文式9 第5页；`VLA_JEPA.py` 第240-274行 |
| LIBERO / simulation finetune | last pretrained checkpoint | 论文未披露冻结差异；公开 config `freeze_modules: ''` | action + world loss；若 batch 含 video target | LR 同公开 YAML；warmup 5000；max grad 1.0 | 30K steps；batch 32/GPU x8=256（论文）；robot_ft YAML per-device batch 32 | bf16；ZeRO-2 | 论文第7页、附录 A.2；`vlajepa_robot_ft.yaml` |
| Real-world finetune | last pretrained checkpoint | 未披露 | 未披露，推测同 robot ft，但论文未明确 | 未披露，公开 real config 未给 | 20K steps；100 demos | 8 A100 论文总体设置 | 论文第7页、附录 A.2 |

**未披露/不一致。** EMA、dropout 仅 action head YAML 给出 0.2；完整 VLM dropout 未披露。论文未披露训练时间、显存、checkpoint 保存策略、随机种子（代码 seed=42）、样本/token 数、是否用 FSDP。公开脚本使用 DeepSpeed ZeRO-2；仓库也有 ZeRO-3 配置但主脚本调用 `deepspeed_zero2.yaml`。

## 7. Benchmark 性能表

| Benchmark | 协议 | VLA-JEPA | 主要对比 | 公平性备注 |
|---|---|---:|---|---|
| LIBERO | 4 suites；每 task 50 episodes；每 suite 500 episodes；success rate | Spatial 96.2；Object 99.6；Goal 97.2；LIBERO-10 95.8；Avg 97.2 | OpenVLA-OFT Avg 97.1；pi0.5 Avg 96.9；UniVLA 95.2 | 论文指出 OpenVLA-OFT/pi0.5 使用 extensive robot datasets；VLA-JEPA 使用更少训练数据。比较有数据规模差异 |
| LIBERO ablation | 同上 | w/o human videos Avg 96.1 | full 97.2 | human video 对 ID LIBERO 提升有限 |
| SimplerEnv Google Robot | visual matching；各 task 平均成功率 | Pick 88.3；Move 64.1；Drawer 59.3；Place 49.1；Avg 65.2 | RoboVLMs Avg 51.7；villa-x Avg 44.9 | 表2注明除 LAPA* 外方法训练于 OXE subsets；数据子集/质量差异较大 |
| SimplerEnv WidowX | visual matching | Spoon 75.0；Carrot 70.8；Block 12.5；Eggplant 70.8；Avg 57.3 | LAPA* Avg 57.3；pi0-Fast Avg 48.3；villa-x Avg 40.8；UniVLA 42.7 | LAPA* 使用 SimplerEnv in-distribution successful rollouts；不完全公平 |
| SimplerEnv ablation | 同上 | w/o human videos Google Avg 78.4；WidowX Avg 57.3 | full Google Avg 65.2；WidowX Avg 57.3 | 论文解释高质量 expert demos 对 ID/real-to-sim gap 更关键 |
| LIBERO-Plus | 7 perturbation dimensions；每维含原 LIBERO 4 suites；success rate | Camera 63.3；Robot 67.1；Language 85.4；Light 95.6；Background 93.6；Noise 66.3；Layout 85.1；Avg 79.5 | OpenVLA-OFT Avg 69.6；pi0-Fast 61.6；pi0 53.6；UniVLA 42.9；WorldVLA 25.0 | VLA-JEPA 在鲁棒性/OOD 上优势明显；baseline 预训练数据不同 |
| LIBERO-Plus ablation | 同上 | w/o human videos Avg 62.9 | full 79.5 | human video 主要提升 robustness/stability |
| Real-world Franka | 每 task 10 trials；ID、task OOD、layout OOD | ID 0.70；task OOD 0.17；layout OOD 0.47 | pi0: 0.57/0.00/0.37；pi0.5: 0.37/0.20/0.27 | pi0/pi0.5 在同自采 demos 上 finetune；但真实任务少、seed 未披露；Figure 4 视觉读取 |
| Future horizon ablation | LIBERO | T=4 Avg 94.8；T=8 Avg 96.1；T=16 Avg 95.5 | T=8 最好 | 论文表4，第10页；注意表4是在 w/o/full 条件外的 horizon 研究 |

论文/代码未披露：推理延迟、FPS、显存、控制频率、真实部署硬件 GPU、动作发送频率、缓存策略。README 仅说明 LIBERO 4 GPUs 并行、LIBERO-Plus/SimplerEnv 8 GPUs 并行评测。

## 8. 消融与局限

**消融结论。**

| 设计 | 结果 | 解释 |
|---|---|---|
| Human video pretraining | LIBERO Avg 97.2 vs 96.1；LIBERO-Plus Avg 79.5 vs 62.9；SimplerEnv Google full 65.2 vs w/o 78.4 | human videos 对 OOD robustness 明显，对 ID/real-to-sim expert-demo 场景不一定提升 |
| Unified pretraining | 论文称优于 prior two-stage pretraining，并通过 attention map 显示 VLA-JEPA 更关注手/机器人/操作物体 | 定量主要来自表1-3；attention map 为定性 |
| Future video horizon | T=8 Avg 96.1 最好；T=4 信息不足，T=16 冗余 | 论文表4，第10页 |
| Latent leakage-free target | 论文主张未来帧只作 target，避免 latent action collapse | 核心机制；无直接去泄漏对照表，更多是架构分析 + baseline 对比 |

**局限。**

- 公开仓库 TODO 明确为“Partial training code”，缺完整实验训练栈和日志。
- 代码许可证状态不干净：GitHub 无 LICENSE 文件，`pyproject.toml` 却写 MIT classifier；HF 页面正文未显示可核验 license 字段。
- Qwen3-VL-2B、V-JEPA2 encoder、DROID/SSV2/LIBERO/Bridge/Fractal 均需外部下载，且部分数据体量大；真实 100 demos 未公开。
- 论文未披露控制频率、训练时间、显存、完整采样比例、数据清洗、真实任务全部成功/失败统计细节。
- 公开 YAML 与论文表5 latent action token 数存在差异；工程复现实验需固定一版并报告。
- `VLA_JEPA.py` 第256-258行从 `trainer.repeated_diffusion_steps` 读取重复次数，但 YAML 把 `repeated_diffusion_steps: 8` 放在 `framework.action_model` 下，因此实际代码可能默认 4；这是复现时必须检查的实现风险。

## 9. 最小复现步骤

1. 克隆官方仓库并安装：Python 3.10，`pip install -r requirements.txt`，安装 `flash-attn`，`pip install -e .`（README 环境部分）。
2. 下载 Qwen3-VL-2B-Instruct 与 V-JEPA2 encoder，并把 `framework.qwenvl.base_vlm`、`framework.vj2_model.base_encoder` 改为本地路径（README 第82-104行）。
3. 下载 ssv2、DROID、LIBERO、BridgeV2、Fractal；将 robot datasets 转/放为 LeRobot v2.1，并按 `examples/*/modality.json` 放入 `meta/`（README 第88-96行）。
4. 预训练：用 `scripts/vlajepa_cotrain.sh` 调 `starVLA/training/train_vlajepa_cotrain.py` 和 `scripts/config/vlajepa_cotrain.yaml`；目标 50K steps；8 GPU ZeRO-2；bf16。
5. LIBERO finetune：用 `scripts/vlajepa_robot_ft.sh` 和 `scripts/config/vlajepa_robot_ft.yaml`，从 last pretrained checkpoint 继续 30K steps。
6. SimplerEnv finetune：用 `examples/SimplerEnv/train_files/run_oxe_train.sh` 与 `vlajepa_ft.yaml`，分别准备 Fractal/BridgeV2 对应 embodiment。
7. 评测：下载 HF checkpoint 可先复现实验评测；LIBERO 用 `examples/LIBERO/eval_libero.sh`，LIBERO-Plus 用 `examples/LIBERO-Plus/eval_libero_plus.sh`，SimplerEnv 用 `batch_evaluate.sh` + `calc_success_rate.sh`。
8. 真实机器人复现：需要 Franka Research 3、Robotiq 2F-85、3 个 RealSense D435、自采 100 demos；官方未公开最小可执行采集/部署脚本，复现成本高。

**算力/存储估算。** 论文明确训练使用 8 NVIDIA A100 GPUs；README 评测 LIBERO 4 GPUs、LIBERO-Plus/SimplerEnv 8 GPUs。训练时间、显存峰值、存储空间未披露，不能可靠估算。

## 10. 未披露信息清单

- VLA-JEPA 总参数量、world/action head 参数量。
- 完整训练 wall-clock time、显存、吞吐、FPS、控制频率、部署延迟。
- SSV2/DROID/LIBERO/Fractal/BridgeV2 实际使用 subset、过滤规则、混合比例、采样权重。
- 数据增强、语言指令清洗、token 数、视频 fps。
- Diffusion/flow timestep 分布除公开代码 Beta(1.5,1.0) 外，论文未给完整超参解释。
- EMA、完整 dropout、所有模块冻结/解冻 schedule。
- 随机种子和多次运行方差；论文表格未报告置信区间。
- 真实机器人数据与采集脚本。
- 完整 checkpoint 训练配置与公开 YAML 是否完全对应论文实验。

## 11. 复现性评级

**中偏低。** 论文给出核心架构、关键表格、主要超参，仓库给出可运行方向的部分训练/评测代码与 HF checkpoint；但完整训练数据混合、真实数据、许可证一致性、若干配置差异和工程日志缺失，使“从零严格复现论文数字”风险较高。若目标是“加载官方 checkpoint 复评 LIBERO/SimplerEnv”，可复现性为中；若目标是“从零训练到论文水平”，可复现性为低到中。

## 12. 一手资料链接

- 论文/arXiv: https://arxiv.org/abs/2602.10098
- PDF: https://arxiv.org/pdf/2602.10098
- 项目主页: https://ginwind.github.io/VLA-JEPA/
- 官方代码: https://github.com/ginwind/VLA-JEPA
- 官方权重/模型卡: https://huggingface.co/ginwind/VLA-JEPA
- Qwen3-VL-2B-Instruct config: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/blame/e2378df056d88153dc44616229fa371fcb87e236/config.json
