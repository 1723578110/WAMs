# WAM4D

## 1. 200 字以内核心结论

WAM4D 是一个 4D World Action Model，而不是纯 VLA。它在 LingBot-VA causal video-action backbone 上加入训练期 spatial register tokens，用 Depth Anything 3 几何头监督未来深度，把 4D 几何先验蒸馏进历史视频特征；推理时移除 register/depth/geometric head，仅保留 RGB/action 路径。RoboTwin 2.0 平均成功率 91.8，A800 单卡 action chunk latency 525.43 ms、9.71 GiB（论文表1/9，第8/15页）。但官方代码 URL 当前 404，权重与数据未公开，复现评级：低。

## 2. 研究对象确认

| 项目 | 结论 | 证据 |
|---|---|---|
| 准确标题 | *WAM4D: Fast 4D World Action Model via Spatial Register Tokens* | arXiv 页面；PDF 第1页；`paper.tex:28` |
| 中文题名对应 | “利用空间寄存器令牌的快速4D世界动作模型”是 alphaXiv/中文翻译标题；本报告以英文论文为准 | alphaXiv 指向 arXiv:2606.14048；arXiv 页面 |
| 作者 | Ying Li, Xiaobao Wei, Jiajun Cao, Hao Wang, Xiaowei Chi, Chengyu Bai, Qianpu Sun, Jiajun Li, Xiaojie Zhang, Peidong Jia, Jian Tang, Sirui Han, Shanghang Zhang | arXiv 页面；PDF 第1页；`paper.tex:30-42` |
| 机构 | Peking University；The Hong Kong University of Science and Technology；Beijing Innovation Center of Humanoid Robotics | PDF 第1页 |
| 版本/日期 | v1: 2026-06-12；v2: 2026-07-05；v3: 2026-07-07；PDF 页眉为 `arXiv:2606.14048v3 [cs.CV] 7 Jul 2026`；封面日期 July 8, 2026 | arXiv 页面；PDF 第1页 |
| 论文/TeX | https://arxiv.org/abs/2606.14048；PDF 和 TeX source 均可下载 | arXiv 页面 |
| 官方代码 | 论文声明 `https://github.com/myendless1/wam4d` | PDF 第1页；`paper.tex:57` |
| 代码实际可访问性 | 当前 GitHub API/raw/clone 均返回 Not Found 或连接失败；无 README/配置/许可证可核对 | 本次检索；GitHub API 404 |
| 项目主页 | 未披露 | 论文和 arXiv 页面未给 |
| 模型权重 | 未披露/未发现官方权重卡 | arXiv 页面和论文未给 |
| 数据集地址 | 未披露 WAM4D 专用数据地址；论文仅说明使用 RoboTwin 2.0、自采 AstriBot S1 数据、Depth Anything 3 伪深度流程 | `paper.tex:450-460` |
| 许可证 | 论文/TeX 为 CC BY 4.0；代码/权重/数据许可证未披露 | arXiv license link；Creative Commons BY 4.0 页面 |
| 同名/近名项目 | 未发现同名官方 WAM4D 项目；存在 X-WAM、Fast-WAM、Kinema4D、TesserAct 等近名 4D/WAM 工作，不能混用配置 | 论文 Figure 1、Related Work |

**一手资料边界。** 本报告只使用 WAM4D 论文 PDF、arXiv 页面、arXiv TeX source；代码仓库当前不可访问，因此所有“代码确认”项均为“无可确认代码”。

## 3. 核心方法与架构

**论文明确说明。** WAM4D 面向 WAM：联合建模未来观测与可执行机器人动作。问题是 2D video/latent WAM 缺少 3D 空间约束和遮挡接触几何，而显式预测 dense 4D geometry 会增加部署解码成本。WAM4D 的核心是把 geometry 作为训练期 readout target，不作为推理输入/输出：spatial registers 从历史视频特征中查询几何信息，经 DA3 head 解码未来深度，用深度损失反传到主 video-action backbone；部署时移除 geometry path（论文第2、5、6页；`paper.tex:101,296-308`）。

### ASCII 架构图

```text
Training
language instruction
        |
        v
  cross-attn into video/action tokens

3-view RGB history + future RGB targets
        |
        v
 Wan2.2 video VAE -> history video latents + clean future video latents
        |                          |
        |                          +-- noised future video states for flow matching
        |
history actions + noised future actions -> action embedding

[history video, future video noise, history action, future action noise]
        |
        v
 LingBot-VA causal video-action MoT backbone
        |                         |
        |                         +-> video flow loss L_video
        |                         +-> action flow loss L_action
        |
selected layers 12/14/16/18 history-video features
        |
spatial register tokens query history video only
        |
4 depth blocks + projection + DA3-GIANT-1.1 DualDPT head
        |
future depth SmoothL1 loss L_depth

Inference
3-view RGB observation queue + executed action history
        |
VAE encodes context latents -> replace video KV cache
action history fills action KV cache
        |
denoise future action tokens, 10 steps
        |
execute 32-action chunk; capture one observation every 4 actions

registers / depth blocks / geometric head are removed
```

## 4. 模型架构表

| 模块 | 结构/参数 | 来源与状态 |
|---|---:|---|
| 范式 | 4D World Action Model / causal video-action WAM；不是纯 VLA | 论文第1-3页 |
| 主干 | LingBot-VA causal video-action backbone；Mixture-of-Transformers (MoT)；Figure 2 显示 VA Block 1...30 | 论文第4页 Figure 2；`paper.tex:191` |
| 初始化 | video-action backbone 初始化自 pretrained LingBot-VA base model；使用 Wan2.2 video VAE | `paper.tex:361` |
| 视频编码 | Wan2.2 VAE 将 RGB sequence 编成 history latents 和 future target latents；future video token 训练时以 noised states 进入 transformer | `paper.tex:202-217` |
| 语言 | language instruction 通过 cross attention 注入，并对 video/action tokens 可见 | `paper.tex:304` |
| 动作 | 轻量 action embedding；未来动作 chunk 作为 flow-matching noised states；默认 action chunk size 32 | `paper.tex:219-227,368` |
| 动作维度 | 双臂绝对 end-effector pose：每臂 3D position + quaternion + gripper open/close，合计 16 维 | `paper.tex:372-374` |
| 动作归一化 | position 用 dataset-level q01/q99 quantile min-max；quaternion 和 gripper 固定 [-1,1]；全部 action channel 归一到 [-1,1] | `paper.tex:375-377` |
| 视觉输入布局 | 3-view mosaic：1 个 head camera + 2 个 wrist camera；主视图 256x320；每个 wrist view 128x160 | `paper.tex:369-370` |
| spatial registers | 每个 register 对应 mosaic 中 32x32 cell；主视图 8x10 grid，两个 wrist view 各 4x5；三视图 mosaic 每帧 12x10；8 个未来深度帧共 960 register tokens | `paper.tex:380-383` |
| register 插入层 | 默认在 transformer layers 12, 14, 16, 18 后做 register cross-attention | `paper.tex:300,384` |
| depth branch | 4 个 depth blocks；4 个 learned linear adapters；pretrained DA3-GIANT-1.1 any-view DualDPT head | PDF 第7页；`paper.tex:300` |
| 注意力可见性 | future action noise 可见 history video/history action/future action noise；register 仅可见 register/history video；future video noise 可见 history video/future video noise；history video 仅自见；history action 仅自见 | `paper.tex:392-408` |
| 默认参数量 | WAM4D full-task 表未给 transformer params；10-task ablation 中 Middle/Deep/Shallow/Uniform/Rand/Train DA3 为 5.690B transformer params；No Depth 为 5.089B；Bi-dir 5.841B；VAE DH 5.744B | 表6，第11页；`paper.tex:740` |
| hidden size / FFN / heads | 未披露；官方代码不可访问，无法代码确认 | 论文未给 |
| 推理路径 | 移除 register tokens、register cross-attention blocks、geometric head；仅 action prediction | `paper.tex:308,411-429,904-906` |
| 推理步数 | WAM4D inference 10 action denoising steps；视频-动作 baseline 用 5 video steps + 10 action steps | 表9，第15页；`paper.tex:918-923` |
| 延迟/显存 | A800 80GB 单卡：525.43 ± 5.64 ms/chunk，9.71 GiB peak memory | 表1/9，第8/15页 |
| 控制频率/FPS | 未披露；只说明每执行 4 个动作采集 1 次 observation | `paper.tex:429` |

## 5. 训练数据表

| 数据/阶段 | 来源、任务与 embodiment | 规模 | 模态与采样 | 动作/状态 | 公开性与处理 |
|---|---|---:|---|---|---|
| RoboTwin 2.0 main training/eval | RoboTwin 2.0 full task suite；bimanual manipulation；clean 与 randomized setting | 50 tasks；每 task 50 clean trajectories + 500 randomized trajectories | 3-view RGB mosaic；最多 17 frames/sequence；history context 为 1/5/9 frames；8 个 future target frames，每 4 collected steps 采样；depth targets 与视频 target 同 timestamp | 16D 双臂 EE action；历史动作队列 | 论文称使用 re-collected RoboTwin demonstrations with depth annotations；是否公开未披露 |
| RoboTwin 2.0 depth supervision | simulation ground-truth / depth-annotated recollection | 同上 | future depth frame-level supervision；invalid depth pixels mask | 不适用 | 论文未给数据地址 |
| AstriBot S1 real-world | AstriBot S1 robot；plate lifting、bottle placement、pen cap removal、LEGO sorting | 100 demonstrations/task，4 tasks，共 400 demos；每方法每 task 10 physical rollouts | 3-view RGB；真实 demo depth 用 DA3 offline pseudo-depth pipeline | 16D 双臂 EE action | 私有/未公开；数据地址未披露 |
| Ablation 10-task split | RoboTwin 2.0 固定 10-task subset | 每设置 clean/randomized 各 100 eval tasks；训练预算 10k steps | 同主配置 | 同主配置 | 仅论文表5列任务；数据文件未公开 |

**未披露。** 小时数、帧率、控制频率、episode 总时长、token 数、具体 train/val/test 划分文件、数据清洗、数据增强、多数据集采样比例、真实数据采集脚本和原始日志均未披露。

## 6. 训练 Recipe 和超参数表

| 阶段 | 初始化 checkpoint | 训练/冻结模块 | Loss | 超参数 | 算力/并行 | 证据 |
|---|---|---|---|---|---|---|
| Main experiments | pretrained LingBot-VA base model；Wan2.2 video VAE；DA3-GIANT-1.1 any-view DualDPT head | depth loss 更新 video-action backbone、spatial registers、depth blocks、projection layer、geometric head；是否冻结 VAE/文本模块未披露 | `L = L_video + λ_act L_action + λ_depth L_depth`；`λ_act=1`，`λ_depth=1`；`L_depth` 为 valid depth pixels 上 SmoothL1 | AdamW；LR `2e-5 * sqrt(N)`，N 为 multi-machine 数；warmup 10 steps；grad clipping 2.0；bf16；50k steps | multi-machine 训练被提到，但机器/GPU 型号和数量未披露 | `paper.tex:322-353,437-439` |
| Ablation experiments | 同主配置，按 ablation 改 depth/readout/head | 所有 ablation 除被测组件外保持 same data、backbone size、history length、action horizon、optimizer、training budget | 同上 | 10k steps；其余同主配置 | 未披露 | `paper.tex:694-697,740-785` |
| Real-world finetune/eval | 未披露是否从 main checkpoint 微调；使用 AstriBot S1 400 demos | 未披露 | depth supervision 使用 DA3 offline pseudo-depth；其他同主框架推断但论文未明确 | 未披露 | 未披露 | `paper.tex:456-460` |

**代码确认。** 无。官方代码仓库当前不可访问，无法核对 optimizer betas、epsilon、weight decay、batch size、gradient accumulation、dropout、EMA、scheduler 细节、FSDP/DeepSpeed/ZeRO 配置、显存峰值来源脚本。

## 7. Benchmark 性能表

| Benchmark | 协议 | WAM4D | Baseline 对比 | 公平性/差异 |
|---|---|---:|---|---|
| RoboTwin 2.0 full suite | 50 tasks；clean + randomized；成功率；latency 为统一设置：10 action denoise steps，video-action 模型若适用用 5 video denoise steps；VRAM 为 inference video-action backbone peak allocated memory | Clean 93.8；Rand. 89.9；Avg 91.8；Latency 525.43 ± 5.64 ms；VRAM 9.71 GiB | pi0 Avg 62.2；pi0.5 79.8；Motus 87.9；LingBot-VA 92.3；Fast-WAM 91.8 | 论文称 reproduced baselines 尽量使用 same train/eval splits、camera config、training budget；但 pi0/pi0.5 属 VLA，是否使用额外大规模预训练未展开 |
| Real-world AstriBot S1 | 4 tasks；每 task 10 physical rollouts；按 sub-action success 计分，顺序任务失败后后续 sub-actions 记 0 | Plate S1 0.9；Bottle S1 0.9；Blocks S1/S2/S3 1.0/0.9/0.8；Pen S1/S2 0.9/0.9；Avg 0.90 | pi0.5 Avg 0.74；LingBot-VA 0.84；Fast-WAM 0.80 | 真实 demo 私有；随机种子和置信区间未披露；比较受硬件/采集分布影响 |
| RoboTwin 10-task ablation | 10 tasks：Handover Block、Hanging Mug、Move Stapler Pad、Open Microwave、Pick Diverse Bottles、Place Can Basket、Press Stapler、Put Object Cabinet、Stack Bowls Three、Turn Switch；报告 clean/randomized 100-eval tasks | Middle registers: Clean 75.2、Rand 69.7；Trainable pretrained DA3: Clean 80.1、Rand 75.4 | No depth Clean 71.7、Rand 69.1；Bi-dir Clean 76.6、Rand 72.5；VAE DH Clean 70.7、Rand 68.6 | ablation 控制了 data/backbone/history/action horizon/optimizer/budget，但只在 10-task subset 上 |
| Video/Depth/Point metrics | FVD/PSNR/SSIM/LPIPS；AbsRel/δ1/δ2；CD1/CD2/F-score/F-score-T | Trainable pretrained DA3: FVD 164.5、PSNR 21.13、SSIM 0.811、LPIPS 0.143、AbsRel 0.049、δ1 0.948、CD1 0.0099、F-score 0.710 | No depth: FVD 181.2、PSNR 20.39、LPIPS 0.165；Fixed pretrained DA3: Clean 75.2、FVD 179.8 | geometry metrics 的 reference depth 来源随 simulation/real pseudo-depth 不同；真实 depth pseudo label 可靠性未量化 |
| Compute | A800 80GB 单卡；latency/chunk mean ± std；peak memory | WAM4D 10 action steps：525.43 ± 5.64 ms，9.71 GiB | pi0 64.16 ms；pi0.5 72.03 ms；Motus 1516.30 ms；LingBot-VA 843.57 ms；Fast-WAM 425.53 ms | WAM4D 快于 LingBot-VA/Motus、慢于 VLA/Fast-WAM；WAM4D 推理移除 geometry path |

## 8. 消融与局限

| 设计 | 结果 | 结论 |
|---|---|---|
| No depth vs depth | No Depth 5.089B，Clean 71.7/Rand 69.1；default middle spatial registers 5.690B，Clean 75.2/Rand 69.7；Trainable pretrained DA3 Clean 80.1/Rand 75.4 | 几何蒸馏和强 geometric prior 是主要收益来源 |
| Depth readout interface | VAE depth head Clean 70.7、AbsRel 0.081、CD1 0.0171；Middle registers Clean 75.2、AbsRel 0.053、CD1 0.0108 | spatial registers 比直接从 future VAE hidden 解码 depth 更有效 |
| Register placement | Shallow registers RGB 指标最好：FVD 168.8、PSNR 21.19、LPIPS 0.143；Middle registers geometry/control 更平衡：Clean 75.2、AbsRel 0.053、F-score-T 0.825 | 默认选择 12/14/16/18 middle layers |
| Visibility | Bi-dir registers Clean 76.6 最高，但需要主 video-action stream 读取 register features，增加计算/复杂度，geometry metrics 下降 | 默认使用 unidirectional registers |
| Geometric head | Trainable random init Clean 70.0/FVD 189.8；Fixed pretrained Clean 75.2/FVD 179.8；Trainable pretrained Clean 80.1/FVD 164.5 | 预训练 DA3 初始化 + 可训练适配最强 |
| 长期 rollout | severe occlusion 后可能把物体补全成另一个物体 | 缺 explicit long-term object memory；但控制评测中持续接收真实观测，论文称不影响 success rate |

**主要收益来源判断。** 论文证据指向架构与预训练几何先验：spatial register interface + DA3 pretrained geometric head + causal MoT visibility 是核心；数据贡献无法分离，因为无数据规模 ablation。

**复现依赖与风险。**

- 官方代码/配置/权重当前不可访问，无法从工程配置精确复现。
- RoboTwin depth-annotated recollection 与 AstriBot S1 400 demos 未公开。
- 依赖 LingBot-VA base model、Wan2.2 VAE、DA3-GIANT-1.1 any-view DualDPT head；具体 checkpoint 地址未披露。
- 训练机器数 N、GPU 数、batch size、weight decay、betas、scheduler、并行策略未披露。
- 真实机器人 AstriBot S1 平台不可替代；控制频率、低层控制接口未披露。

## 9. 最小复现步骤

由于代码不可访问，以下是“按论文可复现逻辑”的最小流程，不是可直接运行的官方流程。

1. 准备 LingBot-VA causal video-action backbone 与 Wan2.2 video VAE，初始化主干。
2. 准备 RoboTwin 2.0 full suite，重新采集或生成每任务 50 clean + 500 randomized trajectories，并保存三视角 RGB、动作、深度或深度监督。
3. 将三视角输入构造成 mosaic：head 256x320，两个 wrist 128x160；按最多 17 帧采样，history 1/5/9 帧，未来 8 帧每 4 collected steps 取样。
4. 将动作表示为 16D 双臂 absolute EE pose，并按 q01/q99 与 [-1,1] 规则归一化。
5. 在 MoT backbone 中加入 spatial register branch：每未来深度帧 12x10 registers，共 960 tokens；在 layers 12/14/16/18 后做 register-to-history-video cross attention。
6. 接入 DA3-GIANT-1.1 any-view DualDPT head 和 4 个 linear adapters，训练 `L_video + L_action + L_depth`，`λ_act=λ_depth=1`，AdamW，LR `2e-5*sqrt(N)`，10 warmup，clip 2.0，bf16，main 50k steps。
7. 推理部署时移除 register/depth/geometric head；维护 observation queue 与 action history KV cache；10 action denoising steps 输出 32-action chunk；每 4 动作采 1 张 observation。
8. 按论文评测：RoboTwin 2.0 full 50 tasks clean/randomized；真实 AstriBot S1 四任务每方法每 task 10 rollouts；A800 80GB 单卡测 latency/VRAM。

**算力估算。** 可确认的只有 inference 测试硬件：单 A800 80GB；训练 GPU/机器数、训练时间和存储未披露。考虑 5.690B transformer params、bf16、50k steps、三视角视频和深度监督，单机小卡复现不现实，但具体需求不能可靠估算。

## 10. 未披露信息清单

- 官方代码、README、训练脚本、配置文件、许可证。
- 模型权重、checkpoint、模型卡。
- WAM4D 训练数据下载地址、数据卡、真实 AstriBot S1 demos。
- backbone hidden size、FFN size、attention heads、MoT 每层细节、文本编码器细节。
- optimizer betas、epsilon、weight decay、batch size、gradient accumulation、dropout、EMA、scheduler 完整形式。
- 训练 GPU/TPU 类型、数量、机器数 N 的实际值、训练时间、存储。
- 控制频率、机器人低层控制接口、action chunk 执行时间。
- 真实任务随机种子、置信区间、失败分布。
- depth pseudo-label 生成的完整代码、过滤阈值、单位/坐标转换。

## 11. 复现性评级

**低。** 论文给出了方法、主要超参、数据规模和表格结果，arXiv source 也包含公式与附录表；但官方代码链接当前不可访问，权重/数据/配置/许可证未开放，训练细节缺失较多，真实机器人数据私有。只能复现实验思想，难以严格复现论文数字。

## 12. 一手资料链接

- arXiv: https://arxiv.org/abs/2606.14048
- PDF: https://arxiv.org/pdf/2606.14048
- TeX source: https://arxiv.org/e-print/2606.14048
- 论文声明代码地址（当前不可访问）: https://github.com/myendless1/wam4d
- arXiv/论文许可证: https://creativecommons.org/licenses/by/4.0/
