# X-WAM 技术复现调研报告

## 1. 200 字以内核心结论

X-WAM（Unified 4D World Action Modeling from Video Priors with Asynchronous Denoising）是一个以 Wan2.2-TI2V-5B 视频扩散 Transformer 为 backbone 的统一 4D World Action Model：输入语言、当前多视角 RGB 和机器人状态，联合预测未来 RGB-D、未来状态和动作。核心贡献是复制 DiT 最后 10 个 block 形成轻量 depth branch，以及用 ANS 让动作 10 步先解码、视频 50 步继续生成。论文结果强，但完整复现依赖 5,873.9 小时预训练数据、256 H20 预训练和未完全公开的细节；基准 SFT/评测复现可行性为“中”，全量预训练复现为“低”。

## 2. 研究对象确认

| 项目 | 结论 | 证据 |
|---|---|---|
| 准确标题 | Unified 4D World Action Modeling from Video Priors with Asynchronous Denoising | arXiv 页面与论文源码标题；arXiv:2604.26694 |
| 模型名 | X-WAM | 论文摘要、项目主页、GitHub README |
| 作者 | Jun Guo, Qiwei Li, Peiyan Li, Zilong Chen, Nan Sun, Yifei Su, Heyun Wang, Yuan Zhang, Xinghang Li, Huaping Liu | 论文源码 `work/xwam_paper_source/neurips_2026.tex:90-107`；arXiv 页面 |
| 机构 | Tsinghua University, Xiaomi Robotics, Peking University, CASIA | 论文源码 `work/xwam_paper_source/neurips_2026.tex:113-114`；项目主页 |
| 版本/日期 | 论文首次发布：2026-04-30；arXiv/项目更新：2026-05-07 | GitHub README News；arXiv HTML v2 |
| 官方论文 | https://arxiv.org/abs/2604.26694；PDF: https://arxiv.org/pdf/2604.26694 | arXiv |
| 项目主页 | https://sharinka0715.github.io/X-WAM/ | GitHub README 与论文源码 `work/xwam_paper_source/neurips_2026.tex:117` |
| 官方代码 | https://github.com/sharinka0715/X-WAM；本次审阅 commit `72cfb86` | GitHub README，Apache-2.0；本地 clone |
| 模型权重 | https://huggingface.co/sharinka0715/X-WAM-checkpoints | GitHub README “Download Checkpoints and Datasets” |
| 数据集地址 | https://huggingface.co/datasets/sharinka0715/X-WAM-RoboCasa；https://huggingface.co/datasets/sharinka0715/X-WAM-RoboTwin | GitHub README benchmark/download 表 |
| backbone 权重 | Wan2.2-TI2V-5B，需要用户从 https://github.com/Wan-Video/Wan2.2 或 Wan-AI/Wan2.2-TI2V-5B 另行下载 | GitHub README “For Wan2.2-TI2V-5B base weights...” |
| 许可证 | 代码仓库 Apache License 2.0，copyright 2026 Xiaomi Corporation | `work/X-WAM/LICENSE` |
| 实际开放范围 | 已开放 post-training code、checkpoints、RoboCasa/RoboTwin post-training datasets；论文所述 5,873.9h 全量预训练语料未以同一仓库公开；20h 真机耳机装盒数据未公开 | GitHub README News/Download；论文附录 Table `pretrain_data` 与 real robot setup |
| 同名/近名项目 | 检索到 WAM survey、MECo-WAM、WAM4D/OA-WAM 等近名 WAM/World Model 项目，但本报告只采用 arXiv:2604.26694 与 `sharinka0715/X-WAM` 一手资料 | 研究对象以论文标题、作者、项目主页和官方仓库一致性确认 |

## 3. 模型架构表

### 范式与任务

| 项目 | 内容 | 来源与状态 |
|---|---|---|
| 解决问题 | 在一个模型内统一真实机器人动作执行、高保真未来视频生成和 3D/4D 空间重建 | 论文摘要、Section 1，明确说明 |
| 范式 | Unified 4D World Action Model；属于 WAM/World Model 交叉范式，不是纯 VLA | 论文 Section 1-3，明确说明 |
| 输入 | language instruction `c`、当前多视角 RGB `O0`、当前 proprio/state `s0` | 论文 Section 3.1；代码 `runners/xwam_runner.py:532-550` |
| 输出 | 未来 RGB `O1:H`、未来 depth `D1:H`、未来 proprio `s1:H`、动作 `a1:K` | 论文 Section 3.1，明确说明 |
| horizon | 1 条件 RGB/state + 8 future RGB/state；32 future actions；`K/H=4` | 论文 Section 3.1；代码 `configs/model/wan22_5b_sft.yaml` 中 `frame_num: 9`、dataset `frame_skip: 4/action_skip: 1` |
| 视角数 | 3 views，代码硬编码 `self.num_views = 3` | `work/X-WAM/runners/xwam_runner.py:23-25`，代码确认 |
| backbone | Wan2.2-TI2V-5B Diffusion Transformer + Wan2.2 VAE + UMT5-XXL text encoder | 论文 Section 3.1；`configs/model/wan22_5b_sft.yaml` |
| 参数量 | backbone 标称 5B；X-WAM 总参数量未披露。代码会额外复制最后 10 个 block 作为 depth branch，但未发布实际 Wan checkpoint config，无法严格重算总量 | 论文/README 明确 backbone；总量未披露 |
| Transformer 默认结构 | `dim=2048`、`ffn_dim=8192`、`num_heads=16`、`num_layers=32`、`patch_size=(1,2,2)`、`in/out_dim=16`、`num_extra_layers=10` | `work/X-WAM/modules/wan_model.py:301-319`，代码确认；注意 `from_pretrained()` 可能被外部 Wan config 覆盖 |
| depth branch | 复制最后 `M=10` 个 DiT blocks；depth branch 读取 main branch 同层 KV，main branch 不读 depth | 论文 Section 3.2；Algorithm 1；代码 `modules/wan_model.py:638-684`, `:832-834` |
| action/proprio encoder | action/proprio 由 MLP 投影到 DiT hidden space，输出再由对称 MLP decode | 论文 Section 3.1；代码 `modules/wan_model.py:430-438` |
| text encoder | `umt5_xxl`，`text_len=512`，`torch.bfloat16` | `configs/model/wan22_5b_sft.yaml` |
| VAE | `Wan2.2_VAE.pth`，stride `(4,16,16)`；VAE frozen | `configs/model/wan22_5b_sft.yaml`；`runners/xwam_runner.py:37-39` |
| position encoding | 视频使用 3D RoPE；多视角加 learnable view embedding；action/proprio 使用 temporal RoPE | 论文 Section 3.1；代码 `modules/wan_model.py:443-462`, `:587-598` |
| attention | 拼接 `[video tokens | action tokens | proprio tokens]` 后 full bidirectional attention | 论文 Section 3.1；代码 `modules/wan_model.py:604-607` |
| action 表示 | 14-D relative vector: `(delta position 3 + delta axis-angle 3 + gripper 1) * 2 arms`；单臂只监督前 7 维 | 论文 Appendix “State and action representation”；代码 `data/robot_dataset.py:61-78` |
| state 表示 | 16-D absolute vector: `(position 3 + quaternion 4 + gripper 1) * 2 arms`；单臂只监督前 8 维 | 同上 |
| 生成方式 | rectified/flow matching velocity prediction + UniPC scheduler；非自回归、非离散 token action | 代码 `runners/xwam_runner.py:147-180`, `:448-527`；论文 Algorithm 2 |
| ANS | 训练中采样联合分布约束 `tO >= ta`；推理中 action 用 `Ta=10`，video 用 `TO=50` | 论文 Section 3.3、Appendix Inference；代码 `configs/model/wan22_5b_sft.yaml` |
| CFG | 训练 text dropout `0.1`；默认 `cfg_list: [0,2,4]`；论文评测 CFG scale 1.0 | config 与论文 Appendix Inference |
| 推理缓存 | depth branch 读取 main branch KV cache；未披露跨 rollout KV-cache/历史缓存；论文局限明确说没有 historical/autoregressive rollout | 代码 `modules/wan_model.py:657-684`；论文 Limitations |
| 显存/硬件 | 论文未披露模拟 benchmark 推理显存；真机单 RTX 5090 D，8 denoise steps，约 300ms/action chunk；ablation 单 RTX 3090 latency | 论文 Appendix real robot setup；Table `ablation` |

### ASCII 架构图

```text
Instruction c
   |
   v
UMT5-XXL text encoder (frozen/not optimized) --> text embeddings

Current multi-view RGB O0 ----+
Future RGB targets O1:H ------+--> Wan2.2 VAE encode --> RGB/video latents
Depth targets D1:H -----------+--> pseudo-RGB depth -> same VAE encode

Current state s0 / future states s1:H --> proprio MLP encoder
Future action chunk a1:K -----------> action MLP encoder

             [video tokens | action tokens | proprio tokens]
                         + view embedding / RoPE / timestep embedding
                                      |
                         shared Wan2.2 DiT blocks 1...(N-M)
                                      |
                 +--------------------+--------------------+
                 |                                         |
       main RGB/action/proprio branch              depth branch
       DiT blocks N-M+1...N                        copied last M=10 blocks
       full sequence                               video/depth tokens only
                 |                                 reads same-layer main KV
                 |                                         |
       RGB velocity head                           depth latent head
       action decoder MLP
       proprio decoder MLP
                 |
       Flow/UniPC denoising
       - action/proprio stop after Ta steps
       - video/depth continue until TO steps
```

## 4. 训练数据表

### 大规模预训练数据

| 数据集 | 来源 | episode | 时长 | 模态/处理 | embodiment/任务 | 公开性 | 来源 |
|---|---:|---:|---:|---|---|---|---|
| AgibotWorld-Beta | Real | 866,562 | 2,221.5h | RGB video；depth 由 Video Depth Anything 生成；统一降采样到 3.75 FPS、320x256 | 未披露到任务/机器人维度 | 原始数据是否完整用于复现未由 X-WAM 仓库打包公开 | 论文 Appendix Table `pretrain_data` |
| DROID | Real | 74,734 | 280.3h | 同上；额外过滤 stationary frames | 未披露 | 外部公开数据；X-WAM 过滤后版本未公开 | 论文 Appendix Table `pretrain_data` |
| InternA1-Aloha | Sim | 184,803 | 1,337.3h | 同上 | 未披露 | 未披露 | 论文 Appendix Table `pretrain_data` |
| InternA1-Genie1 | Sim | 50,638 | 174.0h | 同上 | 未披露 | 未披露 | 论文 Appendix Table `pretrain_data` |
| InternA1-Lift2 | Sim | 231,018 | 1,464.7h | 同上 | 未披露 | 未披露 | 论文 Appendix Table `pretrain_data` |
| RoboCasa MimicGen | Sim | 56,771 | 282.4h | 同上 | RoboCasa kitchen manipulation | 外部/benchmark 数据；X-WAM 发布 post-training dataset 链接 | 论文 Appendix Table `pretrain_data`；GitHub README |
| RoboTwin 2.0 | Sim | 27,500 | 113.7h | 同上 | RoboTwin 2.0 dual-arm tasks | X-WAM 发布 post-training dataset 链接 | 论文 Appendix Table `pretrain_data`；GitHub README |
| Total | -- | 1,492,026 | 5,873.9h | 删除 base locomotion、dexterous manipulation、failed executions；DROID 删除 stationary frames | 混合真实/仿真 | 全量混合后训练集未公开 | 论文 Appendix Table `pretrain_data` |

### Benchmark / SFT 数据

| 数据 | 训练/评测设置 | 关键配置 | action/state | 公开性 | 来源 |
|---|---|---|---|---|---|
| RoboCasa | 24 manipulation tasks；评测每任务 100 episodes；论文使用 benchmark data fine-tune | dataset config: `sequence_length=9`、`frame_skip=4`、`action_skip=1`、`video_size=[256,320]`、augmentation on；README 示例 episode 原始 `fps=20.0`，加载后除以 `frame_skip` | `action_dim=14`、`proprio_dim=16`；RoboCasa 有 `raw_actions` 时直接用 raw command | HF dataset `X-WAM-RoboCasa` 公开；真实 simulator replay seeds 不与测试重叠但具体 seeds 未披露 | 论文 Appendix Fine-tuning/Inference；`configs/data/robocasa.yaml`; `data/robot_dataset.py:187-215` |
| RoboTwin 2.0 | 50 tasks；Clean/Randomized；论文写“all trajectories of AgileX arms, including 50 clean and 500 randomized trajectories on 50 tasks” | 同上；config 中 `inverse_gripper=false`、`normalize_depths_per_view=false`、`shuffle_view_order=false` | 相对动作转为基于 action chunk 首帧状态的 absolute EE poses 后送 simulator | HF dataset `X-WAM-RoboTwin` 公开 | 论文 Section Experiments/Appendix Fine-tuning；`configs/data/robotwin.yaml` |
| Real robot earphone packing | 约 20h demonstration；AC One dual-arm；1 main + 2 wrist cameras；320x256；15Hz control；15 actions/chunk | 64 H20 fine-tune 40,000 steps；RTX 5090 D inference，8 denoise steps，约 300ms/chunk | dual-arm EE control；每 chunk 执行 1s | 未公开 | 论文 Appendix Real Robot Experiments |

### 数据处理与归一化

| 项目 | 内容 | 来源与状态 |
|---|---|---|
| 多视角读取 | episode JSON 中 `observations` 按 view name 排序读取 RGB/depth mp4；view type 为 `static`/`dynamic` | README data structure；`data/robot_dataset.py:342-416` |
| 分辨率 | 预训练论文：320x256；SFT config：`[256,320]` 即 H=256/W=320；RoboCasa eval client 渲染默认 256x256 | 论文 Appendix；`configs/data/*.yaml`; `evaluation/robocasa_client.py:79-127` |
| 数据增强 | crop_ratio 0.95；brightness/contrast/saturation 0.2；hue 0.05 | `configs/data/*.yaml`，代码确认 |
| depth | 预训练多数数据无 depth，用 Video Depth Anything 提取；fine-tuning replay simulator 获得 GT depth | 论文 Appendix Pretraining Data/Fine-tuning |
| 指令 | episode JSON `instructions` 列表中采样 prompt；text dropout 0.1 用于 CFG | README data structure；`runners/xwam_runner.py:91-97` |
| 归一化 | state/action 使用每数据集 `q0.01/q0.99` quantile；action 仅 scale 不加 bias 以保持 zero action 语义；代码实际 `_quantile_normalize_np` 为线性映射到 [-1,1]，raw action 分支亦使用 q01/q99 | 论文 Appendix；`data/robot_dataset.py:527-530`, `:601-627` |
| 多数据集采样/混合比例 | 未披露 | 未披露 |
| 相机数量/帧率/控制频率逐数据集明细 | 除统一 3.75 FPS/320x256 和代码 3 views 外，逐数据集未披露 | 未披露 |

## 5. 训练 Recipe 和超参数表

| 阶段 | 项目 | 数值/设置 | 来源与状态 |
|---|---|---|---|
| 初始化 | base checkpoint | Wan2.2-TI2V-5B；VAE `Wan2.2_VAE.pth`；T5 `umt5_xxl` | 论文 Section 3.1；`configs/model/wan22_5b_sft.yaml` |
| 初始化 | 新增模块 | view/action/proprio/depth heads normal init；extra depth blocks copy backbone 最后 10 层权重 | `modules/wan_model.py:791-834`，代码确认 |
| 冻结/训练 | VAE | `eval()` + `requires_grad_(False)` | `runners/xwam_runner.py:37-39` |
| 冻结/训练 | text encoder | `eval()`；optimizer 只含 `self.model.parameters()`，因此 T5 不训练 | `runners/xwam_runner.py:28-39`, `:62-67`，代码确认 |
| 冻结/训练 | DiT/X-WAM | `self.model.train()`；训练所有 `self.model.parameters()` | `runners/xwam_runner.py:54-60`, `:62-67` |
| Loss | 总式 | `video_loss + λa action_loss + λs proprio_loss + λD depth_loss + λdct dct_loss` | `runners/xwam_runner.py:230-257` |
| Loss | 权重 | `action=1.0`、`depth=1.0`、`proprio=1.0`、`dct=0.0` | `configs/model/wan22_5b_sft.yaml`; 论文 Appendix Pretraining |
| Loss | video/action/proprio | flow matching velocity MSE on noisy latents/actions/states | `runners/xwam_runner.py:147-180`, `:230-232` |
| Loss | depth | 论文说 inverse depth MSE；代码发布版对 `depth_latents_pred[0]` 与 `gt_depth_latents` 做 MSE | 论文 Section 3.2；`runners/xwam_runner.py:233-236` |
| 优化器 | AdamW | pretrain peak LR `1e-4`；fine-tune paper `3e-5`；发布 config 默认 `1e-5` | 论文 Appendix；`configs/model/wan22_5b_sft.yaml`。存在论文/发布配置差异 |
| 优化器默认 | betas/epsilon | 代码未显式传入；根据 PyTorch AdamW 默认推断 betas=(0.9,0.999)、eps=1e-8 | `runners/xwam_runner.py:62-67`，根据实现推断 |
| weight decay | 0.01 | `configs/model/wan22_5b_sft.yaml` |
| Scheduler | linear warmup + cosine decay to 0 | 论文 Appendix；`runners/xwam_runner.py:68-72` |
| warmup | pretrain 1,000 steps；fine-tune 论文称 same warmup；发布 config 默认 200；README RoboTwin 示例 override 400 | 论文 Appendix；config/README |
| steps | pretrain 40,000；benchmark fine-tune 20,000；real robot fine-tune 40,000 | 论文 Appendix |
| batch | pretrain 256 H20 * 8 = 2,048；benchmark SFT 32 H20 * 4 = 128；real robot 64 H20 * 4 = 256；发布 config `batch_size_per_gpu=4` | 论文 Appendix；`configs/model/wan22_5b_sft.yaml` |
| gradient accumulation | 1 | `configs/model/wan22_5b_sft.yaml` |
| precision | `bf16-mixed` | `scripts/train_sft.py:126-137` |
| gradient clipping | norm 1.0 | `configs/model/wan22_5b_sft.yaml`; `scripts/train_sft.py:132-134` |
| dropout | text dropout 0.1；Transformer dropout 未披露/未见显式配置 | `configs/model/wan22_5b_sft.yaml`; `runners/xwam_runner.py:91-97` |
| EMA | 未披露，代码未见 EMA | 未披露/代码未确认 |
| noise/timestep | train timesteps 1000；`rf_distribution=uniform`；`time_shifting=5.0`；ANS `clean_action_ratio=0.5`；joint distribution enabled | `configs/model/wan22_5b_sft.yaml`; `runners/xwam_runner.py:103-145` |
| diffusion/flow | flow matching；sample scheduler FlowUniPCMultistepScheduler | `runners/xwam_runner.py:448-527` |
| 并行策略 | Lightning `DeepSpeedStrategy`，bucket size `5e8`；ZeRO stage 未显式设置；FSDP 未使用 | `scripts/train_sft.py:121-137` |
| GPU/时间 | GPU 类型/数量如上；训练 wall-clock time 未披露 | 论文 Appendix；训练时间未披露 |

## 6. Benchmark 性能表

### Policy benchmark

| Benchmark | 协议 | Method | Avg SR / Clean / Randomized | 比较公平性 | 来源 |
|---|---|---:|---:|---|---|
| RoboCasa | 24 tasks；每任务 100 episodes；metric: success rate | pi0 | 62.5 | 结果取自 Cosmos Policy；额外预训练/数据协议可能不同 | 论文 Table `robocasa` 与 Appendix Baseline Details |
| RoboCasa | 同上 | GR00T-N1.5 | 64.1 | 同上 | 同上 |
| RoboCasa | 同上 | UWM | 60.8 | 结果取自 Cosmos Policy；非同 backbone | 同上 |
| RoboCasa | 同上 | DreamZero | 62.4 | 作者用 official code 复现并替换 Wan2.2-5B，较公平；但无 X-WAM depth/ANS | 同上 |
| RoboCasa | 同上 | Cosmos Policy | 67.1 | 结果取自 Cosmos Policy；数据/预训练可能不完全一致 | 同上 |
| RoboCasa | 同上 | X-WAM | 79.2 | 使用 5,873.9h pretraining + benchmark SFT | 论文 Table `robocasa` |
| RoboTwin 2.0 | 50 tasks；Clean/Randomized；每任务 100 episodes | pi0 | 65.9 / 58.4 | 取自 LingBot-VA；协议/数据可能不同 | 论文 Table `robotwin` 与 Appendix Baseline Details |
| RoboTwin 2.0 | 同上 | pi0.5 | 82.7 / 76.8 | 同上 | 同上 |
| RoboTwin 2.0 | 同上 | UWM | 81.7 / 78.6 | 作者 reimplement + Wan2.2-5B，较公平 | 论文 Appendix Baseline Details |
| RoboTwin 2.0 | 同上 | GigaWorld-Policy | 87.0 / 85.0 | 取自原论文；非完全同训练/评测实现 | 同上 |
| RoboTwin 2.0 | 同上 | Motus | 88.7 / 87.0 | 取自原论文；非完全同训练/评测实现 | 同上 |
| RoboTwin 2.0 | 同上 | X-WAM | 89.8 / 90.7 | 使用 AgileX trajectories + X-WAM pretraining/SFT | 论文 Table `robotwin` |

### 4D reconstruction / generation

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | AbsRel ↓ | δ1 ↑ | CD ↓ | 来源 |
|---|---:|---:|---:|---:|---:|---:|---|
| DreamZero + DA3 | 21.12 | 0.7788 | 0.1580 | 0.1362 | 0.8594 | 0.0680 | 论文 Table `4d_recon` |
| Robot4DGen | 22.67 | 0.8207 | 0.1026 | 0.0736 | 0.9443 | 0.0134 | 同上 |
| X-WAM w/o depth + DA3 | 23.09 | 0.8916 | 0.0548 | 0.1045 | 0.9089 | 0.0401 | 同上 |
| X-WAM | 23.46 | 0.8942 | 0.0513 | 0.0349 | 0.9738 | 0.0049 | 同上 |

### Real robot

| 设置 | XR-0 progress/time | X-WAM progress/time | 协议 | 来源 |
|---|---:|---:|---|---|
| Pack 1 earphone | 100.0% (24/24), 54.66s | 100.0% (24/24), 41.63s | 每设置 6 trials；progress 分 4 阶段，每阶段 25% | 论文 Table `real_robot` |
| Pack 2 earphones | 79.1% (38/48), 115.44s | 93.8% (45/48), 113.25s | 同上 | 同上 |
| Pack 3 earphones | 63.9% (46/72), 195.66s | 68.0% (49/72), 160.72s | 同上 | 同上 |
| Novel placements | 58.3% (14/24), 89.63s | 70.8% (17/24), 46.68s | OOD | 同上 |
| Unseen tablecloth | 66.7% (16/24), 65.73s | 66.7% (16/24), 62.01s | OOD | 同上 |
| Unseen distractors | 66.7% (16/24), 76.32s | 75.0% (18/24), 51.53s | OOD | 同上 |

### 推理延迟与硬件

| 场景 | 延迟/设置 | 来源 |
|---|---|---|
| benchmark 默认 | `sample_steps=50`、`action_denoise_steps=10`、`use_decoupled_inference=true` | `configs/model/wan22_5b_sft.yaml`; evaluation README |
| ablation | async 5 action steps: 1033ms；sync 25 steps: 4665ms；sequence concat 1888ms；channel concat 1266ms；single RTX 3090 | 论文 Table `ablation` |
| real robot | 8 denoise steps，RTX 5090 D，约 300ms/action chunk；RTC delay 6 actions；15Hz control；15 actions/chunk | 论文 Appendix Real Robot Setup |
| 显存/FPS | benchmark 显存未披露；FPS 除 sample_fps=5 和 real robot 15Hz 外未披露 | 未披露/代码确认 |

## 7. 消融与局限

### 消融

| 设计 | 变体 | 结果 | 解读 | 来源 |
|---|---|---|---|---|
| depth architecture | No depth | SR 63.0；latency 1033ms；无 depth/CD | 去掉 depth 监督，SR 比 interleaved branch 低 4.8pp | 论文 Table `ablation(a)` |
| depth architecture | Sequence concatenation | SR 68.7；latency 1888ms；PSNR 23.60；CD 0.0037 | 指标最好但序列翻倍导致 latency 接近 1.83x | 同上 |
| depth architecture | Channel concatenation | SR 64.2；latency 1266ms；CD 0.0052 | 破坏预训练输入分布，SR 低 | 同上 |
| depth architecture | Interleaved branch | SR 67.8；latency 1033ms；CD 0.0049 | latency 等同 no-depth，质量接近 sequence concat | 同上 |
| noise schedule | Sync train + Sync infer | SR 66.4；latency 4665ms | 质量强但动作等待长 | 论文 Table `ablation(b)` |
| noise schedule | Decoupled train + Sync infer | SR 66.3；latency 4665ms | 无明显收益 | 同上 |
| noise schedule | Decoupled train + Async infer | SR 67.2；latency 1033ms；PSNR 22.60 | 快但视频/深度退化，训练-推理分布错配 | 同上 |
| noise schedule | ANS train + Async infer | SR 67.8；latency 1033ms；AbsRel 0.0349 | 在同 latency 下兼顾 SR 和 depth | 同上 |

### 性能来源判断

| 来源 | 证据 | 结论 |
|---|---|---|
| 架构 | depth branch ablation 提升 SR 和 4D reconstruction；sequence concat 更强但慢 | 空间监督/branch 是关键来源之一 |
| 推理策略 | ANS 把 ablation latency 从 4665ms 降至 1033ms，同时保持/提升 SR | 动作实时性主要来自 ANS |
| 数据/预训练 | 主结果基于 5,873.9h 预训练；ablation 明确未用大规模预训练且 SR 为 67.8，不是主表 79.2 | 主表性能很大程度来自大规模 robot pretraining + SFT |
| backbone | DreamZero/UWM 公平版本替换 Wan2.2-5B；X-WAM仍更强 | backbone 重要，但不足以解释全部提升 |

### 论文/代码局限

| 局限 | 说明 | 来源 |
|---|---|---|
| 历史上下文短 | 只用固定长度当前 context，无 historical info/autoregressive rollout/KV caching | 论文 Limitations |
| latency 高于专用 policy | 统一生成视频和 action，真机 8 steps 约 300ms/chunk；可能让机器人基于过去几帧预测行动 | 论文 Limitations |
| 全量数据不可复现 | 5,873.9h 预训练混合数据的清洗后版本、混合比例、采样权重未公开 | 论文 Appendix + GitHub README |
| 训练 wall-clock/显存未披露 | 无训练时间、显存曲线、benchmark 推理硬件显存 | 未披露 |
| 代码配置与论文 SFT LR 有差异 | 论文 fine-tune LR `3e-5`；发布 config 默认 `1e-5`，README 训练示例未覆盖 LR | 论文 Appendix；`configs/model/wan22_5b_sft.yaml` |
| depth loss 表述差异 | 论文写 inverse depth MSE；代码是 VAE depth latent MSE | 论文 Section 3.2；`runners/xwam_runner.py:233-236` |
| checkpoint config 外部依赖 | X-WAM repo 未含 Wan2.2-TI2V-5B base config/weights；实际 model shape 由外部 checkpoint 目录决定 | GitHub README；`XWAMModel.from_pretrained(...)` |

整体可复现性评级：

| 范围 | 评级 | 理由 |
|---|---|---|
| 直接加载公开 checkpoint 做 RoboCasa/RoboTwin evaluation | 中-高 | 代码、评测 broker/server/client、checkpoint/dataset 地址公开；但依赖 submodules、Wan base weights、环境版本和 HF 下载可用性 |
| benchmark SFT 复现 | 中 | SFT 代码/配置/数据链接公开；论文与 config 的 LR/warmup 差异需手动校正；需要 32 H20 级别资源才贴近论文 |
| 全量预训练复现 | 低 | 5,873.9h 数据清洗后版本、混合比例、训练时间、完整预训练 pipeline 未全部公开；算力需 256 H20 * 40k steps |

## 8. 最小复现步骤

以下是“加载官方 checkpoint 复现实验评测”的最小路径；不是全量预训练。

1. 克隆仓库和 submodules：

```bash
git clone --recurse-submodules https://github.com/sharinka0715/X-WAM.git
cd X-WAM
```

2. 安装环境：

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

来源：GitHub README Installation；要求 Python>=3.10、torch>=2.4、numpy<1.26、diffusers>=0.31、transformers 4.49-4.51.3、flash-attn。

3. 下载权重和数据：

```bash
huggingface-cli download sharinka0715/X-WAM-checkpoints --local-dir checkpoints
huggingface-cli download sharinka0715/X-WAM-RoboCasa --repo-type dataset --local-dir datasets/RoboCasa
huggingface-cli download sharinka0715/X-WAM-RoboTwin --repo-type dataset --local-dir datasets/RoboTwin
```

另需下载 Wan2.2-TI2V-5B base weights 并放到 `checkpoints/wan22_5b/` 或通过 `--wan_checkpoint_dir` 指定。来源：GitHub README。

4. 运行 policy broker：

```bash
python evaluation/policy_broker.py --frontend_port 10086 --backend_port 10087
```

5. 运行 policy server：

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/policy_server.py \
  --exp_path checkpoints/robocasa_sft \
  --wan_checkpoint_dir /path/to/wan22_5b \
  --broker_port 10087 \
  --denoise_steps 50 \
  --action_denoise_steps 10
```

RoboTwin 改为 `checkpoints/robotwin_sft`。来源：`evaluation/README.md`。

6. RoboCasa 评测：

```bash
for i in $(seq 0 23); do
  python evaluation/robocasa_client.py \
    --env_global_rank $i \
    --world_size 24 \
    --num_evals_per_worker 100 \
    --server_port 10086 \
    --save_root_dir ./eval_results/robocasa/ &
done
wait
```

7. RoboTwin 2.0 评测：

```bash
python evaluation/robotwin_client.py \
  --task_name adjust_bottle \
  --task_config demo_randomized \
  --num_evals_per_worker 100 \
  --server_port 10086 \
  --save_root_dir ./eval_results/robotwin/
```

全 50 任务任务名列表见 `evaluation/README.md`。

8. 若做 benchmark SFT：

```bash
torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 \
  --master_addr=localhost --master_port=29500 \
  scripts/train_sft.py dataset=robocasa exp_name=robocasa_sft
```

更接近论文应使用 32 NVIDIA H20、global batch 128、20,000 steps、LR `3e-5`、warmup 1,000 steps；发布 config 默认 LR `1e-5`、warmup 200，需要显式 override。

## 9. 未披露信息清单

| 类别 | 未披露项 |
|---|---|
| 模型 | X-WAM 加 depth branch 后总参数量；实际发布 checkpoint 的完整 Wan2.2 config 是否完全等同代码默认值；attention heads/hidden/FFN 若由外部 checkpoint 覆盖后的最终值 |
| 数据 | 5,873.9h 预训练数据的清洗后文件列表、license 汇总、每数据集采样比例、每数据集相机数/分辨率/帧率/action/state 维度细表 |
| 训练 | AdamW betas/eps 没有显式写在论文；训练 wall-clock；训练显存；DeepSpeed ZeRO stage；是否使用 activation checkpointing 的论文配置；EMA |
| 评测 | benchmark 随机种子列表；benchmark 具体版本/commit；simulation FPS/control frequency；推理显存；模拟 benchmark 推理硬件 |
| 真机 | 20h earphone packing 数据、机器人控制器细节、相机标定、动作后处理完整实现、训练数据 split |
| 开放范围 | 全量预训练数据与真实机器人数据未公开；只确认公开 post-training code、checkpoints、RoboCasa/RoboTwin datasets |

## 参考一手资料

- arXiv: [Unified 4D World Action Modeling from Video Priors with Asynchronous Denoising](https://arxiv.org/abs/2604.26694)
- arXiv HTML v2: [2604.26694v2](https://arxiv.org/html/2604.26694v2)
- Project page: [X-WAM](https://sharinka0715.github.io/X-WAM/)
- Code: [sharinka0715/X-WAM](https://github.com/sharinka0715/X-WAM)
- Checkpoints: [sharinka0715/X-WAM-checkpoints](https://huggingface.co/sharinka0715/X-WAM-checkpoints)
- Dataset: [X-WAM-RoboCasa](https://huggingface.co/datasets/sharinka0715/X-WAM-RoboCasa)
- Dataset: [X-WAM-RoboTwin](https://huggingface.co/datasets/sharinka0715/X-WAM-RoboTwin)
- Backbone: [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)
