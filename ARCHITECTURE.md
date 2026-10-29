# heygem12 架构说明：整体运行逻辑与 .so 复现对照

> 本文描述**当前代码**的整体运行逻辑，以及它复现了原版 HeyGem 中哪些编译组件
> （cpython-38 `.so`）的功能。不涉及开发/修改过程（过程见 HANDOFF.md）。
> 最后更新：2026-10-20（V23：DINet crop 几何重定标——T112_REF + TEMPLATE_CROP，
> 修复嘴型不一致/脸胖瘦；chaofen_before 精确链已就绪但默认关闭，门控未实现）

---

## 1. 这个项目是什么

HeyGem（数字人唇形同步）的核心推理，用纯 Python（3.12 + PyTorch + ONNX Runtime）
重写，替代原版全部 `cpython-38` 的 `.so`。权重文件本身是标准 PyTorch / ONNX 格式，
直接复用原版 release 的 `checkpoints\`（`dinet_v1_20240131.pth`、`wenetmodel.pt`、
YuNet / PFLD / SCRFD / GFPGAN 的 onnx）。

**输入**：一段人物视频 + 一段驱动音频
**输出**：嘴部动作与驱动音频同步的新视频（人物、背景、衣着保持不变）

---

## 2. 整体运行流程

`python -m heygem12.cli --video V --audio A --output O [--device cuda]`
（入口 `heygem12/cli.py`，编排 `heygem12/pipeline.py` 的 `LipSyncPipeline.run()`）

```
读视频帧 (cv2, 25fps)
   │
   ▼
① 人脸检测（每帧）
   SCRFD(640, tracking=False, 每帧独立, det_score argmax 选脸)
   〔face/scrfd_chain.py: InsightfaceSCRFD，与原版定制封装逐点差<0.4px〕
   │
   ▼
② 音频特征（整段一次）
   wave+librosa 读 wav → 重采样 16k → 首尾补 0.4s 静音 → 预加重
   → STFT(n_fft=1024, hop=160, win=800, hann) → 80 维 mel(slaney)
   → 10·log10(·+1e-5)−20 → ConformerEncoder(12 层, concat_after=False)
   → (帧数, 256) @ 25fps → 滑窗成 (N帧, T=20, 256)
   〔audio/frontend.py + third_party/wenet/〕
   │
   ▼
③ 逐帧合成（核心循环）
   ┌───────────────────────────────────────────────────────────┐
   │ a. 五点：PFLD-98 在"原版 112 框"上跑 — warp(帧, scrfd五点→T112_REF) │
   │    得 112 crop → PFLD-98 → 取 (96,97,54,76,82) 映射回帧坐标        │
   │    （T112_REF 从原版运行时 dump 拟合；原版 PFLD 语义 ≠ 我们的旧     │
   │    REF_112×1.25，这是 V23 修复"嘴型不一致/脸胖瘦"的关键）            │
   │ b. DINet crop：warp(帧, 五点→TEMPLATE_CROP) 256×256（RGB/255）     │
   │    （TEMPLATE_CROP 同样从原版 dump 拟合，替代旧的 TEMPLATE_256）    │
   │ c. DINetV1 渲染：source = crop×input_mask, ref = crop,          │
   │    音频窗 (1,256,20) → 256×256 渲染图                             │
   │    〔models/dinet_hd.py + blocks.py〕                            │
   │ d. 合成贴回：fuse = fake×stackBlur17(fuse_mask)+crop×(1-mask)，    │
   │    经 M⁻¹ warp 回帧 + 脸椭圆 mask (轴(104,116),σ6) 贴回            │
   └───────────────────────────────────────────────────────────┘
   │
   ▼
④ 写视频：cv2 VideoWriter 写帧 → imageio-ffmpeg 把驱动音频 mux 进去
```

关键设计（均来自对原版服务器实测的逆向，见 HANDOFF.md）：

- **DINet crop 几何（V23）**：驱动五点必须在与原版相同的 112 框上跑 PFLD（原版用自己的
  对齐模板，语义逐人不同），crop 模板也是原版自有常数。两个常数（T112_REF、
  TEMPLATE_CROP）在原版运行时 dump 上拟合，渲染输出与原版 dump MAE 0.053
  （旧 TEMPLATE_256 方案 0.147，且对非拟合人系统性失效）。
- **chaofen_before 门控（未实现）**：原版 GFPGAN 复原在子进程运行且受质量门控；
  清晰素材门控关闭（crop=原帧），模糊素材（demo1）门控开启。本项目默认关闭
  （`--pre-restore` 可强制开，链已就绪）。
- **input_mask**：checkpoint 自带的 256×256 掩码，零掉下半脸，DINet 只重建嘴部。
- **无时间域处理**：与原版一致，不做任何时序平滑/EMA。

---

## 3. 原版 .so 复现对照表

| 原版组件 | 原版功能 | 本项目实现 | 状态/验证 |
|---|---|---|---|
| `landmark2face_wy/models/DINet.cpython-38.so` | DINetV1 唇形渲染网络（另含 DINet/DINetMouth/DINetMouth512 三个变体，本管线未用） | `models/dinet_hd.py` + `models/blocks.py`（结构按原版 repr 逐字重建，97.7M 参数 416 键 strict load） | ✅ 等价：喂原版 dump 的输入，渲染输出 MAE **0.0007**、corr 0.99997 |
| `compute_ctc_att_bnf.so` | wenet 音频前端：`load_ppg_model` / `get_weget` / `hparams1`(hop=160) / `SAMPLE_RATE=16000` | `audio/frontend.py`（mel 配方）+ `third_party/wenet/`（vendored ConformerEncoder，配置 `concat_after=False`） | ✅ 等价：特征 vs 原版 dump MAE **5e-5**、corr 1.0；窗口 T=20/offset 0 逐位一致 |
| `digitalhuman_interface.so` | 张量归一化 `tensor_norm=(x-127.5)/127.5`、`tensor_norm_no_training=x/255`；GFPGAN 复原与合成编排 | `pipeline.py`：GFPGAN onnx 直调、归一化、复原几何（`restore_crop`）、合成（`paste_back_oval`） | ✅ 复现（其 GFPGAN 画布用带噪声检测五点，几何不可逐位复刻，已用 PFLD 几何近似） |
| `test_3dmm_multi_exp_wenet(.0).so` | 原版推理主流程（检测→复原→crop→DINet→合成→写视频） | `pipeline.py` `run()` 全流程 | ✅ 复现 |
| `preprocess_audio_and_3dmm.so` | 音频预处理（重采样/静音/特征）+ 3DMM 系数回归（224 输入，输出 120+66+66） | 音频部分 → `audio/frontend.py`；**3DMM 回归未实现**（原版每帧跑一次，仅内部属性用，不进成片） | ◐ 部分实现 |
| `audio_handler.so` | 音频读取/转码 | `audio/frontend.py` 用 `wave`+`librosa`（不经 torchaudio） | ✅ 复现 |
| `service/trans_dh_service.so`、`server.so`、`app.py` 服务层 | HTTP 服务、任务队列、TTS 编排、`chaofen/nblend/smooth` 等开关 | **未实现**。本项目是离线 CLI，只保留核心推理链；`nblend` 语义已查明=合成掩码，`smooth`/parsing/xseg 在该管线实例中未被触发 | ✗ 不在范围内 |
| `face_detect_utils` / `face_attr_detect`（SCRFD/PFLD onnx） | 人脸检测 + 98 点关键点 | SCRFD→`face/scrfd_chain.py`；PFLD-98→`face/landmarks.py`，但推理时的 112 对齐框必须是**原版自有模板**（`pipeline.T112_REF`，dump 拟合），不是 landmarks.py 默认的 REF_112×1.25 | ✅ 等价（V23 起） |
| `face_lib/face_restore/gfpgan_onnx`（gfpgan_onnx_api .so） | GFPGANv1.4 人脸复原（onnx 封装） | onnxruntime 直载 `GFPGANv1.4.onnx`（`pipeline._gfpgan_512`） | ✅ 复现（+ 眼部保护区为本项目新增） |
| `face_lib/face_parsing/79999_iter.onnx`、`xseg/xseg_211104_4790000.onnx` | 人脸解析 / 换脸分割 | 未使用（原版实测 run 未触发；合成掩码用 canonical 脸椭圆 `face/data/face_oval_256.npy`） | ✗ 原版本管线也未用 |
| `landmark2face_wy/models/pirender_3dmm_mouth_hd_model.so`、`l2faceaudio_model.so` | 训练用模型 / 3DMM 高清渲染变体 | 推理不需要 | ✗ 不需要 |
| `wenet/`（原版目录） | wenet 运行时代码 | `third_party/wenet/`（vendored wenet v2.0.0，配置已修正） | ✅ 复现 |

---

## 4. 端到端对齐指标（同素材 248 帧，本地 CPU 版）

| 指标 | 值 |
|---|---|
| 嘴部框 MAE vs 原版成片 | **0.0189**（像素均值 ~4.8/255） |
| PFLD 唇高序列 corr@lag0 | **0.82**（无时序偏移） |
| 脸部颜色 vs 原版成片 | 逐通道差 < 1/255 |
| DINet 渲染等价性（喂原版输入） | MAE 0.0007 |
| 音频特征等价性 | MAE 5e-5 |
| 速度 | 服务器 2×4090：250 帧 1080p 约 3.5 分钟；本地 CPU 约 20 分钟 |

## 5. 已知差异（不构成缺陷 / 待办）

- **chaofen 门控未实现**：原版按质量门控决定是否 GFPGAN 复原源帧（子进程执行）。
  清晰素材门控关闭，本项目默认一致；模糊素材（如 demo1）原版会复原、我们不会，
  输出会偏软。`--pre-restore` 强制开启（精确链已就绪），自动门控待做。
- **贴回椭圆参数** (104,116)σ6 是估计值；V23 crop 尺度变化后建议按原版 vid dump 重标。
- **逐帧嘴型细节**：渲染 vs 原版 dump MAE ~0.05（嘴区 ~0.03），残余为关键点语义
  残差（~1.3px）与数值环境差异，非系统性 bug。
- **输出帧数**：318 vs 原版 316（音频窗口边界微差，遗留项）。

## 6. 授权

HeyGem 权重仅限个人/非商用。本项目自写代码可自由使用；商用需替换/自训 DINet 权重。
