# 交接文档：heygem12 项目状态与下一步

> 写给下一个 session。读完即可无缝接续。
> 整体运行逻辑与 .so 复现对照见 **ARCHITECTURE.md**（不含修改过程，本文档只管过程与状态）。
> 最后更新：**2026-10-29（V26：原版 landmark/mask 链精确复刻——黑盒实证，不再是近似。
> convert98to68 映射运行时拦截解出（err=0.0000）；mask 规则用原版函数扰动实验辨识
> （重建 landmark 调原版函数 IoU=1.0）；PFLD 裁剪规则模板匹配实证（bbox+0.1 边距）。
> 6 素材+qipao 全部目检干净，闪烁热点 man1 190 / man2 0 / woman 355。V25 及更早见下文）**

---

## 0. 服务器连接方式（必读，先连上再干活）

- **主机**：`jinyun@10.0.70.127`，密码 `123456`（sudo 同密码）。
- **工具**：本地 Windows 用 **PuTTY 套件**（`D:\soft\putty\`），核心是两个命令行：
  - 远程执行：`D:\soft\putty\PLINK.EXE -ssh -pw 123456 -batch jinyun@10.0.70.127 "<远程命令>"`
  - 上传文件：`D:\soft\putty\PSCP.EXE -pw 123456 -batch <本地文件> jinyun@10.0.70.127:<远程路径>`
  - 下载文件：`D:\soft\putty\PSCP.EXE -pw 123456 -batch jinyun@10.0.70.127:<远程路径> <本地文件>`
  - （图形会话也可用 PuTTY 直接 ssh 登录；注意 pscp **不支持一次多个远程源文件**，需分次执行）
- **远程命令的坑（都踩过，务必照做）**：
  1. PowerShell 里嵌 bash 引号/heredoc 极易踩坑——**一律在本地写好 .sh/.py 文件，pscp 上传后再用 plink 执行**；上传 shell 脚本必须用 LF 行尾（用编辑器/Write 工具写，别用 PowerShell `Set-Content` 直接写，会产出 CRLF 导致 `ambiguous redirect`）。
  2. 长任务后台跑：`cd <目录> && nohup bash xxx.sh > /dev/null 2>&1 & disown`，然后另起 plink 轮询日志；plink 断开会杀前台进程。
  3. 运行 Python 必须 `env -u LD_LIBRARY_PATH`（LD_LIBRARY_PATH 里 libs 会冲突）。
  4. `/` 盘空间紧张：`export TMPDIR=/opt/projects/opencode/tmp`。
  5. GPU 指定：`export CUDA_VISIBLE_DEVICES=1`。
- **关键路径**：
  - 本项目（我方 pipeline）：`/opt/projects/heygem12/`（heygem12 包 + assets/media 素材 + checkpoints 模型 + dev/data2 输出）
  - 原版 HeyGem：`/opt/projects/heygem12/HG/`（run.py + 各 .so；跑法见 scripts/make_qipao_compare.sh 第 2 步）
  - Python 环境：我方 `/opt/anaconda3/envs/heygem312/bin/python`；原版 `/opt/anaconda3/envs/heygem38/bin/python`
  - 临时工作区：`/opt/projects/opencode/tmp/`（分析脚本、hook、Ghidra 产物、dump）
  - Ghidra：`/opt/projects/opencode/ghidra_12.1.4_PUBLIC/`（analyzeHeadless，用法见 tmp/run_ghidra*.sh）
- **一键全流程示例**：`/opt/projects/heygem12/scripts/make_qipao_compare.sh`（我方生成 → 原版生成 → 三联对比视频，三步全含，可直接照抄改参数）。

---

## 当前状态（2026-10-29 定版 V26）——新 session 从这里读起

### 一句话现状

**V26 = 原版 landmark/mask 链的精确复刻（黑盒实证，不再是近似）**（`HEYGEM12_GEOM=orig`，
默认开；`=v24` 回退旧链）。逐帧链：SCRFD → **bbox+0.1 边距裁剪→resize 112→PFLD-98
→convert98to68（精确映射，运行时解出）→ 68 点** → boundingRect →
**x1=x_c−0.75w, y1=y−0.15w, side=1.5w（原版公式原样）** → conv5 对称边界
滑动平均 + |raw−smooth|>12px 保 raw 的去抖**（非因果，全片预处理）→
矩形 crop → resize 266×266 INTER_CUBIC → **DINet 输入 = crop266[5:261]**（原版的
`[5:-5]` 切片）→ DINet+checkpoint fuse_mask → 放回 266 画布 → resize 回 side →
**mask = convexHull(下颌[2:15]+鼻尖[29]，共14点) + 下半椭圆(cx=下颌中心,
cy=下巴(8).y−34, axes=下颌宽×0.35) + dilate 6×6** 贴回。6 素材 + qipao 全部验证。

### 原版架构（本轮逆向完整还原，证据 = Ghidra 反编译 + cv2 trace 实证）

- **preprocess_audio_and_3dmm.so**（类 op，__init__(caped_img2, wh, scrfd_detector,
  scrfd_predictor, hp, lm3d_std, img_size, driver_flag)）：`loc_detect_face`（逐帧
  SCRFD→3DMM 68 点）→ `smooth_`（bounding_box_p 序列 convolve2d(ones(5)/5,
  boundary='symm', mode='same') + np.where(|raw−smooth|>12 保 raw）→ `loc_crop_face`
  （boundingRect→上述矩形公式→crop→resize 266 INTER_CUBIC→crop_img/crop_lm）。
  **原版预处理是非因果的**，先看完整视频再合成——这是它不抖的根本原因。
- **landmark2face_wy/digitalhuman_interface.so**（DigitalHumanModel）：
  `inference_notraining`：crop_img(266)+crop_lm → `get_face_mask`（convexHull→
  fillConvexPoly(255)→ellipse 下半 0-180°→dilate 6×6，全程 266 画布）→ mask[5:-5,5:-5]
  → DINet 输入 crop_img[5:-5,5:-5]（266=256+5×2）→ netG → blendImages(GPU
  featherAmount=0.1) 贴回。mask 构建在**预处理父进程**，不在 DINet 子进程。
- **trans_dh_service.so**：`write_video_chaofen` = 主编排（线程链 FaceDetectThread→
  FaceRestoreThread→FaceParseThread→FaceReverseThread→CVVideoWriterThread）；
  `chaofen_src`：get_bboxes→get_single_face(crop_size=512, mode='mtcnn_512')→
  GFPGAN→CVImage.recover_from_reverse_matrix(img_fg_mask=固定椭圆 face_mask)；
  `get_face_mask`（模块级）= 512 画布 ellipse(256,256)(220,160) 90° + 20px 边清零 +
  stackBlur(201,201) + /255（= cv2box.utils.util.common_face_mask，静态一次性）。
- **cv2box 是纯 Python 包**（heygem38 site-packages，v0.5.9）：CVImage.norm_crop /
  recover_from_reverse_matrix / blob_innormal / common_face_mask 全部源码可读。
- **cv2 trace 实证**（hookdir6，man1 3s）：boundingRect n=68 (415,600,292,282) 逐帧
  几乎不动（x=415 恒定）；crop (439,438)→(266,266) INTER_CUBIC + 反向 (266,266)→
  (439,438) INTER_LINEAR；439/292≈1.5 ✓；fillConvexPoly 14 点（下颌 hull，y87..197）
  + ellipse c(133,164) ax(60,60) 0-180°（c_x≈hull 中心，c_y≈嘴顶，ax≈1.5×嘴宽）+
  dilate (6,6)×1；GFPGAN warpAffine (380,289)→512 borderMode=1；检测输入
  1920×1080→(360,640)（挤压检测）。
- **ellipse 参数标定**（probe_ellipse.py 实证）：c=(hull 中心 x, 嘴上沿 y)，
  ax=1.5×|lm[76]−lm[84]|（PFLD-98 嘴角 76/84）；hull 输入 = lm[0:33]+lm[51:60]
  （下颌+鼻，鼻尖为 hull 顶顶点，不含眉眼——与原版 14 点 hull y87..197 一致）。

### 闪烁实测（flick_cmp.py，>2 灰度热点像素数，越小越好）

| 素材 | V24(cur) 热点 | V25(orig 几何) 热点 | 备注 |
|---|---|---|---|
| man1 | **4319** | **6** | 主诉修复；in-region 5.21 vs orig 5.73 |
| man2 | — | **0** | |
| woman | 530 | 898 | in-region 3.10 vs orig 3.29（仍 ≤ 原版）|
| demo1 | — | 352 | 模糊素材；in-region 8.47 vs 8.18 略高 |
| demo2 | — | 1024 | in-region 10.33 vs 10.54（≤ 原版）|
| demo3 | — | 73 | |

### V25.1 双下巴修复（2026-10-28 下午，用户报 0.17s 双下巴/椭圆边界）

- **根因**：我方 PFLD-98 的下巴点比原版 3DMM-68 的下巴**高 24px**（我方 bbox h/w≈0.85，
  原版 0.966）。y1 眉毛锚定（y−0.15w）下，我方下巴在 256 DINet 空间里位于 y171，
  原版在 y195 —— checkpoint fuse_mask 是为 rel≈0.75 的下巴位置调制的，错位后渲染
  下巴与真下巴叠加成双下巴，且 mask 下半椭圆边缘露出"换脸椭圆"。
- **实证链**：hook7 dump 原版 crop 源图（437×437）→ 模板匹配得原版 rect
  (342,556,437)（公式验证 ✓）；原版 din arg1 dump 多尺度模板匹配得 DINet 输入
  内区 (350,564,421)，422/437=256/266 ✓（`[5:-5]` 切片实证）；下巴 rel 位置
  原版 0.745 vs 我方 0.66。
- **修复**：y1 从眉毛锚定改为**下巴锚定** `y1 = y+h−1.116w`（下巴下留白 0.384w，
  与原版公式在其 h/w=0.966 几何下严格等价——原版的 y−0.15w 本质就是下巴+0.384w）。
  man1/man2/woman chin 特写复核无双下巴；demo1 热点 352→80，全套件 in-region ≤ 原版。
- **产物**：`/opt/projects/opencode/tmp/hookdump_man1/`（原版 crop/mask npy），
  分析脚本 fix_match.py / match_dump256.py / analyze_hook7.py（tmp 目录）。

### V26 精确 landmark/mask 链（2026-10-28 深夜，用户坚持要与原版完全一致）

**方法论突破**：不再静态读反编译，改用**运行时黑盒实证**——
1. `face_detect.so` 的 convert98to68 通过模块 dict 查找（`__Pyx_GetBuiltinName`），
   可用 sitecustomize 包装拦截（hookdir8）：拿到 636 帧真实 (98→68) 输入输出对，
   **精确解出映射**：jaw=98[0,2,…,32]；眉=98[33..37]+[42..46]；鼻=98[51..59]；
   眼=60, avg(60,62), avg(62,64), 64, avg(64,66), avg(60,66) | 68, avg(68,70),
   avg(70,72), 72, avg(72,74), avg(68,74)；嘴=98[76..95]（err=0.0000）。
2. **PFLD 裁剪规则**（hook9 dump 输入源图+模板匹配）：bbox 左右+0.1w、下+0.1h、
   上不变 → resize 112 各向异性。**原版的 68 点就是 PFLD-98 转的，不是 3DMM！**
   （hp=Headpose(WHENet model_float32.onnx) 只出 yaw/pitch/roll 用于 pose_threshold 门控。）
3. **mask 黑盒辨识**（直接调用原版 `DigitalHumanModel.get_face_mask(img, lm68_int32)`，
   逐点扰动/分组缩放）：
   - hull 输入 = **lm68[2:15]（下颌去太阳穴端点）+ lm68[29]（鼻尖）= 14 点**
     （与 trace 的 14 点多边形一致）；
   - 椭圆 = 下半 0-180°：**cx=下颌中心 x, cy=下巴(8).y−34, axes=下颌宽×0.35**，
     **完全不响应嘴部点**（之前"嘴宽×1.5/下唇+0.3"的拟合全错）；
   - dilate 6×6×1。
   - 重建 68 点调用原版函数 → 与 dump mask **IoU=1.0**；我方重实现 IoU=0.989-0.993。
4. 矩形公式恢复原版原样（y1=y−0.15w，不再需要下巴锚定 hack——68 点下巴位置天然正确）。

**验证（全套件）**：热点 man1 190 / man2 **0** / woman 355（旧 1177）/ demo1 186 /
demo2 254 / demo3 116；man1/demo3 下巴特写无双下巴无下颌缝；in-region 闪烁 ≤ 原版。

**分析脚本**（/opt/projects/opencode/tmp/）：solve_68map.py, solve_eyes.py,
test_orig_mask.py（IoU 1.0 验证）, probe_mask*.py（mask 黑盒辨识）,
verify_mask_iou.py, match_pfld.py（pfld 裁剪模板匹配）, fit_ellipse68*.py。

### V25.2 mask 精修（2026-10-28 晚，用户报 demo3 0.2s 下颌宽/换脸椭圆边界）

- **对照实证**（mask_cmp2.py：hook7 dump 的原版 mask npy 映射进我方画布）：
  原版 mask 顶边 y~103、底边 y~238、x[48,214]；V25.1 我方顶边 y~60（过高，
  鼻梁区被覆盖产生重采样带/宽下颌观感）、底边 y~211（椭圆偏上偏小）。
- **规则修正**（全部从原版 dump 数据标定）：
  - hull 输入 = `jaw[4:29] + nose[53:60]`（原版 3DMM 下颌端点在眼/耳水平
    canvas y~105；我方 PFLD 端点 0/32 在太阳穴 y~63 必须截掉；鼻部从鼻梁
    中部 53(y~107) 起，对应原版顶顶点）；
  - 椭圆中心 = (hull 中心 x, **下唇均值 + 0.3×(下巴−下唇)**)——原版中心在
    下唇/颏顶水平（之前误以为嘴顶）；axes = **嘴角宽 |lm76−lm82|**。
  - 修后 mask: y[103,235] x[46,218] vs 原版 y[103,238] x[48,214] ✓
- 全套件复跑：demo3 热点 72、demo1 75、man1 119、man2 376、woman 1177
  （热点主要在嘴部内容区，目检下颌/边缘全部干净，in-region 均 ≤ 原版或持平）。

### V25 与原版仍有的小差异（已知、可接受）

- 我方 PFLD-98 下颌点比原版 3DMM 68 点略宽（rect w 306 vs 292）→ crop side 大 ~5%
  （459 vs 438），DINet 输入脸略小；不影响稳定性。
- 贴回 feather 用 GaussianBlur σ1.5 近似原版的 blendImages GPU featherAmount=0.1。
- GFPGAN 门控（chaofen_src 的 blur 检测 need_chaofen_flag）仍未实现：demo1 类模糊
  素材原版会开复原，我们恒关（restore_source CLI 可手动开）。

### V25 之前的历史（V24/V23…见下文）

---
> 用户报告 woman 0.12s/0.21s 脸变胖、1.03s 嘴型不一致、嘴部闪烁。经对原版做 ort 级
> 全量 hook（SCRFD/PFLD+attr/3DMM/GFPGAN 逐会话 dump）+ 三素材 DINet I/O dump 对拍，
> 根因 = DINet crop 的驱动五点语义：原版在其自有的 112 框（非我们 REF_112×1.25）上跑
> PFLD，再用另一固定模板（非 TEMPLATE_256）warp 出 crop。V22 的 PFLD(ours)+TEMPLATE_256
> 对 woman 渲染 MAE 0.147/嘴区 0.098（目视嘴型错、脸偏瘦）。修复 = 两个 dump 拟合的固定
> 常数：T112_REF（112 对齐模板）+ TEMPLATE_CROP（crop 模板），渲染 MAE → 0.053/0.033，
> 全片嘴区 MAE p95 6.95→4.94，1.03s 嘴型与原版一致。教训⑥：TEMPLATE_256 当年 SIFT 拟合
> 的是另一个人，"crop MAE 0.0074"只对那个人成立——跨人必须重新对 dump 验证）**

---

## 当前状态（2026-10-20 定版 V23）——新 session 从这里读起

### 一句话现状

**V23 = V22 的 DINet crop 几何重定标版**。逐帧链：SCRFD → PFLD-98（在**原版 112 框**
T112_REF 上跑）→ 五点(96,97,54,76,82) → warp(帧, 五点→**TEMPLATE_CROP**) 得 256 crop
→ DINet → fuse → M⁻¹warp + 椭圆贴回。全部常数 dump 拟合、无需逐视频标定。
6 素材对比视频已重跑（dev/data/*_compare.mp4，左输入|中原版|右新版，音频 man_cn.wav）。

### 诊断要点（本轮全部实测）

- **音频与渲染网络无问题**：同 clip dump 对拍音频窗 MAE 4e-5；DINet 等价性 0.0007（旧结论复核仍成立）。
- **原版每帧 ort 会话**（hookdir2 全量 dump， woman/demo1/man1 三素材）：
  SCRFD 640 ×2、**PFLD+attr 合一模型 112 ×1**（输出 landms 196 + attr 3）、3DMM 224 ×1
  （输出 120+66）、GFPGAN 512 **只在子进程跑且受门控**（主进程仅启动 warmup 1 次——
  woman/man1 门控关=裁剪自原帧，demo1 门控开=crop 明显复原，lapvar×1.49）。
- **V22 crop 链的错误**：我们的 PFLD 在 REF_112×1.25 上跑（原版不是），模板用
  TEMPLATE_256（原版不是，它是当年对另一个人 SIFT 拟合的）。结果：同一张脸在 crop 里
  大小/位置系统性偏差 → DINet 输入分布偏移 → 嘴型幅度/形状错 + 脸型偏瘦。
- **正确链（已落地）**：`five = PFLD(warp(帧, fit(scrfd_kps→T112_REF)))` 映射回帧坐标
  （与原版 dump landms 残差 ~1.3px）；`crop = warp(帧, fit(five→TEMPLATE_CROP))`。
  T112_REF/TEMPLATE_CROP 为固定常数（pipeline.py 顶部，拟合自 woman dump、man1 交叉
  验证 eyeDist 76.7 vs 75.6）；demo1 因门控开（复原 crop）仍残 ~0.12（未回归，后续做门控）。
- **渲染级验证**（喂同一音频 dump，DINet out vs 原版 out）：woman 0.147→0.053、
  嘴区 0.098→0.033；man1 0.079（新链）/0.053 全帧。全片对比：woman 嘴区盒 MAE
  p95 6.95→4.94，mean 4.58→4.18。

### 与原版的一致性（全部有实证）

| 环节 | 状态 | 依据 |
|---|---|---|
| 音频前端/窗口 | ✅ | MAE 4e-5（含 30fps audio-limited 重采样，本批新验证） |
| DINet 网络 | ✅ | 同输入 MAE 0.0007 |
| DINet crop 几何 | ✅ 重定标 | T112_REF+TEMPLATE_CROP，渲染 MAE 0.053（V22 0.147） |
| SCRFD/选脸 | ✅ | <0.4px |
| chaofen 门控 | ❌ 未实现 | 原版 GFPGAN 子进程+门控；默认关=对 sharp 素材正确；demo1 类模糊素材会偏软 |
| 贴回椭圆参数 | 🔧 近似 | (104,116)σ6 估计值；新 crop 尺度变了，建议用 vid dump 重标 |
| 输出帧数 | 🔧 | 318 vs 316（音频窗口边界，旧遗留） |

### 下一步（按优先级）

1. **chaofen 门控**（demo1 已实证需要）：原版的门疑似由 attr(1,3) 输出或 blur_threshold 控制；
   需要时再 hook 子进程（spawn 子进程不继承 sitecustomize 的 patch——本轮 ort hook 只抓到
   主进程会话，GFPGAN 全程未现）。门控开时的复原画布也可能不是 mtcnn_512（未验证）。
2. **贴回椭圆 (104,116) 在新 crop 几何下重标**（crop 脸大了 ~5%，椭圆应同步放大）。
3. TEMPLATE_CROP 三人联合精拟合（现 woman/man1 平均；demo1 门控实现后再纳入）。
4. 输出帧数 318 vs 316 对齐。

### 关键资源（本轮新增）

- 服务器 dump：`dump_ort_{woman,demo1,man1}/`（ort 全量：scrfd/pfld/mm3d/gfpgan/din 五类 IO）、
  `dump_{woman,demo1,man1}_orig/`（DINet I/O）、`/opt/projects/opencode/tmp/{woman,man1}_fmt.mp4`。
- hook：`hookdir2/sitecustomize.py`（按输入 shape 分类 dump 所有 ort 会话 + DINetV1.forward）。
- 验证脚本：`/opt/projects/opencode/tmp/`（fit_T112_all.py / universal_T.py / final_univ.py 等）。

---

## 0. 一句话现状
﻿
> 📌 **2026-10-19 状态 6（V21：回退到已验证基线）**：V14-V20 的 roundtrip 化被证明是弯路——**原版 DINet crop 就是
> PFLD-98 五点 + TEMPLATE_256 对齐**（dump 实证 crop MAE 0.0074，眼在行 55）；roundtrip 链（bbox 裁块/recover/对齐图
> resize）会让 DINet 输入的脸比原版大一倍（与 dump arg1 对比 MAE 0.27）→ 渲染畸形。V21 = 用户已确认的 V13 组合：
> SCRFD→PFLD(96,97,54,76,82)→Align256(TEMPLATE_256)→DINet→fuse=fake×blur17(fuse_mask)+rgb×(1-M)→M⁻¹warp+face_fg_crop
> 椭圆贴回。教训⑤：**用户已确认的版本就是 ground truth，不要为了"更忠实的复刻"推翻已验证的基线**；中间环节的
> 验证标准 = 与原版运行时 dump 的像素级对比（如 DINet 输入），而不是推理链"看起来合理"。﻿
> 📌 **2026-10-19 状态 5（V19：通道链修正）**："脸糊/蜡像"的根因是**通道序**——原版内部链全程 **BGR**（帧原生）：
> GFPGAN.forward 的切片是通道翻转（输出 BGR，通道均值 B>R 实证），recover 混合 BGR restored 与 BGR roi，
> 只有 DINet 输入处做一次 [..., [2,1,0]] 翻成 RGB（dump2 arg0/arg1 脸区 R>G>B 实证），输出再翻回 BGR 贴帧。
> V16 误用全 RGB 链 → restored 与 roi 通道错位 → 色调/对比度异常（看起来像糊）。V19 全 BGR 链修复。
> 教训④：通道序必须以**运行时数值实证**（通道均值/dump）为准，伪码里的 [2,1,0] 这类片段要结合前后文判断。﻿
> 📌 **2026-10-19 状态 4（V16：行为级完整移植）**：停止假设式补漏，改为**直接执行原版编译模块**（HG/*.so @ heygem38）
> 提取每个函数的精确行为，再照抄。原版环境的 cv2box 0.5.9 源码在 heygem38/site-packages（recover_from_reverse_matrix、
> common_face_mask、blob 全部有源码）。逐组件实测结论：
> - apply_roi_func = **纯 bbox 裁块**（无扩边、非正方），roi_kpss = kps - roi_origin
> - norm_crop('mtcnn_512') 模板（512画布实测）：[(183.94,246.06),(330.45,245.53),(250.60,313.57),(193.34,392.73),(322.16,391.04)]
> - **warpAffine 双向语义**：放大 warp（roi→512）用正向拟合 A；recover 的缩小 warp（512→roi）用 inv(A)。原版 get_single_face
>   返回的 "mat_rev" = inv(A)（recover 消费它，roundtrip 实测 diff 0.05）
> - recover = warpAffine(restored, inv(A), roi尺寸, BORDER_REPLICATE) 与 common_face_mask(roi尺寸) 混合（椭圆(256,256)
>   轴(220,160)@90°、边缘20清零、stackBlur 201）——mask 不随矩阵 warp
> - **DINet 输入 = RGB**（dump2 arg0/arg1 脸区通道 R>G>B 实证，伪码的 [2,1,0] 解读有误）
> - GFPGAN：blobFromImage(直接resize512, 1/127.5, (127.5)*3, swapRB=True) → (out+1)/2 → [0,1] RGB
> - SCRFD 定制封装与 insightface 标准实现逐点一致（<0.4px）
> 三个移植陷阱：① [:, :, ::-1] 视图的负 strides 会让 warpAffine 静默输出黑图（必须 ascontiguousarray）；
> ② warpAffine 矩阵方向（放大用正向/缩小用逆向）③ 通道序必须以运行时 dump 实证。
> demo1 前 8 帧（原变形场景）V16 与原版视觉一致，full MAE 1.87。> ?? **2026-10-19 ?? 3?V15????????**?demo1 ? 0.18s ????????roi ????? clamp ?
> ?????resize(256) ???????? apply_roi ???????????clamp ?????????
> ????roi ??????????????????? clamp ???
﻿
> 📌 **2026-10-19 状态 2（V14：单帧"胖脸"消除）**：woman 等视频出现"偶发单帧脸变胖/嘴型异 then 恢复"
> （1.05s/1.07s 等）。根因 = V13 的 DINet crop 由 PFLD 五点驱动（Align256 warp）：逐帧检测的离群五点直接变成
> 该帧 crop 的缩放突变。原版天然免疫：crop=recovered[roi_box] 且 roi 由 bbox 驱动（稳），norm_crop 的正向 M
> 与逆向 mat_rev 使五点噪声在修复贴回时自相抵消，五点根本不进 crop 几何。
> **V14**：roi=正方形 L=SCRFD bbox 高、垂直锚=PFLD 眼中心(瞳稳±0.6°) → crop=resize(recovered[roi][5:-5],256)
> → DINet/fuse → 矩形硬贴回同一 roi（块边缘=recovered 在 fg mask 外的原始像素，无缝）。
> 验证：woman 嘴区帧间跳变 V14 max7.06/p99 4.96 vs 原版 6.60/5.23（无异常峰）；嘴型与原版逐帧一致。
> 全部 6 素材 A/B 已重跑至 dev/data/（*_compare.mp4，左输入|中原版|右V14，音频 man_cn.wav）。
> 教训②：几何驱动源必须选逐帧稳定量（bbox>五点>眼嘴距——眼嘴距随张嘴大幅变化，绝不能做 roi 尺度）。﻿
> 📌 **2026-10-19 状态（V12：接缝与嘴型同时闭环）**：10-18 的"完全复刻 SCRFD 链"版本引入三个连锁 bug，
> 经 dump 逐一定位修复（V3→V12，每个版本用户/数据双重验证）：
> ① **fuse_mask 方向反**（实测 mask 嘴区=1.0=渲染区；正确 fuse = fake*M + ref*(1-M)）——修复前嘴区输出=原始crop
> （"和生成之前一样"）；② **DINet crop 几何**须与 input_mask 匹配（眼<行86）；③ **贴回方式**：原版 get_complete_imgs
> 实为"recover_from_reverse_matrix（修复内容按 face_fg 贴回帧）+ roi_box 矩形硬贴"（无 warp、无椭圆 mask），
> HeyGem12 用 warp+face_oval 贴回会在额头画线。
> **V12 最终方案**：SCRFD(det_score最大,tracking=False) → get_single_face(mtcnn_512,apply_roi) → GFPGAN
> (x-127.5/127.5) → mat_rev(+roi原点) 按 face_fg_512 贴回帧得 recovered → PFLD-98五点(96,97,54,76,82) on recovered
> → Align256(TEMPLATE_256) crop → source=crop×input_mask/ref=crop → DINet → fuse=fake×blur17(fuse_mask)+ref×(1-M)
> → M⁻¹warp 贴回 + face_fg_crop 椭圆 mask（边缘两侧皆帧内容故无痕）。
> **A/B（HG/1004_format 全长）**：full MAE 1.43、嘴区 5.91、无接缝无结痕、嘴型随音频。
> 教训：逆向伪码的公式方向/坐标系必须以运行时 dump 实证为准（fuse 公式/roi 坐标系两次被伪码误导）。﻿
> 📌 **2026-10-18 状态（原版复刻改造）**：放弃"PFLD+TEMPLATE_256+时域稳定+校准"路线，
> **完整复刻原版检测链**（Ghidra 逆向 face_align_5_landmarks / trans_dh_service /
> digitalhuman_interface 后重写）。新版管线：SCRFD(insightface语义,640,每帧独立检测,
> tracking=False) → 面积选最大脸 → apply_roi(0.5扩边) → mtcnn_512模板对齐 →
> GFPGAN修复(输入(x-127.5)/127.5,输出[-1,1],无保护区) → crop[5:-5] resize 256 →
> source=x/255×input_mask, ref=同帧未掩码x/255 → DINet → fuse_res=ref×fuse_mask+(1-fuse_mask)×fake →
> 贴回512 crop → mat_rev warp到ROI矩形 → 脸椭圆贴回帧。
> 新增 `face/scrfd_chain.py`；`pipeline.py` 重写；移除稳定链/R校准/k校准/gain-match/taper/保护区/render_ema；
> 编码对齐 crf15+输入re-encode+关TF32。
> **A/B 验证（同素材 HG/1004_format, 全长318帧 vs HG/1004-r.mp4）**：全帧MAE **1.42/255**，
> 嘴部区MAE 5.37/255(p95=6.5)，嘴部运动帧间差相关 **0.9846**，重影消除（对比第8轮嘴部MAE 0.0205）。
> 遗留：输出帧数 318 vs 原版 316（音频窗口边界微差）；oval 用 dump 反解 mask。
---

## 历史状态（2026-10-19 定版 V22，已被 V23 取代）——过程记录， §V23 为权威

### V22 一句话现状

**HeyGem12 已定型为"无需 .so 的原版 HeyGem 行为级复刻"**。V22 是用户确认输出合格的版本：
6 个测试素材全帧 MAE mean 1.4~2.3（vs 原版输出），max≈mean 无离群坏帧，无接缝/变形/糊脸/嘴型漂移。
下方的 10-18/10-19 各条目是调试史（含失败弯路），**本节是权威现状**；冲突时以本节为准。

### V22 管线（heygem12/pipeline.py）

```
输入 re-encode（h264 crf15 + wav 16k，HEYGEM12_SKIP_FORMAT=1 可跳过）
→ SCRFD(640, tracking=False, 每帧独立, det_score argmax 选脸)
   [face/scrfd_chain.py: insightface 语义封装，与原版定制封装逐点差<0.4px]
→ PFLD-98 → 五点 (96,97,54,76,82)   [face/landmarks.py]
→ Align256 (TEMPLATE_256) 256 crop   [face/align256.py]
→ restore_crop = chaofen_before：GFPGAN(arcface-512画布, LMEDS) → A=M·M512⁻¹ 映射进 crop
    （含瞳孔/颈部保护区）
→ source = crop×input_mask, ref = crop（未掩码）, audio = WeNet 窗口（T=20）
→ DINetV1 → fake
→ fuse = fake × stackBlur17(fuse_mask) + rgb × (1 − stackBlur17(fuse_mask))
→ 贴回：M⁻¹ warp + face_fg_crop 椭圆 mask（face_fg_crop=(128,128)轴(104,116),σ6）
→ libx264 crf 15 输出
```

### 与原版的一致性（全部有实证，非推测）

| 环节 | 状态 | 依据 |
|---|---|---|
| 音频前端/窗口 | ✅ 一致 | MAE 5e-5、corr 1.0、152/160 位精确（HANDOFF §历史） |
| DINet 网络 | ✅ 等价 | 同输入渲染 MAE 0.0007 |
| **DINet crop 几何** | ✅ 一致 | **原版 dump 实证 MAE 0.0074**（眼位行 55，PFLD+TEMPLATE_256） |
| fuse 公式/mask | ✅ 一致 | dump out-vs-ref：ref 区 0.0046/渲染区 0.026；fuse_mask 嘴=1（实测数值） |
| SCRFD/选脸 | ✅ 一致 | bbox/kps 差<0.4px；det_score argmax |
| GFPGAN 预处理 | ✅ 一致 | (x-127.5)/127.5 入、[-1,1] 出 |
| chaofen_before | 🔧 工程等价 | 原版走 mtcnn_512→recover 复原帧；我们走 Align256→GFPGAN→A 映射。输出 corr 0.99/MAE 0.0145 |
| 贴回椭圆 mask | 🔧 近似 | 方式与原版相同（vid dump 实证原版=椭圆硬贴），参数 (104,116) 是估计值，**可标定** |
| 模糊门控/4K跳过 | ❌ 未实现 | 原版 face_attr 门控（need_chaofen_flag）；标准素材无影响 |
| 数值环境 | 🧮 不可消除 | torch 2.5 vs 1.11（TF32 已关）；逐像素一致物理上不可能 |

### V14-V20 弯路的教训（不要重蹈）

1. **dump 实证是唯一可靠的验证标准**——roundtrip 链"看起来更像原版逻辑"，但 DINet 输入与 dump 对比 MAE 0.27（脸大一倍），
   而用户确认版的 crop 是 0.0074。判断依据 = 与原版运行时 dump 的像素对比，不是推理。
2. **用户确认的版本就是基线**——V13 已被确认合格，V14-V20 为"更彻底复刻"推翻它导致连续回归。
3. **通道序以数值实证为准**（原版 restored 通道均值 B>G>R → 全 BGR 链；伪码的 [2,1,0] 片段曾误导两轮）。
4. warpAffine 矩阵方向（放大用正向拟合/缩小用逆）、`[::-1]` 视图的负 strides（warpAffine 静默黑图）是高频坑。
5. cv2.stackBlur 对 float 输入保持 0..1，不要再 /255。

### 下一步（按优先级）

1. **贴回椭圆参数标定**：用原版最终帧 dump（HANDOFF §P1 的 9 帧 vid dump，face_oval_256 同源方法）反解精确椭圆，
   替换 face_fg_crop=(104,116)。预期再降 10-20% 差异。
2. **输出帧数对齐**：我们 318 vs 原版 316（音频窗口 n_out 边界）。
3. 可选：need_chaofen_flag/4K 门控（face_attr.so 已反编译，逻辑见 dev/so_decompilation）。

### 运行与验证

```bash
# 服务器：跑一版（GPU 1；GPU 0 被 root 服务占满）
cd /opt/projects/heygem12
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=1 TMPDIR=/opt/projects/opencode/tmp \
  /opt/anaconda3/envs/heygem312/bin/python -X utf8 -u -m heygem12.cli \
  --video assets/media/X.mp4 --audio assets/media/man_cn.wav --output out.mp4 --device cuda

# 原版对照（HG 服务，py38；注意 work() 返回后 1004-r.mp4 还要等写完，且需完整 ffmpeg）
cd HG && env -u LD_LIBRARY_PATH /opt/anaconda3/envs/heygem38/bin/python run.py \
  --audio_path ... --video_path ...   # PATH 需前置 imageio ffmpeg（系统 ffmpeg 无 libx264）

# 批量 A/B（含原版缓存复用）：/tmp/batch_v14.sh → dev/data/*_compare.mp4
# 关键 ground truth：
#   dump2/dump3/    原版运行时 DINet 输入/输出 dump（npz：arg0=source, arg1=ref, out=fuse 结果）
#   HG/1004-r.mp4   原版最终输出；HG/ 目录 = 原版完整部署（含 .so 和 cv2box 0.5.9 源码环境）
#   dev/data/_work/ orig_*/new_*.mp4 = 各素材两侧原始输出
```

### 关键文件

- `heygem12/pipeline.py`（V22 定版主流程）、`heygem12/face/scrfd_chain.py`（SCRFD+检测链）
- `heygem12/face/align256.py`（TEMPLATE_256/Align256，未改动）、`heygem12/face/landmarks.py`（PFLD）
- `heygem12/models/dinet_hd.py`（DINet，已验证等价）、`heygem12/audio/frontend.py`（已验证等效）
- 服务器：`/opt/projects/heygem12/HANDOFF.md`（本文）、`dev/so_decompilation/`（33 个 .so 的符号/反汇编/伪码产物）
- 反编译环境：服务器 Ghidra 12.1.4 + REA（`export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64`），
  大函数分析需 `export TMPDIR=/opt/projects/opencode/tmp`（/ 分区常年只剩 ~50MB，/opt 有 50G+）
## 0. 一句话现状

**端到端与原版的对齐已进入 "接近像素级"**：同素材 248 帧，嘴部框 MAE **0.0205**
（第七轮 fuse 方案 0.0468，第六轮 0.0468 的视频实际是静态嘴）；
PFLD 唇高序列 corr **0.82@lag0**（时序无偏移；"lag+20" 是旧 dark-fraction 指标的伪影，已翻案）。
**原版真实管线（第八轮 dump 逆向 + 第十二轮 ref 实证，全部实测）**：
SCRFD(640) 检测 + PFLD(112) 关键点（每帧各 2 次）→ **GFPGAN 复原源帧**
（画布用带噪声的检测器五点对齐，输入 (x-127.5)/127.5，输出 [-1,1]）→ 从**复原后的帧**取 DINet
输入 crop（arg0/ref 带域 lapvar 88.9 vs 原始 35）→ DINet 渲染（**arg1 ref = 同帧 crop
未遮罩**，非固定参考帧——第十二轮 11j 逐位实证）→ **合成 = 源帧 + 脸椭圆掩码
×(渲染−源帧) 硬贴**（掩码在 crop 空间稳定 ~45%，"nblend"），**GFPGAN 内容不进成片**、
无 face_parsing/xseg（本轮 run 未触发）、无时间域 smooth、**无 chaofen_after**。

**我们已实现的对应结构**（`pipeline.py` 当前默认）：chaofen_before 开（`restore_crop`，
GFPGAN(源帧)→512→A=M·M512⁻¹→256 crop）、chaofen_after 关、**ref = 每帧自己的复原 crop
未遮罩**（11j 起）、合成用 canonical 脸椭圆硬贴（`align256.paste_back_oval`，
`face/data/face_oval_256.npy`，由 9 帧 vid dump 的 consensus 掩码 th0.7+σ8 羽化）、
关键点链 = YuNet 检测 → PFLD-98 五点 → **R 眼距语义标定**（11g）→ 时序稳定
（median3 + 双帧对插值 + 硬切换 EMA + 尺度锚定，11a-11e）→ **垂直语义标定
`_vcalib_keypoints`**（11l，SCRFD 中位数 k，自适配每人）。`--restore` /
`--no-pre-restore` / `--fuse-mode oval|fuse_mask` / `--no-kps-stabilize` 可切回旧行为。

⚠️ 前七轮的多数"结论"都建立在错误实现或无效指标之上，§7 列出了**已被推翻**的旧结论，别照着做。

> 📌 **2026-10-02 状态**：抖动类问题全闭环（11a-11o）。10-01 修复露齿帧"红唇团"
> （11j ref 方案）、"双下巴"（11l 垂直语义标定）、颈部接缝（11m mask 渐隐 +
> vcalib 阈值）、颈部 GFPGAN 爬斑（11n 保护区）；10-02 定案 60fps 音频窗口重采样
> （11o）。标准 9 条 + 泛化批 18 条（test_media man/woman，60fps）全量重跑并与
> 原版逐帧对齐验证。当前与原版剩余差异 = ①个别帧张嘴幅度略大、②运动模糊帧
> 4-8px 漂移（原版同样存在）、③颏下 ~1-2px 内容级残余——均非系统性 bug。

---

## 1. 🔑 决定性资源：原版 Linux 服务器（可直连）

### 连接方式（PuTTY）

**用本机 `D:\soft\putty` 里的命令行工具连接**（`plink`/`pscp`，兼容 OpenSSH 语法）：

```
主机    jinyun@10.0.70.127      （内网直连；PuTTY 保存的会话 60.190.99.5:6122 是同一台的 NAT 入口）
密码    123456                  （jinyun，可 sudo，sudo 密码相同）
```

```powershell
# 执行远端命令
D:\soft\putty\PLINK.EXE -ssh -pw 123456 -batch jinyun@10.0.70.127 "<命令>"

# 传文件
D:\soft\putty\PSCP.EXE -pw 123456 -batch <本地文件> jinyun@10.0.70.127:<远端路径>
D:\soft\putty\PSCP.EXE -pw 123456 -batch jinyun@10.0.70.127:<远端路径> <本地文件>
```

⚠️ **PowerShell→bash 的引号/`$`/括号转义极易出错**（踩坑多次）：远端命令一律写成 `.sh` 上传到
`/opt/projects/opencode/tmp/` 再 `bash xxx.sh`；后台任务用
`nohup bash xxx.sh >/dev/null 2>&1 &`（plink 断开会杀前台进程）。本地 .venv 有 paramiko 可用。

### 服务器环境

```
工作目录  /opt/projects/heygem12         (本项目在服务器上的副本 + dump 产物)
上一版本  /opt/projects/HeyGem12-v1     (V22/更早的 heygem12，三方对比用)
原版部署  /opt/projects/heygem12/HG     (1.5G，含全部权重和 .so)
conda     /opt/anaconda3/envs/heygem38   (原版运行环境, Py3.8 + torch 1.11)
conda     /opt/anaconda3/envs/heygem312  (★本项目运行环境, Py3.12 + torch 2.5.1+cu121 + onnxruntime-gpu 1.19.2)
GPU       2× RTX 4090 D 24G —— GPU 0 被 root 服务占满, 所有任务必须 CUDA_VISIBLE_DEVICES=1
```
- ⚠️ **服务器上的 ffmpeg 是坏的**：`/usr/local/bin/ffmpeg`（n5.1.2）**不支持 `-crf`/libx264**，
  而它在 PATH 里比 `/usr/bin/ffmpeg`（4.4.2，正常）更靠前。原版服务用裸 `ffmpeg` 调，
  于是转码直接失败（`[format video error]`，0.014s 就退出）。
  修复：`mkdir -p /opt/projects/heygem12/bin && ln -sf /usr/bin/ffmpeg /opt/projects/heygem12/bin/ffmpeg`，
  然后 `export PATH=/opt/projects/heygem12/bin:$PATH` 再跑（见 `run_hooked3.sh`）。
- ⚠️ 原版会先用 ffmpeg 把输入转成 `./1004_format.mp4` / `./1004_format.wav`；
  **部署目录里的这两个文件属于 root，jinyun 写不进去**（`Permission denied`），
  转码静默失败后会沿用**同名旧文件**跑完全流程 —— 本轮因此白跑了一大圈，
  对比结果全部失真。要在原版上跑自己的素材，必须先把部署目录 `chown` 或复制一份可写副本。
- 原版运行日志里每帧都有 `-> chaofen cost:0.08s`，配置项有
  `chaofen / chaofen_before / chaofen_after / chaofen_src / need_chaofen_flag / blur_threshold`。

**已抓到的原版中间产物**（历史 dump 命名约定 `arg0=源(masked) / arg1=参考 / arg2=音频 / out=渲染`，
DINet batch=4 → 每 call 4 样本）：

```
# ★ 本轮（2026-10-20, V23 诊断）新增 —— 在服务器上：
/opt/projects/heygem12/dump_ort_{woman,demo1,man1}/   ort 级全量 hook（hookdir2）
    scrfd/  每会话 in0(640)+out*   SCRFD 原始 IO（2 次/帧）
    pfld/   in0(112)+out0(1,3)+out1(1,196)   PFLD+attr 合一模型（1 次/帧）：
            in0=喂给 PFLD 的 112 crop（原版自有对齐框）；out1=98 点 landmark（关键 ground truth！）
    mm3d/   in0(224)+out0(1,120)+out1(1,66)  3DMM 回归
    gfpgan/ 仅启动 warmup 1 次（噪声输入）——真正的 GFPGAN 在子进程跑，hook 抓不到
    din/    arg0/arg1/arg2/out (4,...)        DINetV1.forward（前 40 call）
/opt/projects/heygem12/dump_{woman,demo1,man1}_orig/  DINet I/O dump（hookdir v1，160 帧/素材）
    注：早期 dump2/dump3/dump_d1/dump_m6/dump_m15e 已因磁盘满删除（/ 100%、/opt 99%）。
# 历史（多数已被删或覆盖，仅作记录）：
/opt/projects/heygem12/dump/vid_*.npy        最终合成帧 (1080,1920,3) uint8（第八轮 VideoWriter hook）
本地 Temp 副本（C:\Users\faye\AppData\Local\Temp\opencode\pull\ 等）可能已被清理。
```

**hook 方法**：`hookdir/sitecustomize.py`（v1：只 patch DINetV1.forward）与
`hookdir2/sitecustomize.py`（v2：按输入 shape 分类 dump **所有 onnxruntime 会话**的
InferenceSession.run + DINetV1.forward）。用法：`PYTHONPATH=<hookdir> HOOK_DUMP=<目录>`
加进原版 run.py 的环境。注意：
- `DINetV1` 是普通 Python 类，`cls.forward` 可直接替换；但 **不要**包装
  `build_model` / `load_ppg_model`（typeguard 的 `sys._getframe` 会炸）。
- **GFPGAN 会话在子进程创建**（spawn 的子进程不重新 import sitecustomize 里的 patch 时序
  可能漏抓——本轮 v2 hook 全程只抓到 1 次 warmup 调用，确认 GFPGAN 不在主进程）。
- 抓原版 run 前必清 root 拥有的 `HG/1004_format.*` / `HG/1004-r.mp4`（`sudo rm`），
  否则转码静默失败、沿用旧素材（坑过多次）。

---

## 2. 本轮修掉的三个致命错误

### 2.1 🔴 网络结构错 5 处（`models/dinet_hd.py`、`models/blocks.py`）
在服务器上 `repr(DINETV1(3,3,256))` 直接看到真结构。**所有错误的块与被误用的正确块
参数形状完全相同**，所以 416-key `strict load` 一直通过、毫无报错：

| 位置 | 真实 | 我们曾写 | 后果 |
|---|---|---|---|
| `source_in_conv[0][1]` | **Down**Block2d(64,128,k3,s2) | SameBlock2d(64,128,k3) | **少一次降采样**，全网络特征分辨率大 2× |
| `ref_in_conv[0][1]` | 同上 | 同上 | 同上 |
| `trans_conv[8]` | **Down**(256,256,k5,s2) | Same(k5) | 降采样位置错 |
| `trans_conv[9]` | **Same**(256,256,k3) | Down(k3) | 同上 |
| `out_conv[0][1]` | **UpBlock2d**(512,512) | SameBlock2d + 手动 interpolate | 卷积跑在错误分辨率上 |
| `out_conv[1][0]` / `[2][0]` | **UpBlock2d** | SameBlock2d + 手动 interpolate | 需去掉手动 interpolate |

真实结构（已全部落地，`repr` 逐字核对）：
```
source_in_conv / ref_in_conv : ModuleList 3× [SameBlock2d, DownBlock2d]
    (3→64 k7) → Down(64→128) → Same(128→128) → Down(128→256) → Same(256→256) → Down(256→512)
    H → H/2 → H/4 → H/8
trans_conv (11): Same(1024,256,k3) Same(k11) Same(k11) Down(k3)
                 Same(k7) Same(k7) Down(k3)
                 Same(k5) Down(k5,s2) Same(k3) Down(k3)     ← Down 在 idx 3/6/8/10
audio_encoder: SameBlock1d(256,256,k5) Res Down Res Down SameBlock1d(k3)   ✓ 本来就对
appearance_conv_list: [[Same(512,512,k5) + 4×Res], [4×Res]]                 ✓ 本来就对
adaAT: commn_linear(512→512)+ReLU, scale(512→512)+Sigmoid,
       rotation(512→512)+Tanh, translation(512→1024)+Tanh                   ✓ 本来就对
out_conv: [[Same(1024,512,k3), Up(512,512,k3), Res(512,512)],
           [Up(768,256,k3), Res(256,256)],
           [Up(384,128,k3), Res(128,128), Conv2d(128,3,k7,p3), Sigmoid]]
forward: x=out_conv[0](cat[sf,ref_trans]); x=out_conv[1](cat[x,s1]); return out_conv[2](cat[x,s0])
         ← 上采样在 UpBlock2d 内部（先 interpolate 再 conv），forward 里**不要**手动 interpolate
```
`UpBlock2d = interpolate(scale_factor=2) → conv → norm → relu`（在服务器上实测确认：
`allclose(u(x), u.relu(u.norm(u.conv(interpolate(x,2)))))` 为 True，反序为 False）。

### 2.2 🔴 wenet encoder 配置错：`concat_after` 必须是 **False**
上一轮从"checkpoint 里有 `encoders.N.concat_linear`"推断 `concat_after=True` —— **错**。
服务器上读出活对象：`ConformerEncoderLayer.ff_scale=0.5, normalize_before=True, concat_after=False`。
原版**无条件创建** `concat_linear` 但不用它，走标准加性残差 `x + dropout(self_attn(...))`。
**教训：权重键存在 ≠ 该路径被使用。**

同一份 mel 输入下：

| | 输出 std | vs 原版 MAE | corr |
|---|---|---|---|
| concat_after=True（错） | 0.09564 | 0.0486 | 0.757 |
| **concat_after=False（对）** | **0.10067** | — | **1.0000**（服务器）/ 0.9996（本地）|

逐层分歧证据：`embed` 输出两者**逐位相同**（MAE 0.000000），`layer0` corr 0.9986，
逐层累积到 final corr 0.766 —— 典型"每层残差公式写错"的特征。
修好后本地对原版特征 MAE **0.0018**、corr **0.99957**（残差 = 本地 CPU+librosa1.0
vs 服务器 CUDA/TF32+librosa0.8.1 的数值差；`concat_linear` 的 24 个键需按 unused 容忍，
见 `frontend.UNUSED_KEYS_PREFIX`，**按子串匹配**，因为键名是 `encoders.N.concat_linear.*`）。

### 2.3 🔴 音频前端配方 + 窗口映射（`audio/frontend.py`）
从 `compute_ctc_att_bnf.so` 的 `hparams1`（**不是** `_extract_feats.py` 里的 `hparams`）
+ 合成 wav 实验定标：

```
librosa.load(wav, sr=16000)                       # 单声道 + 重采样到 16k
→ 首尾各 np.concatenate(np.zeros(6400))           # add_silence_flag，0.4s×2
→ scipy.signal.lfilter([1,-0.97],[1], y)          # preemphasis
→ librosa.stft(n_fft=1024, hop_length=160, win_length=800, window='hann', center=True)
→ |S|**2 → filters.mel(sr=16000, n_fft=1024, n_mels=80, fmin=0, fmax=8000,
                       htk=False, norm='slaney')
→ 10*log10(mel + 1e-5) - 20                       # ref_db=20
→ ConformerEncoder(concat_after=False) → (N, 256) @ 25 fps
```
关键证据：
- `hparams1 = {..., 'hop_length': 160, ...}`（`_extract_feats.py` 里的 `hparams` 是 hop=200，**死配置**）
- 帧数公式 `25*(时长+0.8)-1`：4s→119、8s→219、16s→419、10.008s→269，**斜率严格 25fps**
- 8k/22k/44.1k/16k 合成 wav **帧数全同** → 内部统一重采样到 16k
- `lead=6400 & tail=6400` 时本地复现与服务端 `get_weget` **MAE 0.000000、逐帧 corr 1.0000**；
  lead 细扫 0…12800 唯有 6400 是极小值，偏移 ±1 帧 MAE 立刻涨到 0.036
- `norm` 必须显式传 `'slaney'`（librosa 0.8 起为默认，1.0 也是，但显式写可跨版本）

**窗口映射**：`window(video_frame i) = feats[i : i+20]`，即 `WINDOW_T=20, WINDOW_OFFSET=0`，
stride=1（特征 25fps == 视频 25fps）。用**原版真实处理的音频**（`1004_format.wav`）
做精确 MAE 搜索，跨 call 0/1/2/10/20/30 共 24 个样本，best_off 严格等于 `call*4+sample`。
⚠️ 上一轮得到的 "offset=+17" 是**拿错音频文件**搜出来的假象 —— 音频不对时任何偏移都不匹配，
argmax 毫无意义。**做这类反解必须先确认输入素材同源。**

### 2.4 其它已确认的事实
- `input_mask_re` = **全 1** → 参考图**不遮罩**（我们的处理正确）。
- 他们的 `src` 在 `input_mask==0` 处**严格为 0**（mean abs 0.000000）→ `rgb * input_mask` 正确。
- `training=False`（`model.eval()`）→ eval 模式正确，BN 用 running stats。
- 原版 batch = 4（4 帧一批），我们是 1；模型全卷积，等价。
- `wh = 0.97` 作为融合强度（`default_fuse_strength`）。

### 2.5 对齐关键点源与模板标定（`face/landmarks.py`、`face/align256.py`）
crop 几何先用 SIFT 反解定为恒等，但前提是**关键点源**正确：

- **YuNet 五点不能用于对齐**。同一静止讲话人，其眼连线倾角逐帧乱跳
  −0.446° / −0.610° / −3.268° / −3.318° / −3.356° / −3.816° / −4.130°，
  而 PFLD-98 瞳孔（96/97）稳定在 −0.594°~0.060°。用 YuNet 对齐会让 crop 逐帧抖几度。
  现改为：YuNet 只做人脸检测 → warp 出 112 crop → PFLD-98 → 取
  `FIVE_FROM_98 = (96,97,54,76,82)`（瞳孔/鼻尖/嘴角，正好对应 arcface 模板五点）。
- **模板常数 `TEMPLATE_256`**（`align256.py`）：用他们的 dump 与我们的 crop 做 SIFT 匹配反解
  （12 帧 × 3 个批次），得 scale 0.9003 / rot −1.671° / trans (5.5, 20.8)，
  反变换回模板即 `TEMPLATE_256`。验证：crop MAE **0.0474→0.0074**、corr 0.801→**0.9934**，
  SIFT 相似变换收敛到 **scale 0.9999 / rot −0.20° / trans (−0.1, 1.3)**（恒等）。
  眼行 55 < input_mask 首零行 86 < 嘴行 139 < 251 ✓。
  传 `template=` 可换模板；`template_256(K, center)` 仍能生成未倾斜的 arcface 变体做 A/B。

---

## 3. 与原版对数的验证结果（`dev/decompose_motion.py`）

同一固定 crop 空间嘴部框 `[121:185, 78:178]`，40 帧：

| | darkness std | mean\|Δ帧间\| | 范围 |
|---|---|---|---|
| A 原版 dump 的渲染 | 0.00698 | 0.00470 | 0.026–0.059 |
| **B 我们的模型吃他们的输入** | 0.00491 | 0.00322 | 0.034–0.053 |
| C 我们的模型吃**我们合成的输入** | **0.01226** | **0.00704** | 0.032–0.101 |

- **A vs B：逐像素 MAE 0.00198** → 网络实现完全等价（此项已结案）。
- **C/A 运动幅度 = 1.757** → 我们合成的输入产生的口型运动**比原版更大**，不是"嘴不动"。
- 我们裁的 crop 与他们差 **MAE 0.0784**（他们相邻帧之间只差 ~0.004）→ 这是**当前最大未解项**。

⚠️ **`corr(嘴部暗区, 音频rms)` 是无效指标**：原版自己的渲染也只有 **−0.072**（loud/quiet 0.91），
和我们（−0.073 / 0.99）一样。口型取决于音素而非响度（响亮的擦音反而闭嘴）。
要判口型请比**运动幅度**（std / mean\|Δ\| / 范围）或直接看视频。

---

## 4. 未解决（按影响排序）

### P0 逐帧嘴部形状的残差（已大幅收敛，剩余来源已定界）
（※ 2026-10-01 第十二/十三轮又有两大进展：**牙齿/露齿帧 = ref 方案 bug（11j），
双下巴 = PFLD 垂直语义（11l）**，均已修复并 9 条全量验证。本节的旧测量值
（MAE 0.0205 等）早于这两轮，仅作历史基线。）
嘴部框 MAE 0.0205、PFLD 唇高 corr 0.82@lag0。剩余差异来源（第八轮已定位）：
1. **他们的 GFPGAN 画布几何无法精确复刻**：其画布用**带噪声的检测器五点**对齐
   （逐帧倾角 −0.2°~+4.3°，我们的 PFLD/SCRFD/YuNet 都解释不了该噪声），
   画布模板≈arcface-512（eye_dist 140.2 vs 140.9）。我们采用 PFLD 几何 +
   `restore_crop`，crop corr 0.99 / MAE 0.0145（vs 原始 crop 0.0074，多出来的是
   锐度代价）。**A/B 实测**：raw crop 端到端 L1 更优（0.0188 vs 0.0205），
   但 restored 的锐度（嘴部 lapvar 80 vs 他们 65 vs raw 35）和牙齿纹理**目视更像原版**，
   且结构上复刻原版管线 ⇒ 默认保留 chaofen_before。想极致 L1 用 `--no-pre-restore`。
2. **"lag+20"已翻案**：PFLD 唇高序列 ours vs theirs 峰值 corr 0.82 **严格在 lag0**
   （±1 即掉到 0.80/0.76），第六轮报告是静态嘴视频 + dark-fraction 伪影的产物。
3. 最差帧（26/123/234，唇高差 ~0.1 眼距）都**不是**遮挡帧（当时无手挡脸），
   就是 crop 内容差 + 模型对输入的敏感性，无系统性 bug。

### P1 后处理链 ✅ 已解决（第八轮）
原版 run 的 ort 调用全量 dump 显示：**没有** face_parsing(79999_iter)、**没有** xseg、
**没有**独立 smooth 步骤；"nblend"就是合成掩码（见 §0）。已实现的 `paste_back_oval`
用 canonical 脸椭圆掩码（9 帧 dump consensus，th0.7+σ8），对他们的 vid 帧重构
MAE ~2.8/255（valid 区），误差集中在掩码边缘和他们噪声几何的高梯度区。

### P2 chaofen_before ✅ 已采用（推翻第六轮"不采用"结论）
第六轮判不采用是因为当时把它和 chaofen_after 叠加测。第八轮证明原版**只做
chaofen_before**（且 DINet 输入带确实锐，lapvar 88.9），单独开它 + 硬贴合成是正确结构。

### P3 解码器 BN running stats 与其权重不自洽（已定性，未解释，影响仍在但已被 chaofen 掩盖）
`dev/test_decoder_bn.py` (A)：给 `out_conv.0.2` 灌入**统计量完全符合 norm1 running stats**
的合成输入，conv1 输出方差仍只有 40.8，而 norm2 存的 running_var 是 1427.9（差 35×）。
全部 BN 的 `num_batches_tracked` 都是 **0**（训练走 vendored `sync_batchnorm`，不自增该计数器）。
后果：eval 下解码器残差支路被压扁（比值 0.037 / 0.004 / 0.001），原始渲染偏软
（嘴部 lapvar 7.6 vs 原片 48.5），靠 `chaofen_after` 补到 65。

---

## 5. 当前产物与指标

**端到端（第八轮，同素材 248 帧全片）**：

| 指标 | raw crop + oval | **restored crop + oval（默认）** | 原版 |
|---|---|---|---|
| 嘴部框 MAE vs 原版 | 0.0188 | **0.0205**（旧 fuse 方案 0.0468） | — |
| 嘴部框 lapvar | 34.9（偏软） | **80.3** | 65.4 |
| PFLD 唇高序列 corr@lag0 | 0.82（两者同） | | |

- 对比图：`examples\cmp_new.jpg`（5 帧 src|THEIRS|ours）、`examples\cmp_raw_vs_rest.jpg`
  （5 帧 src|THEIRS|raw|restored 目视 A/B）。
- `examples\out_current.mp4`：**250 帧全 10s**，chaofen_before 开 / chaofen_after 关 /
  oval 硬贴合成 / PFLD+TEMPLATE_256 对齐。嘴部开合全片正常、牙齿清晰。
  **第九轮补丁版**（增益匹配 + 边界衰减，修方框轮廓/脸变暗，MAE 0.0208 无回退，
  对比图 `examples\cmp_fixed.jpg`）。
- `examples\out_raw_oval.mp4`：同参数但 `--no-pre-restore`（L1 更低、更软的 A/B 版）。
- 历史版本：v3 lapvar 89.5/MAE 0.0550；第六轮"0.0468"版本实为静态嘴（无效）。

**第九轮：方框轮廓 + 脸变暗修复**（`paste_back_oval` 内，用户报告后定位）：
- **原因1 黑边方框**：canonical 掩码顶到 crop 第 0 行（mask[row0]=1.0）+ 渲染边界
  像素不可信（卷积边缘偏暗）→ 沿 crop 边界一条硬边。修法：掩码乘"到 crop 边距离"
  斜坡（4px 内收 0、32px 满）再 σ4 羽化，贴回 alpha 在 crop 内部就收 0。
- **原因2 脸变暗**：我们的渲染带全局暗移（脸比源帧深 ~3.4/255，BN stats 所致）。
  修法：每通道在未遮蔽带拟合 `render ~ gain*src + offset` 整体套用。
  修后与 THEIRS 逐通道 <1/255。**注意 THEIRS 本身也比源帧深 ~2.9/255**——
  那是 DINet 固有行为（原版亦然），不应再向源帧亮度校正，否则反而偏离原版。
- 服务器 align256.py 已同步。

**第九轮 b：眼角 GFPGAN 马赛克修复**（`pipeline.restore_crop`，用户报告眼角脏块）：
GFPGAN 对部分人脸会在**虹膜内产生马赛克闪点 + 内眼角黑色小块**（jie1 素材双眼每帧
都有；换人素材无）。眼睛在 input_mask 首零行（86）之上 = 直通内容，伪影直接进成片。
修法：以 five_kps 的瞳孔点（five[0]/five[1]）经 M 变换到 crop 空间，画两个
(38,28) 椭圆（σ4 羽化）作为保护区，区内保留 raw 内容、区外保留 GFPGAN 复原
（脸颊/下颌锐度不受影响，实测 cheek-band lapvar 20→200 保留）。
修后嘴部 MAE **0.0189**（比修复前 0.0208 还低——干净眼睛让渲染也更接近原版）、
眼角干净。证据图 `examples\eye_final.jpg`。服务器 pipeline.py 已同步。
⚠️ 保护区的代价：眼部比 THEIRS 略软（THEIRS 的眼是 GFPGAN 锐化且无伪影）。
若未来找到 GFPGAN-onnx 眼伪影的根治法（如换 gfpgan 权重/输入归一化），可缩小保护区。

**第八轮服务器 dump（/opt/projects/heygem12/dump，已在本地
`%TEMP%\opencode\pull\` 存副本）**：62 次 DINet 调用全量 arg0/arg1/arg2/out、
主进程 5 个 ort 会话（s1=GFPGAN 512×512 每帧 1 次、s2=SCRFD 640 每帧 2 次、
s3=属性 1 次、s4=PFLD 112 每帧 2 次、s5=3DMM 回归 224 每帧 1 次）、
249 张最终合成帧 `vid_NNNN.npy`（VideoWriter hook，写编码前）。

---

## 5b. 第十轮：ONNX 全面 GPU 化（2026-09-28）

**改动**（本地 + 服务器均已同步）：
- 新增 `heygem12/ortutil.py`：`make_session()` 优先 `CUDAExecutionProvider`、失败自动
  回退 CPU；Linux 下先用 ctypes(RTLD_GLOBAL) 预加载 site-packages/nvidia/*/lib 的 .so
  （torch 自带的 CUDA/cuDNN 库），**无需设 LD_LIBRARY_PATH**。
- `pipeline.py`(GFPGAN)、`face/detector.py`(SCRFD)、`face/landmarks.py`(PFLD) 三处
  `ort.InferenceSession(..., providers=["CPUExecutionProvider"])` 全部改走 `make_session()`。
  本地 CPU 环境自动回退，行为不变（冒烟 `--max-frames 8` 通过）。

**服务器环境关键坑**（driver 550.78，最高支持 CUDA 12.4）：
- `onnxruntime-gpu 1.27+`（含 1.30）**默认 CUDA 13.0 构建** → 与本机驱动不兼容；
- `1.21~1.26` 是 CUDA **12.8** 构建（minor-version 兼容要求 ≥12.8）→ 与 torch cu121
  自带的 12.1 库也不匹配；只有 **1.19.x/1.20.x 标称 CUDA 12.x + cuDNN 9.x，
  与 torch 2.5.1+cu121（12.1/cuDNN 9.1）匹配**。⇒ 环境锁 `onnxruntime-gpu==1.19.2`。
- 服务器 /usr/local/cuda-13 是**空壳**（lib64 无库），nvcc 是 11.5，都别指望。
- 服务器 pip 直连 PyPI 极慢（246MB 下不动），**用中科大镜像
  `-i https://pypi.mirrors.ustc.edu.cn/simple/` 秒装**。

**性能**（jie1 全片 1545 帧，服务器 4090 vs 本地 CPU）：
| 阶段 | CPU | GPU | 加速 |
|---|---|---|---|
| chaofen（GFPGAN 512²） | 803.1s（~520ms/帧） | 26.7s（~15ms/帧） | **30×** |
| dinet | 20.6s | 21.0s | 1×（本来就快） |
| fuse | 71.1s | 58.5s | 1.2× |
| **total** | **1142.3s（19min，580ms/帧）** | **162.9s（2.7min，70ms/帧）** | **7×** |

- 产物：`/opt/projects/heygem12/out4_gpu.mp4`（17M）。
- ⚠️ **又踩了 root 文件权限坑**（同 §1 的 1004_format.mp4）：输出名 `out4.mp4` 被
  root 拥有的旧文件占住，jinyun 写入 `Permission denied`、**静默失败**（合成全部白跑），
  换名 `out4_gpu.mp4` 重跑成功。服务器上用常见输出名前 `ls -l` 确认属主。
- 🐛 **cuDNN 子库不匹配**（当天复跑即爆）：PFLD 的 `FusedConv` 节点报
  `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`（1002）。根因两层：
  ① ORT 图优化把 Conv+激活融成 contrib 算子 FusedConv，走 **cudnn v9 graph API**，
  与 torch 2.5 自带的 cuDNN **9.1** 不兼容；普通 Conv（SCRFD/GFPGAN 用的路径）不受影响
  所以当时没暴露。
  ②（**同一报错在 root 交互 shell 复活的真凶**）`/root/.bashrc:116` 有
  `export LD_LIBRARY_PATH=/usr/local/cuda-12.5/lib64`——12.5 目录里有 **cuDNN 9.12**
  （libcudnn*.so.9.12.0），与 torch 的 9.1  soname 相同（都是 libcudnn*.so.9）被混加载。
  `sudo su` **不会**清掉它（实测 root 交互 shell 里 env 仍有该变量）。
  已实证：`export LD_LIBRARY_PATH=/usr/local/cuda-12.5/lib64` 后单测 PFLD 必复现
  `conv.cc:455 CUDNN failure 1002`，`env -u LD_LIBRARY_PATH` 即恢复。
  **修法**（已在 ortutil.make_session 落地）：CUDA 会话设
  `graph_optimization_level=ORT_ENABLE_BASIC` 跳过 FuseConvActivation 变换（治①）；
  跑 CLI 用 `env -u LD_LIBRARY_PATH python ...`（治②），或注释掉 /root/.bashrc:116。
  性能无感（PFLD 0.7ms/GFPGAN 14ms 每帧，全片仍 163s）。
  备选方案 `--no-deps` 升级 nvidia-cudnn-cu12（≥9.2，包 770MB）**弃用**：
  根分区 /tmp 已满装不下，且没必要。
- ⚠️ **服务器磁盘**：根分区（/、/tmp、/home）**99% 满（仅 1.5G）**，pip 装大包必须先
  `export TMPDIR=/opt/projects/tmp`；/opt 也只剩 ~30G。装 770MB 级 wheel 会直接
  `No space left on device`（本轮实测）。

---

## 5c. 第十一轮～第十三轮：闪烁闭环与逐像素对齐（2026-09-30 ~ 10-01）

（11a-11h = 第十一轮闪烁闭环；11i-11l = 10-01 第十二/十三轮与工具链，按时间序。）

**第十一轮：关键点时序稳定（2026-09-30，修"两帧变化过大/闪烁"）**

**用户报告**：woman_en/woman_cn（开头）、man1_en（~4s）等处两帧间脸部变化过大像闪烁，
猜测是面部框定位变化。1.png/2.png（woman 输出第 83/84 帧）嘴部开→合差异大。

**诊断结论**（dev/kps_dump.py，服务器实测）：
1. **确实存在单帧检测抖动**：YuNet/PFLD 五点在相邻帧间出现孤立跳变 7-11px
   （woman f16/232/250/253、man1 f34/58/104/116…），而相邻帧几乎不动
   （常态中位数 ~2.3px）——即用户猜的"框定位变化"，单帧坏点会让 crop+脸椭圆
   整体偏移一帧 → 闪烁。
2. **但用户看到的大帧间变化 mostly 是源视频自身的快速运动**：输出与源视频的
   脸部区域逐帧 diff 几乎相等（woman f249-257 src 8-10.7 = out 8-10.6，是
   快速挥手；1.png/2.png 相位相关位移 <1px，音频窗口 diff 无离群 → 该例是
   源帧真实快速嘴部动作 + 音频驱动，几何无跳变）。这类"闪烁"管线无法也不应消除。
3. 源视频无重复帧（30fps 容器即真实采集帧率）。

**修法**：`pipeline._keypoints_for_frames` 收集完原始五点后做时序稳定
（`_stabilize_keypoints`）：**median-3（去孤立单帧尖刺）+ EMA(α=0.65)**，
真实运动滞后 <1 帧。默认开，`cli --no-kps-stabilize` 可关。合成验证：
注入 16-22px 尖刺 → 压到 ≤3.7px，真实运动跟踪误差 0.81px；新输出 woman f231
处 out diff 1.56→0.41（src 0.66）。

**服务器重跑**：run_test.sh 加了 round-11 注释，sudo 重跑 6 条（assets/out 是
root 所有必须 sudo；run_test3.log）。⚠️ 这次 GFPGAN 成功加载
（chaofen_before=True，前两次 run 是 False），新输出比旧版更锐，与原版
管线结构一致。

### 11b. 脸型"胖瘦"闪烁修复：自适应渲染 EMA（2026-09-30 当天追加）

**用户再报告**：新输出开头脸型时胖时瘦（3.png/4.png = woman_cn 第 1/2 帧，
4.png 脸明显宽）。

**定位**：对齐几何无罪（相位相关 <1px、眼距归一化后两帧差 <0.5%）——是
**DINet 渲染内容本身逐帧抖动**：解码器每帧重生成整个 masked 区（脸颊/下颌
包含在内），BN running stats 不自洽（P3）使其对近相同输入也输出不同脸型
（本地连渲染 8 帧实测：静止帧渲染间嘴区 diff 0.006-0.06，说话帧 0.35；
脸型宽度逐帧变化 ~2%）。原版结构相同，亦有此特性。

** first try 失败**：全帧固定 EMA(0.5) 虽然压住胖瘦，但嘴部帧间形变差异大
时产生**双唇鬼影**（e2 帧可见），不可接受。

**最终方案（已落地）**：**自适应渲染 EMA**（pipeline run() 合成循环内）：
- 用嘴区权重（模板嘴角中心 rx45/ry35 σ8 羽化）计算当前 raw 渲染与上一 raw
  渲染的嘴区差异 d；
- d ≤ 0.08（近静止帧）→ 强平滑（α=渲染 EMA floor，默认 0.25，即 0.25×raw
  +0.75×历史），胖瘦抖动压到 ~25%；
- d ≥ 0.2（真实说话运动）→ α=1 直通，彻底避免鬼影；
- 之间线性过渡。阈值间隙实测极大（0.06 vs 0.35），稳健。
- CLI：`--render-ema`（默认 0.25；0 关闭）。静止段时序延迟 ~3 帧，但静止段
  本身无运动；说话帧零延迟。

**验证**：离线模拟 8 帧真实渲染：静止帧 diff 压至 0.001-0.015（raw 的 1/4），
说话帧 0.34 保持 0.34；输出条带目视脸型宽度全帧一致、牙齿清晰无鬼影。
服务器 sudo 重跑 6 条（run_test5.log），开头帧宽度 298-305 稳定。

### 11c. 反转：渲染 EMA 整体弃用（2026-09-30，用户对照原版后裁定）

**用户裁决**：原版输出（HG/ 跑的 `woman_en_orig.mp4`，28s，340 帧，见
assets/out）"没有问题"，而我们的仍有 1s/2.23s 明显抖动。

**关键实验与结论**（全部实测）：
1. **渲染 EMA 是净负收益 → 弃用（默认 render_ema=0）**。DINet 在音素边界会
   **整脸重绘**（静止帧渲染间 cheek diff 0.005-0.02，说话帧 0.2-0.5），任何
   时序混合都会在"混合态↔原始态"切换处产生**双唇鬼影/嘴部闪动**——正是用户
   在语音活跃段看到的"抖动"。原版从不做时序处理，我们的正确配置也不该做。
2. **几何早已与原版一致**：PFLD 逐帧测轮廓宽/眼距/眼-下巴距，ours vs orig
   差 1-3px；脸部宽度（肤色区 extents）±1-2px 恒定；嘴型时序 corr 0.79@lag0。
   之前"+15R 嘴部鲜红"的测量大部分是**采样框错位伪影**（两版 crop 里嘴部
   位置差 ~7px，固定框采样内容不同）；严格对齐后真实差距：全脸 +2R、嘴盒 +6R
   （用 hook 抓的原版 DINet 渲染 out_003 与我们渲染同 M 贴回对比：162.4 vs
   168.4 R）。余量小，不动颜色。
3. **hook 重跑确认原版管线**（hookdir/sitecustomize.py 重建，dump2/）：
   原版 arg0 嘴区全零（input_mask 遮嘴）、渲染硬贴、无后处理链变化。
4. **关键点稳定升级为运动自适应**（修固定 α 的 6-8px 运动滞后——那本身也是
   抖动源）：median3 杀单帧尖刺 + **2 帧尖刺对检测**（3 连跳且两侧平静才插值，
   真实运动不触发）+ **α=max(0.15, m/(m+1.5)) 自适应 EMA**（静止强平滑、
   快运动滞后 ≤1.5px）。woman 前 340 帧输出：平滑后帧间跳 p99 5.89（均为
   真实手势段），静止段中位数 1.17px。

**最终配置**（= 原版结构 + 干净几何）：chaofen_before 开 / chaofen_after 关 /
oval 硬贴 / kps 稳定(自适应) / **render_ema=0**。服务器 run_test6.log 重跑 6 条。
本地对照：Temp\opencode\ woman_en_orig.mp4（原版） vs v2_woman_en.mp4。

### 11d. 开头"脸变大"修复：尺度锚定（2026-09-30 收尾）

**用户最后报告**：v2 版 0.04s（第 1 帧）和 2.24s 仍有明显脸部变化。
- **0.04s 根因（真 bug）**：检测器起始沉降——PFLD 眼距在视频头 8 帧从 150.6
  漂到 147.3（源视频几乎静止，非真实运动），EMA 只能延迟斜坡 → 贴回的脸
  在开头几帧逐渐"变大"。修：`_stabilize_keypoints` 末尾加**尺度锚定**——
  每帧五点绕眼心缩放，使眼距 = ±20 帧滑动中位数（静态机位真人脸尺度恒定，
  YuNet bbox 全片散布 ~±1%）。修后开头眼距 149.1→148.5 平滑。平移/旋转不动。
- **2.24s 确认非 bug**：跑原版 HG + woman_cn.wav 对照（woman_cn_orig.mp4），
  第 67→68 帧原版与我们嘴型变化完全一致（音频里 2.24s 本就是快速闭音素），
  DINet+该音频的固有行为。
- 服务器 run_test7.log 重跑 6 条；本地 v3_woman_cn.mp4 验证开头宽度一致。

### 11e. 几何滞后 stutter 修复：硬切换 EMA（2026-09-30 最终）

**用户报告**：0.04s 修好（尺度锚定生效），但 2.24s（帧 67）下半脸仍有"连起来的
突变"，单帧都正常。

**根因**：运动自适应的 α=m/(m+m0) 斜坡在 1-4px 运动带内产生**运动强度依赖的
滞后**——该素材人物常态小运动 ~2.4px/帧，大量帧落在部分混合区，有效速度不
均匀 = 运动起止处的下颌/嘴角 stutter。数据验证：此片段眼周/下颌变化与原版
逐帧同步（非渲染伪影），唯一差异源就是几何滤波的变速滞后。

**修法**：α 改**硬切换**（m ≥ 2px → 完全直通 raw，零滞后；m < 2px → α=0.12
强平滑）。直通期间 state 无偏移 → 停止时零回拉；静止段 state≈raw±1px →
启动时台阶 ≤1px。实测（woman 前 340 帧）：平滑后帧间跳中位数 0.40px
（原 1.2-2.4），p99 7.75（均为真实运动原速跟踪），58-70 帧下颌带逐帧差
≤1.7 无尖刺。服务器 run_test8.log 重跑。

**教训汇总（闪烁问题完整闭环）**：①单帧检测尖刺→median3；②双帧尖刺对→
两侧平静才插值；③检测起始沉降→尺度锚定(±20 滑窗中位数)；④运动滞后→
硬切换 EMA；⑤渲染时序处理（任何形式的混合）→**全部弃用**（音素边界整脸
重绘，混合必鬼影）。终态 = 原版结构 + 干净几何。

### 11f. ★最终决定性问题：音频窗口 floor vs round（2026-09-30，闪烁真凶）

**用户最终报告**（分屏截图 6.png/7.png）：1.02s/5.10s 处原版嘴大张、我们嘴
闭合——同一时刻嘴型相位差数帧；且"单张都好、连起来突变"遍布全片。

**定位过程**：hook 全量 dump 原版 DINet 音频输入（dump3，40 call × 4 样本），
与我们的 windows 逐样本对齐（安静处相邻窗 MAE 0.0006、错位 1 特征则 0.04，
判别比 70×）。结论：**原版窗口起点 = floor(i×25/30)，我们 = round(i×25/30)**。
30fps 视频 stride 为分数，每 6 帧出现一次 .5 分数帧（i≡3 mod 6）——round
在此处进位而 floor 不进 → **我们的音频窗每 6 帧有一次 +1 特征错位、随后追平，
嘴部运动叠加 5Hz 定时抖动**：连续运动的嘴每 0.2s 多走/少走一步 = 全片遍布的
"突变"，快速音素处最显眼。25fps 素材 stride=1 不受影响（旧验证没暴露）。
（另有 8/160 样本（i≡1 mod 6, i≥115）差 2 特征，疑似原版队列块边界 quirk，
量级 ±40ms，不值得复刻。）

**修法**（frontend.frame_windows）：`lo = int(round(...))` → `int(np.floor(...))`。
验证：152/160 样本与原版 dump 逐位吻合（MAE<0.003），剩余 8 个 off-by-1。
25fps 行为不变（floor==round）。本地测试通过；服务器 run_test9.log 重跑 6 条。
**教训：与原版做逐帧数值对齐时，分数 stride 的取整方式必须按 dump 实证，
不能假设 round。**

**附带发现**：那 8 个对不上的样本（i≥115 起偏移增大）是**原版管线自己在
~f115 处解码丢了一帧**（其后它的帧号相对源偏移 +1，窗口随之错位）——我们
不丢帧，后半段与它的 ±1 帧差异源于此。另外 v5 验收时用户报告的 5.08s
"嘴鼻下移/拉长" 经查为源视频 f146-150 的真实低头+眨眼（源/原版/我们三方
逐帧一致，含闭眼复现），属素材固有运动。至此闪烁类问题全部闭环。

### 11g. 嘴型幅度与形状：五点语义标定（2026-09-30，部分解决+明确剩余路径）

**用户报告**（v5 分屏截图）：1.02/2.24/5.10s 原版嘴大张露齿、我们嘴近平闭。

**决定性实验**（dump3 输入置换，全部本地 CPU 可复现）：
- 我们的模型 + 他们的 arg0/arg1/arg2 → 渲染与他们逐位等价（模型无问题）；
- 我们的 src crop 是阻尼主因：换他们的 src→幅度全恢复；换他们的 ref→无变化；
- 几何测量：他们的 DINet crop 脸比我们**大 ~13%**（crop 内眼距 86.8 vs 76.7，
  光流显示水平径向 15% 扩张）。原因：**PFLD 五点与检测器五点语义随人漂移**
  （SCRFD 眼距 137 vs PFLD 147；jie1 上 SIFT 拟合吸收了 jie1 的比率 0.90，
  换人就失效——round 2.5 模板的人依赖缺陷）。
- **已落地**：`_keypoints_for_frames` 做语义眼距标定——逐帧算
  YuNet眼距/PFLD眼距 比，取**视频中位数 R**（woman=0.93-0.96），把 PFLD 眼点对
  绕中点缩放 R（倾角保留），再进稳定链。全片唇高 std 1.51→1.86。
- **剩余差距**：嘴**张开形状**（原版上唇抬起露齿的高开口 vs 我们的横向扁平
  开口）。暗区分数已对齐（0.23-0.25 vs 0.21-0.24）但 PFLD 内唇高仍只有原版
  ~60%。所有快速变体（YuNet 原始五点/GFPGAN 归一化/旋转补偿/保护开关）均
  不能复刻该形状。
- **下一步（已定位，后由 11i 修订）**：修 SCRFD 解码（我们的 SCRFD 类在 1080x1920 上输出
  乱码 kps，原版 HG/face_detect_utils 有其正确解码可对照），用 SCRFD 五点
  （与原版同检测器=跨人稳定语义）+ 对 dump3 arg0 带做模板拟合（SIFT/坐标
  下降，目标 bandMAE ~0.02），替换 PFLD 管线。验证锚点：their-src+our-ref
  实验已证明 crop 对了形状即对。
  **※ 11i 已实测否决该路线**（SCRFD 精度差、模板边缘拟合更差）；SCRFD 检测器
  本身已修好（锚点交织 bug），最终只用作 11l 的语义标定基准。

### 11h. demo1（大衣女士，大头部动作）验收（2026-09-30）

**用户报告**：7.17s 等处嘴部有伪影（截图 ORIG/OURS 分屏）。实测确认：
- 几何跟踪完美（眼位误差 0.78px vs 原版 0.72px，大动作段一致；仅 f127-130
  有 7px 偏差待查）；唇高均值/std 与原版几乎相同（23.4/2.7 vs 23.5/2.8）；
  颜色/对比度/饱和度/时序动态全部对齐。
- **差异仍集中在嘴部**：f179 等处我们的嘴过张+红唇（嘴区 MAE ~3/255），
  与 woman 视频同类——五点语义问题。逐帧 A/B（27 帧）：YuNet 原始五点 2.93
  vs PFLD+R 3.05 vs 原版——YuNet 略优但不决定性。
- **SCRFD 修复发现**：本仓库 SCRFD 类两个缺陷——①整帧 letterbox 通路在
  640 画布上漏检（脸 ~83px < 阈值）；②金字塔通路把非方形中心裁剪**压扁**
  到 640×640（demo1 的 540×960→640×640，纵横比 1:1.78 扭曲），跨尺度
  映射不一致（同一张脸在 frac0.25/0.35/0.5 映射到三个不同位置）。修法：
  金字塔改 letterbox（保纵横比+pad）+ score_thresh 降到 ~0.3 靠几何校验
  门控。修好后用 SCRFD 五点 + dump3 模板标定替换 PFLD 管线（=11g 下一步）。
- ⚠️ demo1 是 25fps：floor==round，音频无 5Hz 问题；眨眼帧（f6-8）仍有
  睁眼表情翻转（闭眼时 PFLD 瞳孔点不可靠 → 同 11g 语义问题家族）。

### 11i. SCRFD 路线实测定论（2026-10-01）：不做主检测器，检测器已修好

**动机**：demo1/2/3 角度变化时的漂移 + 嘴型形状差 → 怀疑需要原版同款检测器。

**修复（已落地，detector.py，有独立价值）**：
1. ★**锚点交织 bug**：模型输出是逐 cell 交织（cell0_a0, cell0_a1, ...），
   我们原按 anchor-major 拼接——这就是 HANDOFF 旧结论"SCRFD 误检严重别用"
   的真正原因。修复后 SCRFD 在所有测试视频上检出正常（分数 0.75-0.89）。
2. 金字塔通路改 letterbox（保纵横比），新增 `detect_refined`（人脸裁剪
   上采样二次检测，复刻原版"每帧 2 次 SCRFD"）。

**实测否决集成**：
- SCRFD 原始五点 jump：demo1 median 5.0px/p99 20px、demo2 p99 34px——
  比我们的 YuNet+PFLD（median ~2.3px）**差得多**；detect_refined 后
  仍 3.9px median。SCRFD-500m 在这些分辨率下精度不够，不是根治。
- 模板标定（边缘图坐标下降拟合 10 DOF 到 dump3 arg0 带）结果**更差**
  （mouth MAE 3.19 vs 未标定 2.98）——restored/raw 边缘分布差异污染目标。
- 嘴区 MAE 对比（demo1 28 帧）：PFLD+R 3.05 / YuNet 2.93 / SCRFD 2.98
  / SCRFD+标定 3.19——**无决定性赢家**，管线维持 YuNet+PFLD+R。

**漂移结论**：demo1/2/3 各有 3-5 个 fast-rotation 帧（demo1 f127-130、
demo2 f14/17/130-134、demo3 f309）出现 4-8px 偏移——**原版同样漂移**
（demo1 该处原版 4.3px/我们 7px），是运动模糊下所有检测器的固有噪声，
不是滤波失败；硬切换 EMA 已把静止段压到 0.4px。残余改善空间很小。

**嘴型残余**：~~根治需逐像素复刻原版 crop 配方~~ —— 后续 11j（ref 方案，
修露齿帧红唇团）与 11l（垂直语义标定，修双下巴）已解决其中两大块；
剩余"个别帧张嘴幅度略大"见 11l 遗留。

**SCRFD 的最终用途**：不做主检测器（精度不够，见上），只做**每视频语义
标定基准**——11l 的 `_vcalib_keypoints` 用其五点中位数比率校正 PFLD 的
鼻/嘴垂直位置。11g 的"用 SCRFD 五点替换 PFLD 管线"计划据此**作废**。


### 11j. ★★ DINet ref 输入方案纠错（2026-10-01，第十二轮）

**症状**：demo1 露齿帧（f54/103/179 等）我们输出嘴部是一团纯红唇（无牙齿、
无口腔细节），脸上像"多一块东西"；原版同帧有清晰牙齿。

**根因（hook dump 实证，demo1 全片原版运行 arg1 批次抓取）**：
- DINetV1.forward 是 4 帧一批：arg0=同帧 crop×input_mask（mask 区全黑），
  **arg1=同帧 crop 不 mask**（arg1[j] 与 arg0[j] 在 mask 区外逐位 0 差，
  mask 区内是真实内容）——参考图就是**当前帧自己未遮罩的 crop**，不是
  固定参考帧。
- 我们之前 ref = 视频中帧（f100）复原 crop 重复喂所有帧 → 网络只能从
  参考帧已有的嘴型"借"纹理；参考帧张嘴小 → 露齿帧生成红唇团。

**修复**（pipeline.py，2 行）：循环内 `ref_t = self._to_tensor(rgb)`（同一
复原 crop，未 mask）；删除 middle/first 参考帧逻辑（ref_frame 参数保留但
弃用）。零额外算力（复用已复原 crop）。

**本地验证**（demo1 f54/103/179，与 dump 同一次原版运行配对）：三帧牙齿
全部出现、唇色自然，红唇伪影消失；嘴部 MAE 在此类帧不适用（牙齿边缘
1px 配准差即抬 MAE），以视觉为准。

**注意**：mask 区外 arg0==arg1 位精确 ⇒ "我们的模型+他们的输入=逐位等价"
结论不变。woman 视频在 11l 批次已重跑验证：张口帧牙齿+舌头恢复，
"嘴型过平"问题随之大幅改善。

### 11k. 原版服务运行陷阱 + 对比工具链（2026-10-01）

**HG 服务两个坑（实测）**：
1. **work_id 硬编码 "1004"（HG/run.py）且整任务缓存**：`task.work()` 发现
   上次残留（`1004_format.*`/`1004-r.mp4`）直接复用不重新处理——批量跑多
   个视频时全拿到第一个的输出。每次运行前必须清 `HG/1004-*.{mp4,wav}`、
   `HG/result/1004*`（run_compare.py 已内置）。
2. **ffmpeg PATH 坑**：其预处理 shell 调 `ffmpeg`，`/usr/local/bin/ffmpeg`
   （自制 5.1.2，无 libx264）报 "Option not found"；`/usr/bin/ffmpeg`
   （Ubuntu 4.4.2 有 libx264）正常。运行原版前把 `/usr/bin` 放 PATH 最前。
   9/30 首跑能成功纯粹因为缓存命中跳过了 format 步骤。

**对比工具链**（`scripts/`，服务器 /opt/projects/heygem12/scripts/）：
- `run_compare.py`：输入若干 `视频 音频` 对 → 逐对跑原版+本项目 → 合成
  左右对比（ORIG|OURS 红标签，音频取 ours）。选项：`--skip-orig`、
  `--skip-ours`、`--show-mask`、`--layout v`。含首帧查重守卫防缓存复发。
- `compare.py`：`make_compare(orig, ours, out, layout, show_mask,
  checkpoints)`；自动模式扫 `assets/out/*_orig.mp4` 配对生成 `cmp_*`。
  `--show-mask` 用项目自身检测器（YuNet→PFLD→R→稳定链）逐帧算 oval
  贴回 alpha（`Align256.paste_mask_alpha`，新公开方法），画绿色轮廓于
  两侧，标示被替换的面部区域。
- 服务器 ffmpeg 无 libx264 → 自动回退 `h264_nvenc`（GPU）。

### 11l. ★ 双下巴伪影修复：PFLD<->SCRFD 垂直语义标定（2026-10-01）

**症状**：demo1 等多帧 OURS 下颌/下巴比原版胖一圈，颏下多一道折叠
（双下巴）；ours-vs-raw 差异图显示整个下颌环 + 颏下条带，而原版
差异集中在嘴部。

**排查链**：渲染与 src crop 垂直零偏移（crop 内 band50-95 相关 dy=0）；
贴回自洽 => 是纯内容问题：我们喂给 DINet 的 crop 下庭比原版高 ~18 像素
（同帧 crop 下巴行：ours 212 vs 原版 194），网络按输入几何渲染出胖下颌。

**根因**：PFLD 五点中鼻尖/嘴角相对眼线的位置比 SCRFD（原版对齐所用）
高 ~20%（demo1 实测 mouthY/eye 比：PFLD 0.80 vs SCRFD 1.0）。相似度
拟合被鼻部/嘴部点上拉 => 下巴在 crop 中位置偏低。水平方向同类问题已
由 R（眼距语义标定）+ 拟合模板 TEMPLATE_256 吸收；垂直方向一直缺。

**修复**（pipeline _vcalib_keypoints，开关 kps_vcalib=True）：每视频
采样 <=50 帧跑 SCRFD（复用已修复的检测器），取
median((mouth_y-eye_y)/eye_dist)_SCRFD / _ours 作为 k，把五点中 2..4 点
（鼻/嘴角）绕眼线垂直缩放 k；帧间共享数组用 seen 守卫防重复缩放。
各视频实测 k：demo1 1.185、demo2 1.082、demo3 1.147、woman 1.057、
man1 1.145、man2 1.119——自适配，woman 因模板已吸收故接近 1。
修后 demo1 crop 下巴行 212->198（原版 194），双下巴消失；woman 无回退。

**遗留**：f129-133/151-154 等帧颏下仍有 ~1-2px 级残余（内容生成差异，
非几何）；张嘴幅度个别帧仍略大（11j 前已存在）。

### 11m. 颈部黑色接缝修复：mask 下缘渐隐 + vcalib 小修正阈值（2026-10-01）

**用户报告**：man1_cn / woman_cn 脖子处有细微黑色接缝，其他视频没有。

**排查与结论**（⚠️ 教训：跨输出比较渲染内容必须**同音频**，本例一度拿
woman_EN 渲染对比 woman_CN 渲染得出错误结论）：
- 接缝 = DINet 重绘的颏下阴影比原帧深 + oval mask 边界恰落在露出的颈部
  皮肤上。实测**原版同样存在且更深**（同音频 f52 颏下亮度谷：ours 94 /
  原版 89 / 原帧 114）——两管线共性，只是 demo1/2/3、man2 颈部被衣领
  遮住不可见，woman/man1 露颈才暴露。
- oval mask 下缘在 crop 第 230 行，下巴仅到 ~195 —— 中间 35 行纯颈部
  被满强度粘贴。

**修复**（两处）：
1. _oval_mask 新增**下缘渐隐**：crop 行 208→246 垂直渐隐到 0（下巴 195
   留 13 行全强度余量；颈部无语音信息，混回原帧零代价）。对所有视频生效。
2. _vcalib_keypoints 加**小修正阈值** |k-1|<0.08 跳过：SIFT 拟合模板
   （以 woman 拟合、变换自带 ~21px 垂直平移）已吸收小量逐人偏移，重复
   校正=双重校正。实测 woman k=1.057 时 vcalib 对颈部影响仅 ~1.6/255
   （可忽略），跳过保其模板匹配几何；demo1 k=1.185（18px 下巴行误差）
   等大偏差仍生效。

**验证**：本地渲染同音频 A/B（kSkip|k1.055|orig|raw 四联图）颈部与原版
相当；服务器 9 条全量重跑（run_test14 批次），calibration 日志确认
woman 两条跳过、其余六条生效；test_alignment 通过。demo1 下巴无回退
（taper 起点 208 在其下巴行 198 之下 10 行）。

**遗留**：颏下暗带本身（DINet 重绘固有，原版更深）仍在；taper 只消除
边界线。若用户仍敏感，下一步可考虑颏下区域亮度保护（render 明度向 raw
回拉），但会偏离原版行为。

---

### 11n. 颈部闪烁伪影修复：restore_crop 颈部保护区（2026-10-01）

**用户报告**：man_6/12/15 输出脖子处有类似闪烁的伪影（黑斑爬动）。

**排查链**（统计指标全部正常，最终靠逐帧目视定位）：
- 颈部亮度带均值/逐像素时序方差与源视频持平（OUT 9.15 vs SRC 9.84）——
  不是亮度闪烁；8 连帧颈部条带目视发现**低频频谱斑在帧间形变爬动**。
- 查 DINet 输入：RAW crop 颈部干净；**restore_crop（GFPGAN 复原）的颈部
  有横向虚线接缝 + 斑块**（crop 行 ~190-230）。颈部远离人脸先验，GFPGAN
  在 512 画布上对其输出不稳定的"缝合"重绘。
- 该复原 crop 是 DINet 的 **arg1 参考图（未遮罩）**（11j）→ AdaAT 把参考
  图的颈部纹理复制进渲染 → 输出颈部爬斑。**原版同帧颈部干净**（其画布
  对齐不同，不产生此伪影）。

**修复**（`restore_crop`，与眼部保护同机制）：在眼部椭圆之外再画一个
颈部保护椭圆 (128,226)/(85,40) σ4 羽化，区内保留 raw 内容、区外保留
GFPGAN 复原。颈部无超分需求，且颈区在 input_mask 遮罩内（arg0 本来全黑），
保护区只影响参考图。修复后渲染颈部与原版相当（NEW|OLD|ORIG 三联验证：
OLD 有爬斑，NEW≈ORIG 平滑）。

**教训**：① 时序伪影先用逐帧条带目视，统计指标（带均值/逐像素方差）对
低频形变不敏感；② GFPGAN 的非人脸区域（颈/发际/耳）输出可能不稳定，
参考图途径会把任何复原伪影带进渲染，新增保护区时优先考虑"该区域是否
需要超分"而非"复原是否更好"。

### 11o. ★ 60fps 音频窗口重采样映射（2026-10-02，"还有点抖动"的定案）

**用户报告**：cmp_man_6 等与原版比仍有轻微抖动（几何与内容稳定性实测都与
原版持平甚至更好，差异只能在时序映射）。

**排查**（hook dump 原版 DINet arg2，60fps 素材 man_6/man_15 ×
man_cn/man_en，与本地特征库逐窗反解起点）：
- 60fps 下原版**不是** floor(i×25/60)：整数点（i≡0 mod 12）处起点少 1，
  且 i≥101 后整体永久 -1 滞后——恒速率假设全灭。
- 两个不同长度音频的 dump 均以 **0/160 失配**吻合：
  `start(i) = floor(i × (n_feat−T) / (n_out−1))`，
  `n_out = ceil((n_feat−T+1) × fps / 25)`（284 特征@60fps → 636 帧；
  332 特征 → 752 帧）。即可用特征跨度被**线性重采样**到原版输出长度。
- 视频受限情形（30fps woman：340 视频帧 < 840 音频上限）维持 11f 实证
  的 floor(i×25/fps) 不变；25fps（demo1）行为不变。

**修复**（`frontend.windows_for_video`）：视频帧数 > n_out_audio 时走
重采样分支（n=n_out_audio, stride=span/(n−1)），否则维持原 per-frame
速率；`pipeline` 循环帧数改用 windows.shape[0]（输出长度与原版一致，
cmp 完全对齐）。注意 ours 输出长度从此与原版相同（音频受限时短于视频）。

**验证**：两个 dump 序列 0/160 失配（本地脚本 verify_formula）；
test_audio_frontend 通过；服务器 run_test2 批次重跑 + test_cmp 重出。

**教训**：60fps（及任何"视频长于音频"）情形是独立代码路径，别假设
25/30fps 的 floor(i×25/fps) 泛化；分数帧率映射必须按 fps 分别 dump 实证。

### 11p. 泛化批量测试基础设施 + 原版"人脸过多"限制（2026-10-02）

**目录约定**（服务器 /opt/projects/heygem12/assets/）：
- `test_media/man|woman/`：泛化素材（18 条，15s 截断，命名 `man_N.mp4` /
  `woman_N.mp4`，60fps）。⚠️ 曾用 `1 (N).mp4` 带括号命名，已批量改名。
- `test_out/`：我们的输出（run_test2.sh 生成，平铺同名）。
- `test_out_orig/`：原版输出（HG 直跑；man_37 为角部涂黑版，见下）。
- `test_cmp/cmp_*.mp4`：左右对比（含 demo1/2/3：cmp_demo1/2/3.mp4）。

**脚本**：
- `run_test2.sh`：遍历 test_media，man→man_cn.wav、woman→woman_cn.wav →
  test_out/。
- 原版批量：无固化脚本，参照 `/tmp/orig_batch.sh`（11k 的两坑清理 + PATH
  /usr/bin 置顶 + 逐条 cp HG/1004-r.mp4）。单条用 `scripts/run_compare.py`。
- 对比批量：`bash /tmp/mkcmp.sh`（compare.py 显式三元组循环）。

**原版"人脸过多"硬失败（重要）**：man_37 前 2 秒左上角背景有 5 帧假第二张
人脸（置信度 ~0.5），原版 `.so` 直接抛"人脸过多"拒绝处理整段（无任何选脸
逻辑）。我们的管线自动选最大人脸不受影响。变通：把前 2 秒左上角 280×64
涂黑后跑原版（/tmp/m37clean.sh）。**实际使用含义：素材可能有路人/海报时
原版不可用，我们可用——算我们的稳健性优势。**

**60fps 素材注意**：泛化批全是 60fps，走了 11o 的音频受限重采样路径；
输出长度=原版长度（636/752 等），与原版逐帧对齐。

## 6. 已排除的候选（别再重走）

- **BN train 模式**：`train_decoder` selfMAE 0.4007 / lapvar 18223；`train_all` 0.3257 / 14597（纯噪声）。
  且 BatchNorm1d 在 T 维归一化会把音频信息整个抹掉。`model.eval()` 是对的。
- **BN 重标定**（用真实数据重估 running stats）：batch=1 与 **batch=16×2passes 都发散**
  （selfMAE 0.041→0.29~0.32、lapvar 冲到 18000+、解码器 act/run 比值飙到 47×~9,554,886×）。
  stored stats 虽不自洽，却是唯一能维持稳定的配置。
- **GFPGAN 作用在整帧**：嘴部 lapvar 192.6（原片 48.5，过锐 4×）；改到对齐 crop 后 51.6。
  另有一个**黑边 bug**：256→512 的相似变换带 +66.8px 平移（我们 eye_y=70，arcface-512 是
  206.8，而 2×70=140），crop 的 rows>222 / cols<16 越界变黑，正好压在 `fuse_mask` 下缘/左缘。
  现改为**从原帧**做 arcface-512 对齐（永不越界）→ GFPGAN → 用 `A = M256·M512⁻¹` 变回 crop 空间，
  并用 warp 全 1 图得到的 valid 掩码兜底。sliver 平均亮度 87.8→93.4（原片 95.9）。
- **`removegreen` 按字面实现**：样例视频 GT crop 的 green-dominant 像素占 **33.94%**（渲染 33.78%）
  ⇒ 绿色是合法场景内容（绿背景/服装），不是伪影；强行压制使自重构 MAE 0.0406→**0.0714**。已移除。
- **全局色彩校正**：未遮蔽直通带（占 crop 49.7%）本身零色偏（R+0.001 G−0.003 B+0.001）、
  MAE 仅 0.0177，拟合出 gain≈1.000/offset≈0.000，套用到嘴部后 MAE **0.0408→0.0408（+0.0%）**。
- **用"与原图 MAE"标定融合强度**：退化目标（强度越小越像原图），1.0→0.70 时 MAE 单调下降。
- **收紧融合范围**已采用：`align256.MOUTH_ELLIPSE=(75,60,41)`，把替换面积从 38.8%（官方
  fuse_mask 覆盖整个下半脸）降到 **21.5%**，静音 MAE 0.0365→0.0244、corr 0.960→0.974。
  传 `mouth_ellipse=None` 可还原官方行为。
- **融合边界纹理断层（贴片感）**：融合区内/外局部高频比 0.952 vs 原片同位置 0.972，
  边界 |原图−输出| 仅 1.9/255。无缝。
- **特征采样率上采样到 80fps**、**窗口 T=5/9/13/25**、**特征缩放/居中/z-norm**：均无效。
- **conformer 中间层特征**（block0~11）：渲染垃圾，BN 比差 100~70000 倍。
- **kaldi fbank**（含 ×2^15、dither、CMVN 全组合）：CTC 全死（0 token）。
- **320×416 嘴部裁剪 + 硬零矩形**（开源 DINet 约定）：对 dinet_v1 是错的，
  输入就是 256×256 乘 `input_mask`。`face/crop.py` 已删除。
- **AdaAT 权重学废了音频半边**：已排除。`adaAT.*` 对 trans_para 后半（音频）的行范数
  与前半同级：commn_linear 0.891×、scale 0.954×、rotation 1.095×、translation **1.124×**。
- **chaofen_before（源图 GFPGAN 锐化）**：已标定但**判定不采用**。按 lapvar 把源图锐度
  调到与原版一致（band 159 vs 他们 145）后，渲染反而离原版更远
  （raw 0.0140 < blend 0.0152 < full 0.0235）。清晰度不是渲染差异的原因。
- **按帧内容匹配定帧索引**：讲话人几乎静止，相邻帧 crop 内容差 <0.005，
  匹配给出的 offset 中位数 2、散布 −8~+17，**不可信**。定索引只能靠音频窗口
  （我们的窗口与他们同帧索引逐位一致，MAE 5e-5）这类有区分度的信号。

---

## 7. ⚠️ 已被推翻的旧结论（前几轮文档写过，别再信）

1. ~~"DINetV1 raw 渲染天生低质，GFPGAN 一上就清晰，问题已闭环"~~ —— 错。糊的主因是
   **网络结构错 5 处 + wenet encoder 配置错 + 音频前端配方错**。
2. ~~"crop 内嘴宽 233、嘴心映射到 (160,190)、320×416"~~ —— 全错（开源 DINet 约定，不适用于 dinet_v1）。
3. ~~"鼻锚定 compute_crop_radius 是正确几何"~~（v4）—— 也不对，被 256×256 + `input_mask` 取代。
4. ~~"音频窗口 T=5"~~ —— 错，T=20；~~"offset=+17"~~ —— 错，offset=0（是拿错音频搜出来的）。
5. ~~"checkpoint 有 concat_linear ⇒ concat_after=True"~~ —— 错，原版是 False（键存在但不用）。
6. ~~"口型已复活"~~（第五轮中途）与~~"口型彻底死了"~~（第五轮结论）—— 两次都不严谨：
   前者用了退化的短音频切片 + 边缘复制窗口；后者用了无效指标 `corr(openness, rms)`。
   正确判据见 §3。
7. ~~"wenet 前端是 kaldi fbank / hop=200"~~ —— 错，见 §2.3。
8. ~~"用 YuNet 五点做对齐"~~ —— 错。它的眼连线倾角逐帧乱跳（−0.4°~−4.1°），
   既 misalign 又导致 crop 时序抖动。对齐必须用 PFLD-98 五点（§2.5）。
9. ~~"chaofen 是画质差距的主因，把源图锐度调上去就行"~~ —— 错。
   锐度调到与原版一致后渲染反而更远（§6）。当前最大差距在**后处理链**（§4 P1）。
10. ~~"§5 端到端指标正常（openness std 达原版 91%），口型没问题"~~ —— 错。
    第七轮实测 `_restore_render` 把 GFPGAN(**源帧**) 贴回渲染图，输出≈锐化原视频，
    嘴完全不动；GFPGAN 逐帧抖动虚增 dark-fraction std 骗过了指标。
    **教训：dark-fraction 类指标会被后处理伪影欺骗，判口型必须看逐帧嘴部 strip 目视。**
11. ~~"恢复步骤对最终画面无影响，随便贴"~~ —— 错。任何在渲染图之后、
    融合之前的步骤如果引入源帧内容，都会静默抹掉口型。改后处理链时必须先 `--no-restore` 对拍。
12. ~~"chaofen_after 应该存在（GFPGAN 作用在渲染上）"~~ —— 错（第八轮 dump 实锤）。
    原版**没有任何 chaofen_after**；GFPGAN 只作用于**源帧**（chaofen_before），
    最终合成帧里 GFPGAN 内容为零（脸椭圆外与源帧逐位相同 d=0.00）。
    `_restore_render` 已修复为可选项但默认关。
13. ~~"fuse_mask ∩ 嘴部椭圆 + wh=0.97 羽化融合是正确合成"~~ —— 已被取代。
    原版是**脸椭圆硬贴**（全脸 ~45% crop 区域，α=1，边缘微羽化）；
    `paste_back`（fuse_mask 方案）保留作 A/B（`--fuse-mode fuse_mask`）。
14. ~~"openness corr lag+20 说明帧索引差 20"~~ —— 伪影。PFLD 唇高序列在 lag0 即峰值
    （0.82，±1 掉到 0.80/0.76）。第六轮的视频是静态嘴 + dark-fraction 指标被
    GFPGAN 抖动欺骗的产物。

## 7b. 第八轮新排除的候选

- **GFPGAN 画布几何的各种假设**：arcface-512+PFLD（corr 0.62）、crop×2（0.37）、
  SCRFD 五点（0.12）——全对不上。真相是**带噪声的检测器五点**（逐帧倾角 −0.2°~+4.3°，
  无一现有检测器能复现），精确复刻不可行，用 PFLD 几何近似（crop corr 0.99）。
- **raw vs restored crop 的默认选择**：raw 端到端 L1 更优（0.0188 vs 0.0205）但嘴部
  lapvar 35（原版 65）偏软；restored 的 L1 差 8% 但锐度和牙齿纹理目视更像原版。
  默认 restored（结构上复刻原版）；`--no-pre-restore` 可切 raw。
- **PFLD 唇高 corr 0.82 的残差不需要 SCRFD**：第八轮证实原版 DINet crop 也用
  PFLD（SIFT 恒等），其噪声来自 GFPGAN 画布而非 DINet crop，换 SCRFD 无意义。

---

## 8. 使用与环境

**服务器运行（第八轮已搭好，`/opt/projects/heygem12`，conda 环境 heygem312）**：
```bash
ssh jinyun@10.0.70.127
cd /opt/projects/heygem12
/opt/anaconda3/envs/heygem312/bin/python -X utf8 -u -m heygem12.cli \
  --video HG/example/ours.mp4 --audio HG/example/ours.wav \
  --output out.mp4 --device cuda
# 250 帧 1080p 全片 GPU 约 3.5 分钟；1545 帧全长素材 163s（chaofen 全 GPU，§5b）
# 运行中打印各阶段耗时：detection / 音频特征 / 合成（align|chaofen|dinet|fuse 分项）/ 写视频
# 注意 GPU 利用率：DINet/fuse 小的主因是逐帧串行 + fuse 的 CPU 图像处理（§5b）
```
⚠️ 服务器 /tmp 在根分区（仅 ~2G 空余），pip 装大包必须先 `export TMPDIR=/opt/projects/tmp`
（setup_server2.sh 已这么做）；/opt 自身也只剩 ~25G。
⚠️ 服务器 GPU 版与本地 CPU 版有数值差（torch 2.5.1+cu121/TF32 vs 2.14+CPU）：
嘴部 MAE vs 原版 GPU 0.0253 / CPU 0.0205，目视等价。

**本地运行**：
```powershell
cd D:\codes\pythonProject\heygem12
.venv\Scripts\python -X utf8 -u -m heygem12.cli --video examples\jie1_10s.mp4 --audio examples\gongting_10s.WAV --output out.mp4 --max-frames 20
# --device cpu|cuda  --max-frames N  --checkpoints DIR
# --restore          开 chaofen_after（GFPGAN 作用在渲染上；原版不做，默认关）
# --no-pre-restore   关 chaofen_before（默认开：GFPGAN 复原源帧后取 crop）
# --fuse-mode oval|fuse_mask   oval=原版脸椭圆硬贴（默认）；fuse_mask=旧羽化融合
# -u 必加(进度不被缓冲); -X utf8 建议加(有中文输出)
```

- 环境：`.venv` = Python 3.12.4 + torch 2.14.0+cpu + torchaudio 2.11 + opencv 5.0 + onnxruntime
  + **librosa 1.0.0 + scipy + resampy**（前端必需；服务器是 librosa 0.8.1）
  + **capstone + pyelftools**（逆向 .so 用）+ typeguard==2.13.3。
- pip 用阿里云镜像 `-i https://mirrors.aliyun.com/pypi/simple/`。
- ⚠️ `torchaudio.load` 本机缺 torchcodec 会 ImportError；前端走 `wave` + `librosa`，别用 torchaudio.load。
- ⚠️ ffmpeg 不在 PATH；`pipeline._write_video` 用 `imageio_ffmpeg.get_ffmpeg_exe()`。
- 性能：CPU 下 DINet ~3-4s/帧 + GFPGAN ~1-2s/帧（开 chaofen_before 再翻倍）。100 帧约 7-8 分钟。
- checkpoint 本地 `checkpoints\`（~890MB，md5 与服务器一致）：
  `wenetmodel.pt` md5 `396f59e6ca0523a4809710f235d1d611`(191093698B)、
  `dinet_v1_20240131.pth` md5 `edfa9458f418a8aa5905b4fc4fb9c70a`。

**服务器 assets 目录约定**（11p）：`media/`=标准素材；`out/`=标准 9 条输出 +
  历史产物；`test_media/{man,woman}/`=泛化 18 条（15s 截断）；`test_out/`=我们
  的泛化输出；`test_out_orig/`=原版泛化输出；`test_cmp/`=全部对比视频。

**仓库结构**
```
heygem12\
  cli.py / pipeline.py            # 入口 / 端到端（对齐→chaofen_before→渲染→oval 硬贴合成）
  ortutil.py                      # ★ ONNX 会话工厂：优先 CUDA、自动回退 CPU、预加载 nvidia 库（第十轮）
  face\align256.py                # ★ 256 对齐 + paste_back_oval（canonical 脸椭圆硬贴，第八轮）
                                    #   + TEMPLATE_256（SIFT 拟合模板，2.5 节）+ paste_mask_alpha（11l 新增）
  face\data\face_oval_256.npy     # ★ 脸椭圆合成掩码（9 帧原版 vid dump consensus th0.7+σ8）
  face\detection.py               # YuNet 人脸检测 + 五点（对齐链第一级）
  face\landmarks.py               # ★ PFLD-98（对齐关键点源：瞳孔/鼻尖/嘴角五点归约）
  face\detector.py                # ★ SCRFD（11i 修复锚点交织 bug；不做主检测器，
                                    #   用于 11l 垂直语义标定 + detect_refined 二次精检）
  models\dinet_hd.py / blocks.py  # ★ DINetV1，结构已逐字核对原版 repr
  audio\frontend.py               # ★ wav2mfcc_v2(hop160)+0.4s×2 静音 + concat_after=False
                                    #   + 窗口映射：视频短 floor(i*25/fps)，音频受限 span 重采样（11o）
  third_party\wenet\...           # vendored wenet v2.0.0（结构对，但 concat_after 配置见上）
scripts\                          # 在用工具：download_models.py / compare.py / run_compare.py（11k）
dev\                              # 开发过程产物，2026-10-02 归拢（不参与运行）：
  一次性分析脚本（见各轮引用：decompose_motion / compare_* / gap_* / debug_* /
  kps_dump / test_decoder_bn / measure_lip_sync(已证伪) / reverse_so / dump_compare）
  examples\                       # 开发期素材与证据图（jie1/gongting/out_current/cmp_*.jpg/
                                    #   region_compare.png 等；历史文档里的 examples\ 路径指这里）
  assets\                         # 服务器同步副本（out/test_cmp/test_media/test_out/test_out_orig；
                                    # 注意服务器上的同名目录仍在 assets\ 下，run_test2.sh 依赖）
  tests\                          # 回归测试 4 件（2026-10-02 随 tests 目录归入 dev\，
                                    #   运行路径 = dev\tests\test_*.py，见 §9 验证清单）
```

---

## 9. 验证清单（改动后必跑，均自带 sys.path）

```powershell
$env:PYTHONIOENCODING='utf-8'
.venv\Scripts\python -X utf8 dev\tests\test_model_load.py     # 416 keys | (1,3,256,256) range[0,1]
.venv\Scripts\python -X utf8 dev\tests\test_wenet_load.py     # (169,256) / windows(100,20,256)
.venv\Scripts\python -X utf8 dev\tests\test_audio_frontend.py # (269,256) std=0.1007 | CTC≥5 | cos≤0.80
.venv\Scripts\python -X utf8 dev\tests\test_alignment.py      # 眼55<86<嘴139 | BN比 | MAE≤0.055
.venv\Scripts\python -X utf8 dev\compare_same_media.py # 同源对数：音频 MAE≤5e-4 | crop MAE≤0.025 | 渲染 MAE≤0.001
.venv\Scripts\python -X utf8 dev\compare_vs_original.py # 端到端 vs 原版成片（dark-fraction 列已证伪，只看 MAE/corr 与 jpg）
.venv\Scripts\python -X utf8 dev\dump_compare.py examples\jie1_10s.mp4 examples\out_current.mp4
```
关键不变量：
- 特征 std ≈ **0.1007**（与原版 0.10076 一致）、帧数 = `25*(时长+0.8)-1`
- windows **T=20 / offset 0**、input_mask 首零行 **86**、模板眼行 **55** < 86 < 嘴行 **139** < 251
- `compare_same_media`：音频窗口 MAE ≤ 5e-4、masked crop MAE ≤ 0.025（恒等几何）、
  **他们 crop+音频 → 我们模型 vs 他们渲染 MAE ≤ 0.001**（模型等价硬指标）
- CTC token ≥ 5、cos(静音,语音) ≤ 0.80

---

## 10. 下一步（建议顺序）

0. ~~**服务器迁移跑 GPU**~~ ✅ **已 done（第十轮，§5b）**：三个 ONNX 会话全 GPU，
   1545 帧 1142s→163s。剩余空间：fuse（58.5s，CPU 图像处理为主）与逐帧串行架构
   （可批处理，原版就是 batch=4）、DINet ONNX 导出。
1. **P3 性能**：批处理（原版就是 batch=4）、ONNX 导出 DINet。
2. ~~**P0 嘴部残差**~~ ✅ **已基本闭环（11j/11l）**：牙齿（ref 方案）与双下巴
   （垂直语义标定）已修；剩余已知残余 = ①个别帧张嘴幅度略大、②运动模糊帧
   4-8px 漂移（原版同样存在）、③f129-133 等帧颏下 ~1-2px 内容差。若还要追，
   唯一已知更深来源是原版 GFPGAN 画布的噪声几何（§4 P0-1，已定为不可复刻）
   或其 .so 内 SCRFD 精确用法（可用 dev/reverse_so.py 逆向，成本高收益低）。
3. **泛化测试**：✅ 已做 9 条（demo1/2/3 + woman×2 + man×4，11l 批次全量重跑）；
   vcalib 的 k 逐视频自适配（1.06~1.19）。新素材建议先用
   `scripts/run_compare.py --show-mask` 出一条对比视频人工验收。

---

## 11. 逆向 .so 备忘（`dev/reverse_so.py`）

```powershell
python dev\reverse_so.py names <so> "audio_para" "adaAT" ...   # 字面量→引用它的函数
python dev\reverse_so.py trace <so> <func>                     # rip 相对引用序列
python dev\reverse_so.py plt   <so> <func>                     # 调用的 CPython API 统计
```
- 原理：Cython 把每个字面量在 `__Pyx_CreateStringTabAndInitStrings` 里与 `.bss/.data` 的
  `PyObject**` 全局配对（相邻 rip-relative `lea`），配对后反查 `.text` 的 rip 引用即可归属到函数。
- 已查明：`DINet.so` 有 4 个模型类（`DINet/DINetV1/DINetMouth/DINetMouth512`）、
  `AdaAT(+AdaAT_operator)`、`make_coordinate_grid_3d`、einops `rearrange/repeat`；
  `AdaAT.forward` 用了 einops 模式 `b (d h w) t e ↔ b d h w (t e)`、`b d e -> b d () () e`
  （开源版没有，但 `adaAT.*` 权重形状与开源一致 ⇒ 差异只在无参数的重排步骤；
  **注意**：既然 forward 已与原版逐位对齐，这条大概率不再是问题）。
- `digitalhuman_interface.so`：`tensor_norm = (x-127.5)/127.5`、`tensor_norm_no_training = x/255`。
- ❌ 走不通：`.so` 带完整 DWARF（`.debug_info` 691KB / `.debug_line` 450KB / `.debug_loc` 8.3MB），
  但 `.debug_line` 只映射到生成的 `/code/landmark2face_wy/models/DINet.c`（forward 在 C 行 20109+），
  **没有 .py 行号**；`DINetV1.forward` 的 DWARF 局部变量只有 4 个参数；`__pyx_n_s_*` 符号被 strip
  （只剩 `.rodata` 的 `__pyx_k_*`）。**但既然有可运行的原版环境，直接 hook 取真值远比反汇编可靠。**
- `compute_ctc_att_bnf.so` 的 API：`load_ppg_model(train_config, model_file, device)`、
  `get_weget(wavpath, wenet_model, section=560000)`、`compute_bnf(wav_dir, wenet_model, section)`、
  `PPGModel(encoder)` / `PPGModel._extract_feats(speech, speech_lengths)`（需 `self.frontend`，
  而 `load_ppg_model` 返回的实例没有该属性，故只能走 `get_weget`）、`hparams1`、`SAMPLE_RATE=16000`。

---

## 12. 授权与出处

- 动机：用户要把 HeyGem 迁到 Python 3.12；核心全是 `cpython-38` 的 `.so`；
  验证权重是标准 PyTorch 格式后反向工程重建。
- 出处：DINet=`github.com/MRzzm/DINet`（无 LICENSE，仅限研究）；wenet=`wenet-e2e/wenet`；
  YuNet=opencv_zoo；HeyGem=`github.com/Holasyb918/HeyGem-Linux-Python-Hack`
  （权重在 release `ckpts_and_onnx`，见仓库 `download.sh`）。
- **授权红线**：HeyGem 权重仅限个人/非商用；heygem12 自写代码可自由使用，商用需替换/自训 DINet 权重。
- 网络：github.com 直连超时；`raw.githubusercontent.com` / `api.github.com` 通；pip 用阿里云镜像。
- 本地临时资料（**Temp 可能被清，关键结论已录入本文档**）：
  `C:\Users\faye\AppData\Local\Temp\opencode\` 下
  `hookdir\sitecustomize.py`(全局 hook)、`probe*.sh`、`run_hooked*.sh`、
  `hgdumps\`(原版 dump)、`hgdump\`(get_weget.npy + 1004_format.*)、`heygem-src\`(各 .so 与明文 .py)、
  `wenet-official\`、`wenet-pkg\`、`20211025_conformer_exp\`(官方 aishell conformer，
  曾用来证明 vendored conformer 栈本身正确)、`real_repr.txt`(原版 DINetV1 完整 repr)、
  `patch_v21*.py`(V21 补丁系列)、`fit_*.py`/`cand_test.py`/`final_univ.py`(V23 模板拟合脚本，
  服务器副本在 /opt/projects/opencode/tmp/)。
  ⚠️ `rc.py`(旧 SSH 助手)已删除——远程连接改用 PuTTY（见 §1）。
