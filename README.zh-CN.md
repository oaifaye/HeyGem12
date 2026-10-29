# HeyGem12

**[English](README.md) | 简体中文**

Python 3.12+ 的 HeyGem 核心对口型引擎重实现。输入视频 + 语音,输出对口型视频。

**为什么重写?** 原版 HeyGem 的大量核心模块(人脸检测、关键点、音频特征等)是
**闭源的 Cython 编译 `.so`**,只能运行在 **Python 3.8** 上,升级 Python 即崩溃。
本项目将推理链路全部重写为**纯 Python**,**不依赖任何 `.so`**,彻底摆脱
Python 3.8 绑定,可直接运行在现代 Python(3.12+)上。

本项目的模型权重(`dinet_v1_20240131.pth`、`wenetmodel.pt`)来自
[HeyGem-Linux-Python-Hack](https://github.com/Holasyb918/HeyGem-Linux-Python-Hack)
的公开下载;推理代码为纯 Python 重写。

**效果:与原版 HeyGem 输出效果一致。** 嘴型时序、牙齿、颜色、时序稳定性与
输出长度均与原版逐帧对齐(23 组同素材对比视频实测),同时修复了原版的若干
已知问题(多脸直接报错、批跑缓存错乱、露齿帧伪影等),并额外支持自动选脸、
60fps 素材与输出 faststart。

## 效果演示

6 个不同的源视频,分别用中文、英文语音驱动(推理设备:RTX 4090D,CUDA)。
点击文件名下载观看(mp4):

| 原视频 | 中文语音驱动 | 英文语音驱动 |
|---|---|---|
| [woman1_video.mp4](assets/woman1_video.mp4) | [woman1_cn.mp4](assets/woman1_cn.mp4) | [woman1_en.mp4](assets/woman1_en.mp4) |
| [woman2_video.mp4](assets/woman2_video.mp4) | [woman2_cn.mp4](assets/woman2_cn.mp4) | [woman2_en.mp4](assets/woman2_en.mp4) |
| [man1_video.mp4](assets/man1_video.mp4) | [man1_cn.mp4](assets/man1_cn.mp4) | [man1_en.mp4](assets/man1_en.mp4) |
| [man2_video.mp4](assets/man2_video.mp4) | [man2_cn.mp4](assets/man2_cn.mp4) | [man2_en.mp4](assets/man2_en.mp4) |
| [man3_video.mp4](assets/man3_video.mp4) | [man3_cn.mp4](assets/man3_cn.mp4) | [man3_en.mp4](assets/man3_en.mp4) |
| [man4_video.mp4](assets/man4_video.mp4) | [man4_cn.mp4](assets/man4_cn.mp4) | [man4_en.mp4](assets/man4_en.mp4) |

驱动音频:[man_cn.wav](assets/man_cn.wav) / [woman_cn.wav](assets/woman_cn.wav)(中文)、
[man_en.wav](assets/man_en.wav) / [women_en.wav](assets/women_en.wav)(英文)。

## 原理

整条推理链完全复刻原版 HeyGem 的逻辑（通过 Ghidra 反编译原版 `.so` 并用
运行时 trace/hook 逐一实证每个常数后重写；运行时不依赖任何 `.so`）：

```
音频 wav ──dB-mel──▶ WeNet conformer encoder(256 维, 25fps)
视频 mp4 ──逐帧──▶ SCRFD 检测 ──▶ bbox 左右/下方 +10% 裁剪
                        ──resize 112──▶ PFLD-98 ──convert98to68──▶ 68 点 landmark
                        ▶ boundingRect: x1=xc−0.75w, y1=y−0.15w, side=1.5w(轴对齐,无旋转)
                        ▶ 时域平滑:居中 conv5 滑动平均 + 12px 去抖(非因果,全片)
                        ▶ 矩形 crop ──resize 266×266 INTER_CUBIC──▶ DINet 输入 = crop[5:-5](256×256)
                        ▶ (可选)GFPGAN 超分:mtcnn_512 norm_crop + 静态椭圆 mask
                        ▶ DINetV1: 遮罩 crop + 同帧未遮罩参考图 + 音频窗(T=20)
                 ──▶ 嘴部 inpainting + AdaAT 外观变形
                        ▶ 贴回:266 画布 resize 回矩形,经逐帧 landmark mask 融合——
                          convexHull(下颌[2:15]+鼻尖) + 下半椭圆 + dilate 6×6 ──▶ 输出 mp4(+驱动音频)
```

要点:

- 音频前端:16 kHz 单声道、首尾各 0.4 s 静音、80 维 dB-mel,经 conformer
  encoder 输出 256 维特征(25 fps);视频帧 `i` 的音频窗按视频帧率与音频
  时长对齐——视频较短时逐帧滑动取窗,视频较长时把可用特征跨度线性
  重采样到输出长度(25/30/60 fps 均已适配)。
- landmark 链(精确复刻):SCRFD bbox → 裁剪 `[x1−0.1w, y1, x2+0.1w, y2+0.1h]`
  → 各向异性 resize 112 → PFLD-98 → 原版的 `convert98to68` 索引映射
  (眼部点为两点均值)→ 68 点。裁剪矩形由 `boundingRect(landmarks68)` 按
  原版公式(x1=xc−0.75w, y1=y−0.15w, side=1.5w)计算,并按原版的居中
  窗口 5 卷积 + >12px 直通去抖做时域平滑。
- 合成:逐帧贴回 mask 与原版 `get_face_mask` 一致(IoU≈0.99):下颌(去太阳穴
  端点)+鼻尖的凸包,加下颌几何驱动的下半椭圆,dilate 6×6;mask 随 266 画布
  resize 回矩形贴入,非脸区域像素与输入帧逐位一致。

## 相比上一版(YuNet/Align256 时代)的优化

- **检测**:YuNet → SCRFD(与原版相同的 insightface 语义)。
- **裁剪几何**:自研五点仿射 warp(TEMPLATE_K,带旋转)→ 原版 68 点
  landmark 驱动的轴对齐矩形裁剪 + 原版非因果时域平滑;消除旋转抖动和
  逐人脸的模板失配(脸忽胖忽瘦)。
- **关键点**:PFLD-98 五点 → 从原版二进制实证解出的 `convert98to68` 精确
  映射(误差为 0)。
- **贴回 mask**:固定脸椭圆 → 原版逐帧 landmark mask(下颌凸包+下半椭圆
  +dilate),消除可见的椭圆接缝与"双下巴"伪影。
- **人脸复原**:新增原版 GFPGAN 超分(chaofen)链:mtcnn_512 norm_crop →
  GFPGAN → 反向矩阵贴回(静态椭圆 common_face_mask),与 cv2box 语义完全一致。
- **验证方式**:全部几何常数均来自原版二进制的证据(Ghidra 反编译 +
  cv2/运行时 hook),不再依赖逐视频 SIFT 拟合或手工调参。

## 安装

支持 Python 3.12+(无需 3.8)。CPU 与 GPU 分别使用不同的依赖文件:

```bash
cd heygem12
python -m venv .venv

# CPU 版
.venv\Scripts\python -m pip install -r requirements-cpu.txt

# GPU 版(CUDA 12.1,torch 自动从 PyTorch 官方源安装)
.venv\Scripts\python -m pip install -r requirements-gpu.txt

python scripts\download_models.py        # 下载模型到 checkpoints\ (~890MB)
```

ONNX 运行时(GFPGAN/SCRFD/PFLD)自动优先 CUDA、失败回退 CPU,无需手动配置。

## 模型清单

下载方式三选一：运行 `python scripts\download_models.py` 自动下载全部；或点击下表链接逐个下载；或从**百度网盘**下载全部模型(提取码 `mgas`):[https://pan.baidu.com/s/109Npnbya4_r97e0BrJbRCg?pwd=mgas](https://pan.baidu.com/s/109Npnbya4_r97e0BrJbRCg?pwd=mgas)

| 模型 | 大小 | 用途 |
|---|---|---|
| [dinet_v1_20240131.pth](https://github.com/Holasyb918/HeyGem-Linux-Python-Hack/releases/download/ckpts_and_onnx/dinet_v1_20240131.pth) | 374MB | ★对口型核心网络 DINetV1:遮罩 crop + 同帧参考图 + 音频窗 → 新嘴型渲染 |
| [wenetmodel.pt](https://github.com/Holasyb918/HeyGem-Linux-Python-Hack/releases/download/ckpts_and_onnx/wenetmodel.pt) | 182MB | 语音特征:conformer encoder,wav → 256 维声学特征(25fps) |
| [GFPGANv1.4.onnx](https://github.com/Holasyb918/HeyGem-Linux-Python-Hack/releases/download/ckpts_and_onnx/GFPGANv1.4.onnx) | 325MB | 人脸复原:源帧人脸超分,为 DINet 提供锐利输入(眼/颈部保留原图) |
| [pfpld_robust_sim_bs1_8003.onnx](https://github.com/Holasyb918/HeyGem-Linux-Python-Hack/releases/download/ckpts_and_onnx/pfpld_robust_sim_bs1_8003.onnx) | 6.6MB | 人脸关键点:PFLD-98 点,经原版 convert98to68 转成 68 点,驱动裁剪矩形与贴回 mask |
| [scrfd_500m_bnkps_shape640x640.onnx](https://github.com/Holasyb918/HeyGem-Linux-Python-Hack/releases/download/ckpts_and_onnx/scrfd_500m_bnkps_shape640x640.onnx) | 2.4MB | 人脸检测:逐帧脸框与五点(与原版同模型、同 insightface 语义) |
| [face_detection_yunet_2023mar.onnx](https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx) | 0.2MB | (旧版遗留备用,默认 SCRFD 链不再使用) |

共 ~890MB,放入 `checkpoints/` 目录即可。前 5 个来自
[HeyGem-Linux-Python-Hack](https://github.com/Holasyb918/HeyGem-Linux-Python-Hack)
的 `ckpts_and_onnx` Release;GitHub 直连慢时脚本会自动走 gh-proxy 镜像。

## 使用

```bash
.venv\Scripts\python -m heygem12.cli --video input.mp4 --audio voice.wav --output result.mp4
```

参数:

| 参数 | 说明 |
|---|---|
| `--video` | 源视频(人物正脸最佳;25/30/60fps 均可) |
| `--audio` | 驱动音频(wav,任意时长;音频较长时输出按音频时长截断) |
| `--output` | 输出 mp4 路径 |
| `--device` | `cpu`(默认)或 `cuda`(需 GPU 版 torch) |
| `--max-frames` | 限制处理帧数(调试用) |
| `--no-kps-stabilize` | 关闭关键点时序稳定(默认开,建议保持) |
| `--no-pre-restore` | 关闭 GFPGAN 源帧复原(默认开) |
| `--restore` | 开启渲染后 GFPGAN(默认关) |
| `--fuse-mode` | `oval`(默认,脸椭圆硬贴)或 `fuse_mask`(羽化融合) |

## Python API

```python
from heygem12.pipeline import LipSyncPipeline

pipe = LipSyncPipeline("checkpoints", device="cpu")
pipe.run("input.mp4", "voice.wav", "result.mp4")
```

## 速度参考

| 硬件 | 合成(含 GFPGAN 复原) | 15s 视频(900 帧 @60fps)全程 |
|---|---|---|
| CPU(16 线程) | ~600 ms/帧 | 不推荐 |
| RTX 4090D(CUDA) | ~76 ms/帧 | ~100 s |

(人脸检测 ~25 ms/帧、写视频另计;fuse 阶段为 CPU 图像处理,约 38 ms/帧。)

## 已知限制

- 个别帧嘴型幅度有 ±20~30% 的偏差;快速转头帧(运动模糊)有几帧 4-8px
  的几何偏移,属检测器噪声地板;颏下区域有 ~1-2px 内容级差异。
- 模型权重仅限个人/非商业用途(HeyGem 协议);代码部分可自由使用,
  商用需替换/自训 DINet 权重。

## 目录结构

```
heygem12/
├── heygem12/
│   ├── models/        DINetV1 网络(反向工程重实现)
│   ├── audio/         WeNet 特征前端(vendored wenet transformer)
│   ├── face/          SCRFD 检测 / PFLD-98 关键点 / convert98to68 与裁剪、mask 几何
│   ├── third_party/   wenet transformer 代码(Apache-2.0)
│   ├── pipeline.py    端到端管线
│   └── cli.py         命令行入口
├── scripts/           模型下载 / 对比视频工具
├── checkpoints/       模型权重(下载后 ~890MB)
└── assets/            演示视频与驱动音频
```
