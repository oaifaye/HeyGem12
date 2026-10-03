# checkpoints

模型权重目录。本目录下的 6 个模型文件均为下载产物,**不要手工编辑**;
下载方式:运行 `python scripts/download_models.py` 自动下载全部,或点击下表链接逐个下载,
或从百度网盘下载全部(提取码 `mgas`)。

Model weights directory. All 6 files here are downloaded artifacts — do not
hand-edit. Fetch them with `python scripts/download_models.py`, via the links
below, or from the Baidu Netdisk mirror (extraction code `mgas`):

**百度网盘 Baidu Netdisk(含全部模型 / full set)**:
https://pan.baidu.com/s/109Npnbya4_r97e0BrJbRCg?pwd=mgas

| Model | Size | Purpose | Download |
|---|---|---|---|
| dinet_v1_20240131.pth | 374MB | 对口型核心网络 DINetV1 / core lip-sync network | [GitHub release](https://github.com/Holasyb918/HeyGem-Linux-Python-Hack/releases/download/ckpts_and_onnx/dinet_v1_20240131.pth) |
| wenetmodel.pt | 182MB | 语音特征 conformer encoder / speech features | [GitHub release](https://github.com/Holasyb918/HeyGem-Linux-Python-Hack/releases/download/ckpts_and_onnx/wenetmodel.pt) |
| GFPGANv1.4.onnx | 325MB | 人脸复原(输入超分)/ face restoration | [GitHub release](https://github.com/Holasyb918/HeyGem-Linux-Python-Hack/releases/download/ckpts_and_onnx/GFPGANv1.4.onnx) |
| pfpld_robust_sim_bs1_8003.onnx | 6.6MB | 98 关键点 / facial landmarks | [GitHub release](https://github.com/Holasyb918/HeyGem-Linux-Python-Hack/releases/download/ckpts_and_onnx/pfpld_robust_sim_bs1_8003.onnx) |
| scrfd_500m_bnkps_shape640x640.onnx | 2.4MB | 语义标定基准 / semantic calibration reference | [GitHub release](https://github.com/Holasyb918/HeyGem-Linux-Python-Hack/releases/download/ckpts_and_onnx/scrfd_500m_bnkps_shape640x640.onnx) |
| face_detection_yunet_2023mar.onnx | 0.2MB | 人脸检测 / face detection(OpenCV Zoo) | [OpenCV Zoo](https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx) |

共 ~890MB / ~890MB total。前 5 个来自
[HeyGem-Linux-Python-Hack](https://github.com/Holasyb918/HeyGem-Linux-Python-Hack)
的 `ckpts_and_onnx` Release;GitHub 直连慢时脚本自动走 gh-proxy 镜像。

完整性校验 / integrity (md5):

- `wenetmodel.pt`: `396f59e6ca0523a4809710f235d1d611`
- `dinet_v1_20240131.pth`: `edfa9458f418a8aa5905b4fc4fb9c70a`

⚠️ GFPGANv1.4.onnx 较大(~325MB),若下载中断请断点续传:

```bat
curl.exe -L -C - -o checkpoints\GFPGANv1.4.onnx ^
  "https://gh-proxy.com/https://github.com/Holasyb918/HeyGem-Linux-Python-Hack/releases/download/ckpts_and_onnx/GFPGANv1.4.onnx"
```
