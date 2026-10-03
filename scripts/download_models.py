"""Download all model checkpoints into ./checkpoints.

Sources:
  - dinet_v1_20240131.pth / wenetmodel.pt / scrfd / pfpld / GFPGANv1.4.onnx:
    HeyGem-Linux-Python-Hack releases (github, fetched via gh-proxy).
  - face_detection_yunet_2023mar.onnx: opencv_zoo (media.githubusercontent.com).

注意: GFPGANv1.4.onnx ~340MB, gh-proxy 单连接较慢且容易超时,
若下载中断, 用 curl 断点续传:
  curl.exe -L -C - -o checkpoints\GFPGANv1.4.onnx ^
    "https://gh-proxy.com/https://github.com/Holasyb918/HeyGem-Linux-Python-Hack/releases/download/ckpts_and_onnx/GFPGANv1.4.onnx"
"""

import os
import urllib.request

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")

GH_PROXY = "https://gh-proxy.com/"
RELEASE = "https://github.com/Holasyb918/HeyGem-Linux-Python-Hack/releases/download/ckpts_and_onnx/"

FILES = {
    "dinet_v1_20240131.pth": GH_PROXY + RELEASE + "dinet_v1_20240131.pth",
    "wenetmodel.pt": GH_PROXY + RELEASE + "wenetmodel.pt",
    "pfpld_robust_sim_bs1_8003.onnx": GH_PROXY + RELEASE + "pfpld_robust_sim_bs1_8003.onnx",
    "scrfd_500m_bnkps_shape640x640.onnx": GH_PROXY + RELEASE + "scrfd_500m_bnkps_shape640x640.onnx",
    "GFPGANv1.4.onnx": GH_PROXY + RELEASE + "GFPGANv1.4.onnx",
    "face_detection_yunet_2023mar.onnx":
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
        "models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
}


def main():
    os.makedirs(ROOT, exist_ok=True)
    for name, url in FILES.items():
        dst = os.path.join(ROOT, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 1000:
            print(f"[skip] {name}")
            continue
        print(f"[downloading] {name} ...")
        urllib.request.urlretrieve(url, dst)
        print(f"  -> {dst} ({os.path.getsize(dst) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
