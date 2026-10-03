"""256x256 face alignment for DINetV1, driven by the checkpoint's own masks.

``dinet_v1_20240131.pth`` ships inference ground truth at the top level (outside
``face_G``):

    model_input_size  = (256, 256)     # the network's expected input
    model_output_size = (256, 256)
    input_mask        = (256,256) int   # zeroed region: rows ~86-251, cols ~33-222
    fuse_mask         = (256,256) float # feathered blend region: rows 94-234
    removegreen       = True
    wh                = 0.97

So the real pipeline is NOT open-DINet's 320x416 mouth crop with a hard zero
rectangle. It aligns the face to a canonical 256x256 crop (``use_scrfd: True``
in the metadata), multiplies by ``input_mask``, renders 256x256, and fuses with
``fuse_mask``. Feeding 320x416 crops put the network out of distribution:
self-reconstruction MAE 0.066 vs 0.041 for the aligned input, and mouth
openness stopped tracking the audio entirely.

``removegreen`` is deliberately NOT implemented. A naive reading (clamp the green
channel wherever it dominates) was measured: the ground-truth crop of the sample
video already has 33.9% green-dominant pixels (green backdrop/clothing) versus
33.8% for the render, so green is legitimate scene content here, and clamping it
raised self-reconstruction MAE from 0.0406 to 0.0714. The flag presumably targets
a specific fringing artifact of HeyGem's own renderer; without their code the
semantics are unknown and guessing is measurably harmful.

``wh`` (0.97) is likewise unverified; it is exposed as ``fuse_strength`` and
calibrated by self-reconstruction MAE rather than assumed.

The canonical layout is a similarity transform of the arcface-112 5-point
template scaled by ``TEMPLATE_K`` with the eye mid-point placed at
``EYE_CENTER``; both were calibrated by minimising self-reconstruction MAE
and matching ``source_in_conv``'s BatchNorm ``running_var`` (the constraint that
matters is geometric: the eyes must sit above ``input_mask``'s first zeroed row
~86 while the mouth falls inside it, which rules out the naive 256/112 arcface
scale that would put the eyes at row 118).
"""

import os
from typing import Optional, Tuple

import cv2
import numpy as np

# arcface-112 5-point template (L eye, R eye, nose, L mouth, R mouth)
ARCFACE112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

INPUT_SIZE = 256
TEMPLATE_K = 2.0                      # arcface-112 scale factor for the 256 crop
EYE_CENTER = (128.0, 70.0)            # eye mid-point position inside the crop
TEMPLATE_CENTROID = 56.0              # arcface-112 reference centre
# (rx, ry, feather) of the mouth-limited blend region; None keeps the full
# shipped fuse_mask. See Align256._effective_mask for the measurement.
MOUTH_ELLIPSE = (75, 60, 41)

# Canonical 5-point layout actually used, fitted against the original pipeline's
# own dumped crops (see scripts/fit_their_template.py). Equivalent to an arcface
# template with K=2.078, eye mid-point (125.75, 54.79) and a 1.354 deg eye-line
# tilt. Measured on 12 frames spanning three separate dump batches:
#     using this template    crop MAE 0.0074  corr 0.9934
#     template_256(2.0, 70)  crop MAE 0.0474  corr 0.8013
# The residual tilt is a fixed offset between our PFLD-98 five points and the
# SCRFD five points the original aligns with (spread 0.30 deg across frames at
# three different timestamps), not a property of this clip.
TEMPLATE_256 = np.array([
    [89.151, 53.921],    # left eye
    [162.339, 55.650],   # right eye
    [124.758, 96.612],   # nose tip
    [93.447, 138.575],   # left mouth corner
    [154.055, 140.007],  # right mouth corner
], dtype=np.float32)


def template_256(K: float = TEMPLATE_K,
                 center: Tuple[float, float] = EYE_CENTER) -> np.ndarray:
    """5-point reference template laid out for a 256x256 crop."""
    t = (ARCFACE112 - TEMPLATE_CENTROID) * K
    eye_mid = ((ARCFACE112[0] + ARCFACE112[1]) / 2.0 - TEMPLATE_CENTROID) * K
    return (t + np.asarray(center, np.float32) - eye_mid).astype(np.float32)


class Align256:
    """Align frames to the canonical 256x256 DINet input and fuse renders back."""

    def __init__(self, meta: dict, K: float = TEMPLATE_K,
                 center: Tuple[float, float] = EYE_CENTER,
                 fuse_strength: float = 1.0,
                 mouth_ellipse: Optional[Tuple[int, int, int]] = MOUTH_ELLIPSE,
                 template: Optional[np.ndarray] = None):
        self.input_size = int(INPUT_SIZE)
        self.input_mask = np.asarray(meta["input_mask"]).astype(np.float32)[:, :, None]
        self.fuse_mask = np.asarray(meta["fuse_mask"]).astype(np.float32)
        self.fuse_strength = float(fuse_strength)
        # TEMPLATE_256 (fitted against the original's dumps) is the default;
        # K/center rebuild a plain untilted arcface template, useful for A/B.
        self.template = (np.asarray(template, dtype=np.float32) if template is not None
                         else TEMPLATE_256.copy())
        self.mouth_ellipse = mouth_ellipse
        self._eff_mask = self._effective_mask()

    def _effective_mask(self, strength: Optional[float] = None) -> np.ndarray:
        """fuse_mask optionally intersected with a feathered mouth-centred ellipse.

        The shipped ``fuse_mask`` blends the whole lower face (rows 94-234,
        cols 12-244 = 38.8% of the crop). That is right for HeyGem's renderer,
        but ours is low-pass (the checkpoint's decoder BatchNorm stats are
        inconsistent with its weights, squeezing every residual branch), so
        replacing cheeks and chin with it reads as a texture patch against the
        untouched upper face. Restricting the blend to the mouth is measured
        better on near-silent audio, where the ideal output equals the source:

            extent        fused%  MAE     corr     speech/silence openness
            full mask      38.8   0.0365  0.9604   2.2x
            ellipse 90x75  30.1   0.0313  0.9669   3.1x
            ellipse 75x60  21.6   0.0244  0.9740   3.2x   <- adopted
            ellipse 60x48  13.9   0.0167  0.9813   3.1x

        75x60 leaves ample margin for wide openings and jaw movement (the mouth
        itself is only ~58 px wide in the crop) while cutting the replaced area by
        nearly half. Lip contrast is preserved -- it does not trade motion away.
        """
        s = self.fuse_strength if strength is None else float(strength)
        base = np.clip(self.fuse_mask * s, 0.0, 1.0)
        if not self.mouth_ellipse:
            return base
        rx, ry, feather = self.mouth_ellipse
        if rx <= 0 or ry <= 0:
            return base
        cx = float((self.template[3][0] + self.template[4][0]) / 2)
        cy = float((self.template[3][1] + self.template[4][1]) / 2) + 8.0
        ell = np.zeros((self.input_size, self.input_size), np.float32)
        cv2.ellipse(ell, (int(round(cx)), int(round(cy))), (int(rx), int(ry)),
                    0, 0, 360, 1.0, -1)
        k = 2 * (int(feather) // 2) + 1
        ell = cv2.GaussianBlur(ell, (k, k), 0)
        return base * ell

    # ------------------------------------------------------------------ align
    def matrix(self, five_kps: np.ndarray) -> Optional[np.ndarray]:
        """Similarity transform mapping frame coords -> 256x256 crop coords."""
        M, _ = cv2.estimateAffinePartial2D(
            np.asarray(five_kps, dtype=np.float32), self.template, method=cv2.LMEDS)
        return M

    def align(self, frame: np.ndarray, five_kps: np.ndarray):
        """Returns (rgb01 crop (256,256,3), affine matrix) or (None, None)."""
        M = self.matrix(five_kps)
        if M is None:
            return None, None
        crop = cv2.warpAffine(frame, M, (self.input_size, self.input_size))
        rgb = crop[:, :, ::-1].astype(np.float32) / 255.0
        return rgb, M

    def mask_source(self, rgb: np.ndarray) -> np.ndarray:
        """The network's source input: RGB [0,1] with input_mask applied."""
        return rgb * self.input_mask

    # ------------------------------------------------------------------- fuse
    def paste_back(self, frame: np.ndarray, M: np.ndarray, rendered: np.ndarray,
                   strength: Optional[float] = None) -> np.ndarray:
        """Fuse a rendered 256x256 RGB crop into the frame via fuse_mask.

        Compositing happens in crop space (where the mask is defined), then the
        result is warped back with the inverse transform and blended against the
        original pixels by the inverse-warped mask, so no rectangular seam
        appears in frame space.
        """
        m = (self._eff_mask if strength is None
             else self._effective_mask(float(strength)))[:, :, None]
        rgb = frame[:, :, ::-1].astype(np.float32) / 255.0
        w = cv2.warpAffine(rgb, M, (self.input_size, self.input_size))
        fused_crop = np.clip(rendered * m + w * (1.0 - m), 0.0, 1.0)

        inv = cv2.invertAffineTransform(M)
        h, wd = frame.shape[:2]
        fused_frame = cv2.warpAffine(fused_crop, inv, (wd, h))
        mask_frame = cv2.warpAffine(m[:, :, 0], inv, (wd, h))
        # only touch pixels the crop actually covers
        cover = cv2.warpAffine(np.ones((self.input_size, self.input_size), np.float32),
                               inv, (wd, h))
        alpha = np.clip(mask_frame * cover, 0.0, 1.0)[:, :, None]
        out = frame.astype(np.float32) / 255.0
        out = fused_frame[:, :, ::-1] * alpha + out * (1.0 - alpha)
        return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)

    def paste_back_oval(self, frame: np.ndarray, M: np.ndarray,
                        rendered: np.ndarray) -> np.ndarray:
        """Composite the render into the frame through the canonical face-oval
        mask (hard paste, feathered edge) -- the original's actual scheme.

        Round-7 server dumps (hooking the original's VideoWriter) show its
        final frame equals the RAW source frame outside a stable ~45% face-oval
        region in crop space, and equals the DINet render inside it (median
        |vid-render| ~2.7/255). No GFPGAN content reaches the output; the
        restoration only sharpens the DINet INPUT crop (chaofen_before).
        ``data/face_oval_256.npy`` is that mask: the per-pixel fraction of
        frames whose vid differed from src (9 sample frames), thresholded at
        0.7 and feathered with sigma 8; measured reconstruction MAE vs the
        original's final frames ~2.8/255 on the valid face region.

        Two measured fixes on top of the raw scheme (ours render differs from
        src more than theirs does, so a naive full-oval paste showed artifacts
        theirs doesn't):

        * **Gain-match**: our render carries a small global dark shift
          (face interior ~3.4/255 darker than src, from the decoder BN
          stats). Per channel we fit ``render ~ gain*src + offset`` on the
          unmasked band (which the network should reproduce) and apply it to
          the whole render before pasting.
        * **Border falloff**: the oval mask touches the crop's top row, and
          render border pixels are unreliable (conv edge), which drew a
          visible dark square contour along the crop boundary. The mask is
          multiplied by a distance-to-crop-border ramp and re-feathered, so
          the paste alpha reaches 0 well inside the crop.
        """
        import os
        m = self._oval_mask()[:, :, None]

        rgb = frame[:, :, ::-1].astype(np.float32) / 255.0
        w = cv2.warpAffine(rgb, M, (self.input_size, self.input_size))

        # gain-match the render to the source on the unmasked band
        band = self.input_mask[..., 0] > 0.5
        r = rendered[band].reshape(-1, 3)
        s = w[band].reshape(-1, 3)
        gain = np.ones(3, np.float32)
        off = np.zeros(3, np.float32)
        for c in range(3):
            A = np.stack([r[:, c], np.ones(len(r), np.float32)], axis=1)
            sol, *_ = np.linalg.lstsq(A, s[:, c], rcond=None)
            gain[c], off[c] = sol
        rendered = np.clip(rendered * gain + off, 0.0, 1.0)

        fused_crop = np.clip(rendered * m + w * (1.0 - m), 0.0, 1.0)

        inv = cv2.invertAffineTransform(M)
        h, wd = frame.shape[:2]
        fused_frame = cv2.warpAffine(fused_crop, inv, (wd, h))
        mask_frame = cv2.warpAffine(m[:, :, 0], inv, (wd, h))
        cover = cv2.warpAffine(np.ones((self.input_size, self.input_size), np.float32),
                               inv, (wd, h))
        alpha = np.clip(mask_frame * cover, 0.0, 1.0)[:, :, None]
        out = frame.astype(np.float32) / 255.0
        out = fused_frame[:, :, ::-1] * alpha + out * (1.0 - alpha)
        return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)

    def _oval_mask(self) -> np.ndarray:
        """Canonical face-oval paste mask (256x256 float, feathered, with the
        border falloff applied). Built lazily from ``data/face_oval_256.npy``.

        A bottom taper fades the mask out below the chin (crop rows 208→246):
        the oval's lower edge otherwise sits ~35 px below the chin on bare
        neck skin, and the network-repainted neck shadow lands darker than the
        raw frame there -- a visible dark seam on clothing-free necks
        (measured woman_cn: under-chin dip 92 vs raw 114, orig only 101).
        The neck carries no speech signal, so blending it back to raw is free.
        """
        if not hasattr(self, "_oval_mask_b"):
            import os
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "data", "face_oval_256.npy")
            base = np.load(path).astype(np.float32)
            side = self.input_size
            yy, xx = np.mgrid[0:side, 0:side]
            dist = np.minimum(np.minimum(yy, xx),
                              np.minimum(side - 1 - yy, side - 1 - xx)).astype(np.float32)
            border_w = np.clip((dist - 4.0) / 28.0, 0.0, 1.0)
            taper = np.clip((246.0 - np.arange(side, dtype=np.float32))
                            / (246.0 - 208.0), 0.0, 1.0)
            self._oval_mask_b = cv2.GaussianBlur(base * border_w * taper[:, None],
                                                 (0, 0), 4.0)
        return self._oval_mask_b

    def paste_mask_alpha(self, frame_shape, M: np.ndarray) -> np.ndarray:
        """The oval paste alpha warped into frame coordinates: (H, W) float in
        [0, 1], exactly the blend weight ``paste_back_oval`` uses. Useful for
        visualising the replaced region."""
        m = self._oval_mask()[:, :, None]
        inv = cv2.invertAffineTransform(M)
        h, wd = frame_shape[:2]
        mask_frame = cv2.warpAffine(m[:, :, 0], inv, (wd, h))
        cover = cv2.warpAffine(np.ones((self.input_size, self.input_size), np.float32),
                               inv, (wd, h))
        return np.clip(mask_frame * cover, 0.0, 1.0)


def default_fuse_strength(meta: dict, fallback: float = 1.0) -> float:
    """Blend strength from the checkpoint metadata (``wh``), else ``fallback``."""
    try:
        v = float(meta.get("wh", fallback))
    except (TypeError, ValueError):
        return fallback
    return v if 0.0 < v <= 1.0 else fallback


def load_meta(checkpoint_path: str) -> dict:
    """Load the non-weight metadata + masks shipped inside the checkpoint."""
    import torch

    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    need = ("input_mask", "fuse_mask")
    for k in need:
        if k not in ck:
            raise KeyError(f"checkpoint lacks {k!r}; expected HeyGem dinet_v1 metadata")
    return {k: v for k, v in ck.items() if k != "face_G"}
