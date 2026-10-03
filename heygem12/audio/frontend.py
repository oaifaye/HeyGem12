"""WeNet-based audio frontend: wav -> 256-dim per-frame features for DINet_HD.

Frontend pipeline (reproduces HeyGem's ``wenet/compute_ctc_att_bnf``):

    wav (16k) -> preemphasis 0.97 -> STFT(n_fft 1024, hop 200, win 800, hann)
              -> power -> mel(80, fmin 0, fmax 8000) -> 10*log10(p+1e-5) - 20
              -> wenet ConformerEncoder -> (T, 256) encoder features

This is ``wav2mfcc_v2`` from HeyGem's ``wenet/tools/_extract_feats.py`` returning
``log_melspec`` in dB WITHOUT the ``log_power_normalize`` [0,1] step. It is NOT
kaldi fbank: feeding kaldi fbank to this checkpoint makes the encoder produce
speech-agnostic output (CTC greedy decode yields 0 tokens, silence/speech feature
cosine 0.91), whereas the dB-mel input decodes Chinese correctly (30 tokens,
cosine 0.37) and matches DINet's ``audio_encoder`` BatchNorm statistics far
better (predicted/training variance ratio 0.45 vs 0.06).

Mel frames are at 80 fps (hop 200 @ 16 kHz); the encoder's conv2d subsampling
reduces that by 4, so features are at ~20 fps.

HeyGem's dataset builds a 20-frame window per video frame
(``obtain_seq_index(i) = [i-10, i+9]``), so ``frame_windows`` uses a 20-wide
window rather than open-DINet's 5-wide deepspeech window; measured audio
conditioning is ~3.5x stronger at T=20.
"""

import os
import sys
from typing import Optional

import numpy as np
import torch

# make vendored wenet importable: <project_root>/heygem12/third_party
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # heygem12 package dir
_THIRD_PARTY = os.path.join(_PKG_DIR, "third_party")
if _THIRD_PARTY not in sys.path:
    sys.path.insert(0, _THIRD_PARTY)

from wenet.transformer.encoder import ConformerEncoder  # noqa: E402

# encoder config mirrored from HeyGem's train_conformer_multi_cn.yaml
# (dropout values are irrelevant at inference time)
#
# concat_after=False is NOT a guess: it was read off the original's live module
# (`ConformerEncoderLayer.ff_scale=0.5, normalize_before=True,
# concat_after=False`). The checkpoint does contain `encoders.N.concat_linear`
# (256,512) weights, and an earlier round wrongly inferred concat_after=True
# from their presence -- but the original creates that Linear unconditionally
# and simply does not use it, falling back to the additive residual
# `x + dropout(self_attn(...))`. With concat_after=True our features correlate
# only 0.757 with theirs (MAE 0.050 on identical mel input, embedding output
# bit-identical, divergence compounding from encoder layer 0), which is enough
# to break DINet's audio conditioning entirely.
ENCODER_CONF = dict(
    output_size=256,
    attention_heads=4,
    linear_units=2048,
    num_blocks=12,
    dropout_rate=0.1,
    positional_dropout_rate=0.1,
    attention_dropout_rate=0.0,
    input_layer="conv2d",
    normalize_before=True,
    cnn_module_kernel=15,
    use_cnn_module=True,
    activation_type="swish",
    pos_enc_layer_type="rel_pos",
    selfattention_layer_type="rel_selfattn",
    concat_after=False,
)

# weights present in the checkpoint but unused by the encoder configuration
# above (created but not consulted); strict loading must tolerate them.
UNUSED_KEYS_PREFIX = ("global_cmvn", "concat_linear", "decoders.", "ctc.")

_SAMPLE_RATE = 16000

# wav2mfcc_v2 hparams -- taken from the `hparams1` dict that the compiled
# wenet/compute_ctc_att_bnf.so actually uses (NOT the `hparams` dict in
# wenet/tools/_extract_feats.py, whose hop_length=200 is dead configuration).
_HP = dict(
    preemphasis=0.97,
    n_fft=1024,
    hop_length=160,      # 10 ms -> 100 fps mel frames
    win_length=800,      # 50 ms window
    num_mels=80,
    fmin=0.0,
    fmax=8000.0,
    ref_db=20,
)

# `add_silence_flag`: 6400 samples (0.4 s) of zeros are concatenated to BOTH
# ends of the waveform before extraction. Measured directly against the
# original's dumped features -- with this padding our features reproduce theirs
# bit-exactly (MAE 0.000000, per-frame corr 1.0000); dropping it, halving it or
# moving it to one side only raises MAE to ~0.09.
SILENCE_PAD = 6400

# mel frames per second, and the encoder's subsampling factor
MEL_FPS = _SAMPLE_RATE / _HP["hop_length"]          # 100
SUBSAMPLE = 4
FEATURE_FPS = MEL_FPS / SUBSAMPLE                   # 25 (== video rate)

# DINet audio window length and its start offset in feature frames.
# Recovered by exact-MAE search against the original pipeline's dumped
# audio_feature tensors, using the audio file the original actually processed
# (an ffmpeg-converted 16 kHz mono pcm_s16le wav): for video frame i the window
# is feats[i : i + WINDOW_T], i.e. stride 1 and NO offset. (An earlier "+17"
# reading was an artifact of matching against the wrong audio file; with the
# wrong audio no offset fits and the argmax is meaningless.)
WINDOW_T = 20
WINDOW_OFFSET = 0


class WeNetFeatureExtractor:
    """Extracts (T, 256) audio features at ~20 fps, aligned to video frames."""

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = device
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = ConformerEncoder(input_size=_HP["num_mels"], **ENCODER_CONF)
        encoder_sd = {k[len("encoder."):]: v for k, v in ckpt.items() if k.startswith("encoder.")}
        missing, unexpected = model.load_state_dict(encoder_sd, strict=False)
        # concat_after=False means the layer never builds concat_linear, so those
        # checkpoint tensors are expected to be unused (they ship anyway). Match by
        # substring: the keys arrive as "encoders.N.concat_linear.weight".
        skipped = [k for k in unexpected if any(p in k for p in UNUSED_KEYS_PREFIX)]
        unexpected = [k for k in unexpected if k not in skipped]
        if unexpected:
            raise RuntimeError(f"unexpected keys when loading wenet encoder: {unexpected[:5]}")
        if missing:
            raise RuntimeError(f"missing keys when loading wenet encoder: {missing[:5]}")
        model.to(device).eval()
        self.model = model
        self._mel_basis = None
        self.skipped_keys = skipped

    # ------------------------------------------------------------- audio load
    def _load_wav(self, wav_path: str):
        """Load wav -> (float32 tensor (1, N) in [-1, 1], sample_rate)."""
        import wave

        try:
            with wave.open(wav_path, "rb") as wf:
                sr = wf.getframerate()
                ch = wf.getnchannels()
                sw = wf.getsampwidth()
                raw = wf.readframes(wf.getnframes())
            if sw == 2:
                data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            elif sw == 4:
                data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
            elif sw == 1:
                data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            else:
                raise ValueError(f"unsupported sample width: {sw}")
            if ch > 1:
                data = data.reshape(-1, ch).mean(axis=1)
            return torch.from_numpy(data).unsqueeze(0), sr
        except (wave.Error, EOFError):
            pass  # not a plain PCM wav; fall through to librosa
        import librosa

        y, sr = librosa.load(wav_path, sr=None, mono=True)
        return torch.from_numpy(np.ascontiguousarray(y)).unsqueeze(0), sr

    # --------------------------------------------------------- mel front end
    def _log_mel_db(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """wav2mfcc_v2's log_melspec: (T_mel, 80) in dB (ref_db=20)."""
        import librosa
        from scipy import signal as sps

        wav = sps.lfilter([1, -_HP["preemphasis"]], [1], waveform)
        stft = librosa.core.stft(
            wav, n_fft=_HP["n_fft"], hop_length=_HP["hop_length"],
            win_length=_HP["win_length"], window="hann", center=True)
        power = np.abs(stft) ** 2
        if self._mel_basis is None or self._mel_basis.shape[1] != power.shape[0]:
            # norm='slaney' is librosa's default since 0.8 (the original runs
            # 0.8.1); passing it explicitly keeps us version-independent.
            self._mel_basis = librosa.filters.mel(
                sr=sr, n_fft=_HP["n_fft"], n_mels=_HP["num_mels"],
                fmin=_HP["fmin"], fmax=_HP["fmax"], htk=False, norm="slaney")
        mel = self._mel_basis @ power                                    # (80, T)
        return (10.0 * np.log10(mel.T + 1e-5) - _HP["ref_db"]).astype(np.float32)

    # -------------------------------------------------------------- features
    @torch.no_grad()
    def extract_features(self, wav_path: str, pad: int = SILENCE_PAD) -> np.ndarray:
        """Returns (T, 256) float32 encoder features at 25 fps.

        Reproduces the original ``get_weget`` bit-exactly: 16 kHz mono, 0.4 s of
        silence on BOTH ends, then the wav2mfcc_v2 dB-mel front end. Because of
        the leading pad, feature index 0 sits 0.4 s before the first audio sample
        (i.e. feature index == audio frame index + 10).
        """
        import librosa

        waveform, sr = self._load_wav(wav_path)
        y = waveform.squeeze(0).numpy()
        if sr != _SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=_SAMPLE_RATE)
        if pad:
            y = np.concatenate([np.zeros(pad, np.float32), y,
                                np.zeros(pad, np.float32)])
        log_mel = self._log_mel_db(y, _SAMPLE_RATE)                      # (T_mel, 80)
        lengths = torch.tensor([log_mel.shape[0]], device=self.device)
        encoder_out, _ = self.model(
            torch.from_numpy(log_mel).unsqueeze(0).to(self.device), lengths)
        return encoder_out.squeeze(0).cpu().numpy().astype(np.float32)

    def windows_for_video(self, wav_path: str, num_frames: int,
                          fps: float = 25.0) -> np.ndarray:
        """Returns (n, WINDOW_T, 256) exactly as the original feeds DINet, and
        determines the output length ``n`` the same way.

        Two regimes (both verified against hook dumps of the original's DINet
        audio inputs):

        - **Video-limited** (``num_frames <= n_out_audio``): per-frame windows
          ``feats[floor(i*25/fps) : +T]`` (dump3, 30 fps, 152/160 bit-exact).
        - **Audio-limited** (video longer than the audio): the original
          RESAMPLES the usable feature span across its output -- hook dumps on
          two 60 fps clips (man_6/man_15, 0/160 mismatches each) show
          ``start(i) = floor(i * (n_feat-T) / (n_out-1))`` with
          ``n_out = ceil((n_feat-T+1) * fps / 25)`` (636 frames for a 284-feature
          audio @60 fps; 752 for 332 features). A constant floor(i*25/fps) rate
          drifts a full feature by mid-clip there and reads as mouth-timing
          jitter vs the original.
        """
        feats = self.extract_features(wav_path)
        if feats.shape[0] == 0:
            raise RuntimeError("no audio features extracted")
        n_feat = feats.shape[0]
        span = max(1, n_feat - WINDOW_T)
        n_out_audio = int(np.ceil((span + 1) * fps / FEATURE_FPS))
        if num_frames > n_out_audio:
            n = n_out_audio
            stride = span / float(n - 1)
        else:
            n = num_frames
            stride = FEATURE_FPS / fps
        return self.frame_windows(feats, num_frames=n, stride=stride)

    @staticmethod
    def frame_windows(features: np.ndarray, num_frames: Optional[int] = None,
                      T: int = WINDOW_T, offset: int = WINDOW_OFFSET,
                      stride: float = 1.0) -> np.ndarray:
        """Per-video-frame audio windows ``feats[i*stride + offset : +T]``.

        Edges are clamped (the original clamps its section index the same way).
        Returns (num_frames, T, 256).
        """
        n_feat, d = features.shape
        n = num_frames if num_frames is not None else n_feat
        out = np.empty((n, T, d), np.float32)
        last = max(0, n_feat - T)
        for i in range(n):
            # FLOOR, not round: verified against the original pipeline's DINet
            # audio-input dumps (hook, dump3): their window start for video
            # frame i is floor(i*stride). With round() the start wobbles +1
            # feature on ~1 of every 6 frames at 30 fps (stride 25/30 hits
            # exact .5 fractions), making the lip motion judder at ~5 Hz --
            # perceived as sudden lower-face "jumps" at fast phonemes.
            lo = int(np.floor(i * stride + offset))
            lo = max(0, min(last, lo))
            win = features[lo:lo + T]
            if win.shape[0] < T:            # tail-padded audio shorter than T
                win = np.concatenate([win, np.repeat(win[-1:], T - win.shape[0], axis=0)])
            out[i] = win
        return out

    def extract_for_video(self, wav_path: str, num_frames: int,
                          fps: float = 25.0) -> np.ndarray:
        """Deprecated: kept for call compatibility. Returns the per-video-frame
        feature (the first row of each DINet window)."""
        return self.windows_for_video(wav_path, num_frames, fps)[:, 0, :]

    @property
    def feature_fps(self) -> float:
        return FEATURE_FPS
