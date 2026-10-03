"""DINetV1: the DINet variant used by HeyGem's ``dinet_v1_20240131.pth``.

Structure verified against ``repr()`` of the compiled original (see HANDOFF §5):
every block type, kernel and stride below is what ``DINet.DINetV1(3, 3, 256)``
reports on the server, and the constructed state_dict matches the checkpoint's
416 tensors exactly with no shape mismatch.

    source_in_conv : ModuleList of 3 x [SameBlock2d, DownBlock2d]
                     3ch @H -> 64 -> 128 @H/2 -> 256 @H/4 -> 512 @H/8
    ref_in_conv    : same (single reference frame, NOT masked: input_mask_re==1)
    trans_conv     : 11 blocks, Down at idx 3/6/8/10 -> global_avg2d -> img_para(256)
                     [Same(1024,256,k3), Same(k11), Same(k11), Down(k3),
                      Same(k7), Same(k7), Down(k3),
                      Same(k5), Down(k5), Same(k3), Down(k3)]
    audio_encoder  : SameBlock1d(256,256,k5) Res Down Res Down Same(k3)
                     (B,256,T) -> global_avg1d -> audio_para(256)
    appearance_conv_list : [[Same(512,512,k5) + 4x Res], [4x Res]]
    adaAT          : AdaAT(para_ch=512, feature_ch=512)
    out_conv       : [[Same(1024,512,k3), UpBlock2d(512,512,k3), Res(512,512)],
                      [UpBlock2d(768,256,k3), Res(256,256)],
                      [UpBlock2d(384,128,k3), Res(128,128), Conv2d(128,3,k7), Sigmoid]]

Because every block type here has the same parameter shapes as the alternatives we
used to build (SameBlock2d(64,128,k3) vs DownBlock2d(64,128,k3), SameBlock2d vs
UpBlock2d), a strict ``load_state_dict`` PASSES EVEN WHEN THE TYPES ARE WRONG --
only the strides/upsampling differ. That silently produced a network whose
bottleneck was H/4 instead of H/8 and whose decoder convolutions ran at the wrong
resolution, which is why the render came out soft and audio barely mattered.

Skip concatenations are resolution-matched by construction:
    out_conv[1] input = 512 @H/4 (up) + 256 @H/4 (s1) = 768
    out_conv[2] input = 256 @H/2 (up) + 128 @H/2 (s0) = 384
"""

import torch
from torch import nn

from .blocks import (
    AdaAT,
    DownBlock1d,
    DownBlock2d,
    ResBlock1d,
    ResBlock2d,
    SameBlock1d,
    SameBlock2d,
    UpBlock2d,
)


class DINetHD(nn.Module):
    def __init__(self, source_channel=3, ref_channel=3, audio_channel=256):
        super().__init__()
        # ---- source / reference image encoders: 3 stages, each Sequential(a, b)
        self.source_in_conv = nn.Sequential(
            nn.Sequential(
                SameBlock2d(source_channel, 64, kernel_size=7, padding=3),
                DownBlock2d(64, 128, kernel_size=3, padding=1),
            ),
            nn.Sequential(
                SameBlock2d(128, 128, kernel_size=3, padding=1),
                DownBlock2d(128, 256, kernel_size=3, padding=1),
            ),
            nn.Sequential(
                SameBlock2d(256, 256, kernel_size=3, padding=1),
                DownBlock2d(256, 512, kernel_size=3, padding=1),
            ),
        )
        self.ref_in_conv = nn.Sequential(
            nn.Sequential(
                SameBlock2d(ref_channel, 64, kernel_size=7, padding=3),
                DownBlock2d(64, 128, kernel_size=3, padding=1),
            ),
            nn.Sequential(
                SameBlock2d(128, 128, kernel_size=3, padding=1),
                DownBlock2d(128, 256, kernel_size=3, padding=1),
            ),
            nn.Sequential(
                SameBlock2d(256, 256, kernel_size=3, padding=1),
                DownBlock2d(256, 512, kernel_size=3, padding=1),
            ),
        )
        # ---- alignment (transformation parameter) encoder
        self.trans_conv = nn.Sequential(
            SameBlock2d(1024, 256, kernel_size=3, padding=1),
            SameBlock2d(256, 256, kernel_size=11, padding=5),
            SameBlock2d(256, 256, kernel_size=11, padding=5),
            DownBlock2d(256, 256, kernel_size=3, padding=1),
            SameBlock2d(256, 256, kernel_size=7, padding=3),
            SameBlock2d(256, 256, kernel_size=7, padding=3),
            DownBlock2d(256, 256, kernel_size=3, padding=1),
            SameBlock2d(256, 256, kernel_size=5, padding=2),
            DownBlock2d(256, 256, kernel_size=5, padding=2),
            SameBlock2d(256, 256, kernel_size=3, padding=1),
            DownBlock2d(256, 256, kernel_size=3, padding=1),
        )
        # ---- audio encoder (256-dim WeNet features, 256 channels wide)
        self.audio_encoder = nn.Sequential(
            SameBlock1d(audio_channel, 256, kernel_size=5, padding=2),
            ResBlock1d(256, 256, 3, 1),
            DownBlock1d(256, 256, 3, 1),
            ResBlock1d(256, 256, 3, 1),
            DownBlock1d(256, 256, 3, 1),
            SameBlock1d(256, 256, kernel_size=3, padding=1),
        )
        # ---- appearance (reference) feature transformation
        self.appearance_conv_list = nn.ModuleList([
            nn.Sequential(
                SameBlock2d(512, 512, kernel_size=5, padding=2),
                ResBlock2d(512, 512, 3, 1),
                ResBlock2d(512, 512, 3, 1),
                ResBlock2d(512, 512, 3, 1),
                ResBlock2d(512, 512, 3, 1),
            ),
            nn.Sequential(
                ResBlock2d(512, 512, 3, 1),
                ResBlock2d(512, 512, 3, 1),
                ResBlock2d(512, 512, 3, 1),
                ResBlock2d(512, 512, 3, 1),
            ),
        ])
        self.adaAT = AdaAT(512, 512)
        # ---- decoder with skip connections
        # Block types verified against repr() of the compiled DINetV1:
        #   [0] Same(1024,512) -> UpBlock2d(512,512) -> Res(512,512)
        #   [1] UpBlock2d(768,256) -> Res(256,256)
        #   [2] UpBlock2d(384,128) -> Res(128,128) -> Conv2d(128,3,k7) -> Sigmoid
        # UpBlock2d upsamples FIRST then convolves, so no explicit interpolate is
        # needed in forward. With three DownBlocks in each encoder the bottleneck
        # is H/8, and the three UpBlocks bring it back to H.
        self.out_conv = nn.Sequential(
            nn.Sequential(
                SameBlock2d(1024, 512, kernel_size=3, padding=1),
                UpBlock2d(512, 512, kernel_size=3, padding=1),
                ResBlock2d(512, 512, 3, 1),
            ),
            nn.Sequential(
                UpBlock2d(768, 256, kernel_size=3, padding=1),
                ResBlock2d(256, 256, 3, 1),
            ),
            nn.Sequential(
                UpBlock2d(384, 128, kernel_size=3, padding=1),
                ResBlock2d(128, 128, 3, 1),
                nn.Conv2d(128, 3, kernel_size=(7, 7), padding=(3, 3)),
                nn.Sigmoid(),
            ),
        )
        self.global_avg2d = nn.AdaptiveAvgPool2d(1)
        self.global_avg1d = nn.AdaptiveAvgPool1d(1)

    def forward(self, source_img, ref_img, audio_feature):
        """
        Args:
            source_img   : (B, 3, H, W) crop with the mouth region zeroed out.
            ref_img      : (B, 3, H, W) unmasked reference crop (single frame).
            audio_feature: (B, 256, T) WeNet encoder features (window of T=5).
        Returns:
            (B, 3, H, W) rendered crop.
        """
        # source encoder, keeping skip features (3 DownBlocks -> H/8 bottleneck)
        s0 = self.source_in_conv[0](source_img)          # 128ch @ H/2
        s1 = self.source_in_conv[1](s0)                  # 256ch @ H/4
        source_feat = self.source_in_conv[2](s1)         # 512ch @ H/8
        # reference encoder
        ref_feat = self.ref_in_conv(ref_img)             # 512ch @ H/8
        # transformation parameters
        img_para = self.trans_conv(torch.cat([source_feat, ref_feat], 1))
        img_para = self.global_avg2d(img_para).squeeze(-1).squeeze(-1)   # (B,256)
        audio_para = self.audio_encoder(audio_feature)
        audio_para = self.global_avg1d(audio_para).squeeze(-1)           # (B,256)
        trans_para = torch.cat([img_para, audio_para], 1)                # (B,512)
        # deform reference appearance
        ref_trans = self.appearance_conv_list[0](ref_feat)
        ref_trans = self.adaAT(ref_trans, trans_para)
        ref_trans = self.appearance_conv_list[1](ref_trans)
        # decoder with skips; each stage's leading UpBlock2d does the x2 upsample
        x = self.out_conv[0](torch.cat([source_feat, ref_trans], 1))     # 512 @ H/4
        x = self.out_conv[1](torch.cat([x, s1], 1))                      # 256 @ H/2
        return self.out_conv[2](torch.cat([x, s0], 1))                   # 3 @ H (sigmoid)


def load_dinet_hd(checkpoint_path: str, device: str = "cpu") -> DINetHD:
    """Load HeyGem's dinet checkpoint into DINetHD (strict)."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["face_G"] if "face_G" in checkpoint else checkpoint
    model = DINetHD()
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model
