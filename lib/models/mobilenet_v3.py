import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from lib.utils.workspace import register
from lib.models.hrnet import ShapeSpec, ConvNormLayer

@register
class MobileNetV3(nn.Layer):
    def __init__(self,
                 mode="large",
                 return_idx=[0],
                 upsample=False,
                 width=None,
                 ):
        super().__init__()
        import paddleclas
        assert mode in ["large", "small"]
        self.mode = mode
        self.return_idx = return_idx
        self.upsample = upsample
        self.cfg = {
            "large": {
                "keys": ["blocks[2]", "blocks[5]", "blocks[11]", "blocks[14]"],
                "channels": [24, 40, 112, 160],
                "strides": [4, 8, 16, 32],
            },
            "small": {
                "keys": ["blocks[0]", "blocks[2]", "blocks[7]", "blocks[10]"],
                "channels": [16, 24, 48, 96],
                "strides": [4, 8, 16, 32],
            }
        }
        if mode == "large":
            self.model = paddleclas.MobileNetV3_large_x1_0(pretrained=True, return_stages=True)
        else:
            self.model = paddleclas.MobileNetV3_small_x1_0(pretrained=True, return_stages=True)
        
        if width is not None and upsample:
            input_ch = sum(self.cfg[self.mode]["channels"])
            self.final_conv = ConvNormLayer(
                input_ch,
                width,
                filter_size=3,
                stride=1,
                norm_type='bn',
                norm_groups=1,
            )
        else:
            self.final_conv = None
    
    def forward(self, x):
        x = x['image']
        out = self.model(x)
        out = [out[k] for k in self.cfg[self.mode]["keys"]]
        if self.upsample:
            x0_h, x0_w = out[0].shape[2:4]
            x1 = F.upsample(out[1], size=(x0_h, x0_w), mode='bilinear')
            x2 = F.upsample(out[2], size=(x0_h, x0_w), mode='bilinear')
            x3 = F.upsample(out[3], size=(x0_h, x0_w), mode='bilinear')
            x = paddle.concat([out[0], x1, x2, x3], 1)
            if self.final_conv is not None:
                x = self.final_conv(x)
            out = [x]
            self.return_idx = [0]
        
        res = [out[k] for k in self.return_idx]
        
        return res

    def out_shape(self):
        return [
            ShapeSpec(
                channels=self.cfg[self.mode]["channels"][idx], stride=self.cfg[self.mode]["strides"][idx])
            for idx in self.return_idx
        ]