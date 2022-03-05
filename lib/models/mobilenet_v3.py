import paddle
import paddle.nn as nn

from lib.utils.workspace import register
from lib.models.hrnet import ShapeSpec

@register
class MobileNetV3(nn.Layer):
    def __init__(self,
                 mode="large",
                 return_idx=[0],
                 ):
        super().__init__()
        import paddleclas
        assert mode in ["large", "small"]
        self.mode = mode
        self.return_idx = return_idx
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
    
    def forward(self, x):
        x = x['image']
        out = self.model(x)
        out = [out[k] for k in self.cfg[self.mode]["keys"]]
        res = [out[k] for k in self.return_idx]
        return res

    def out_shape(self):
        return [
            ShapeSpec(
                channels=self.cfg[self.mode]["channels"][idx], stride=self.cfg[self.mode]["strides"][idx])
            for idx in self.return_idx
        ]