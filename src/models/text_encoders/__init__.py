"""Text encoder reference mapping (architecture -> open_clip config params).

All text encoders are implemented by open_clip's TextTransformer.
This package documents the text encoder configs from Table 2 of the paper.

Teacher/Student      | width | heads | layers | Params
---------------------|-------|-------|--------|-------
ViT-L/14 text        |  768  |  12   |   12   | 85.1M
ViT-B/16, RN50/101   |  512  |   8   |   12   | 37.8M
ViT-T/16, MobileViT  |  384  |   6   |   12   | 21.3M

These are encoded in the model JSON configs under src/open_clip/model_configs/.
"""
