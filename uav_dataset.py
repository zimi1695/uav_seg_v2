from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

@DATASETS.register_module()
class UAVDataset(BaseSegDataset):
    METAINFO = dict(
        classes=('Ignore','Background','Building','Road',
                 'Water','Barren','Vegetation','Agricultural','Vehicle'),
        palette=[
            [0,0,0],[200,200,200],[255,120,0],[128,64,128],
            [0,100,255],[192,192,128],[0,200,0],[255,200,0],[255,0,0]
        ]
    )
    def __init__(self, img_suffix='.png', seg_map_suffix='.png',
                 ignore_index=0, **kwargs):
        super().__init__(img_suffix=img_suffix,
                         seg_map_suffix=seg_map_suffix,
                         ignore_index=ignore_index, **kwargs)