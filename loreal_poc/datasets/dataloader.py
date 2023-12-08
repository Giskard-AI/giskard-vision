from torch.utils.data import DataLoader
import numpy as np


class LandmarksDataloader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.collate_fn = lambda items:  {'image': [data_item['image'] for data_item in items],
                                        'marks': np.stack([data_item['marks'] for data_item in items], axis=0),
                                        'file_id': [data_item['file_id'] for data_item in items]}