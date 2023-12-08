import numpy as np

class LandmarksDataloader():
    def __init__(self, dataset, batch_size=1, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size

        self.samples_indices = [idx for idx in range(len(self.dataset))]
        print(self.samples_indices)
        if shuffle:
            np.random.shuffle(self.samples_indices)
    
    def __iter__(self):
        while len(self.samples_indices) > 0:
            indices = self.samples_indices[:self.batch_size]
            self.samples_indices = self.samples_indices[self.batch_size:]
        
            batch_items = [self.dataset[idx] for idx in indices]
            yield self.collate_fn(batch_items)
    
    def collate_fn(self, items):
        return {'image': [data_item['image'] for data_item in items],
                                    'marks': np.stack([data_item['marks'] for data_item in items], axis=0),
                                    'file_id': [data_item['file_id'] for data_item in items]}