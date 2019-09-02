import numpy as np
class Dataset:
    def __init__(self, xs, ys, shuffle=True):
        assert len(xs) == len(ys)
        self.xs, self.ys = xs, ys
    def __len__(self):
        return len(self.xs)
    def __getitem__(self, i):
        return self.xs[i], self.ys[i]
    def __iter__(self):
        if self.shuffle: idxs = np.random.permutation(len(self))
        else:       idxs = list(range(len(self)))
        for i in range(len(self)):
            yield self[idxs[i]]
    def iter_batches(self, bs=64, shuffle=True):
        if shuffle: idxs = np.random.permutation(len(self))
        else:       idxs = list(range(len(self)))
        batch_start = 0
        while batch_start<len(self):
            yield self[batch_start:batch_start+bs]
            batch_start += bs
            
    def dataloader(self, bs=64, shuffle=True):
        return Dataloader(self, bs, shuffle)
            
    

class Dataloader:
    def __init__(self, dataset, bs=64, shuffle=True):
        self.ds = dataset
        self.bs = bs
        self.shuffle = shuffle
        
    def __len__(self):
        return int(np.ceil(len(self.ds)/self.bs))
        
    def __iter__(self):
        if self.shuffle: idxs = np.random.permutation(len(self.ds))
        else:       idxs = list(range(len(self.ds)))
        batch_start = 0
        while batch_start<len(self.ds):
            bx, by = self.ds[idxs[batch_start:batch_start+self.bs]]
            yield list(bx), by
            batch_start += self.bs
        
class Databunch:
    def __init__(self, train_ds, valid_ds, test_ds=None, bs=64):
        self.train_ds, self.valid_ds, self.test_ds = train_ds, valid_ds, test_ds
        self.train_dl = self.train_ds.dataloader(bs, shuffle=True)
        self.valid_dl = self.valid_ds.dataloader(bs, shuffle=False)
        if self.test_ds:
            self.test_dl = self.test_ds.iter_batches(bs, shuffle=False)