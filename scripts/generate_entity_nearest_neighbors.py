import faiss                   # make faiss available
import numpy as np
import time
import torch

# entities = torch.load("/private/home/belindali/BLINK/models/all_entities_large.t7")
# xb = entities.numpy()
# entities = np.save("/private/home/belindali/BLINK/models/all_entities_large.npy", xb)

# ~26s
xb = np.load("/private/home/belindali/BLINK/models/all_entities_large.npy")  # ~26s

d = entities.shape[1]            # dimension
np.random.seed(1234)             # make reproducible

# ~23s
index = faiss.IndexFlatL2(d)   # build the index
print(index.is_trained)
index.add(xb)                  # add vectors to the index ()
print(index.ntotal)

# TRY IVFFLAT
# quantizer = faiss.IndexFlatL2(d)  # the other index
nlist = 100
index_ivf = faiss.IndexIVFFlat(index, d, nlist)
assert not index_ivf.is_trained
index_ivf.train(xb)  # 15s
assert index_ivf.is_trained
index_ivf.add(xb)  # 41s
print(index_ivf.ntotal)
index_ivf.nprobe = 10

nq = 1                         # nb of queries
xq = np.random.random((nq, d)).astype('float32')

k = 10                         # we want to see 10 nearest neighbors
D, I = index_ivf.search(xq, k)  # 2.5-2.7s for pure index, partitioned: (0.13s for 1/100 partitions, 0.53s for 10/100 partitions)
