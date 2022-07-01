import json
from dandiio.array import DANDIArrayReader
from dandiio.sliceable import SliceableArray
from dandiio.registration import MultiLevelArrayKeepSize
from dandi_webserver import make_spec
import matplotlib.pyplot as plt
import numpy as np


class DANDIArrayWrapper(DANDIArrayReader):
    """Implement dat[x0:x1, y0:y1, z0:z1] in DANDIArrayReader"""
    def __getitem__(self, item):
        sx, sy, sz = item
        x0, x1 = sx.start, sx.stop
        y0, y1 = sy.start, sy.stop
        z0, z1 = sz.start, sz.stop
        return self.read_chunk(x0, x1, y0, y1, z0, z1)


class DANDIArray(SliceableArray):
    """Sliceable DANDIArray"""

    def __init__(self, spec, level=1):
        if isinstance(spec, DANDIArrayWrapper):
            subarray = spec
        elif isinstance(spec, DANDIArrayReader):
            subarray = DANDIArrayWrapper(spec)
        else:
            subarray = DANDIArrayReader(spec, level)
        super().__init__(subarray)


with open('dandiset108.json', 'rt') as f:
    config = json.load(f)

spec1 = make_spec(config, 'MITU01', '123', 'LEC')
spec2 = make_spec(config, 'MITU01', '124', 'LEC')
arrays1 = {l: DANDIArray(spec1, level=2**l) for l in range(6)}
arrays2 = {l: DANDIArray(spec2, level=2**l) for l in range(6)}

array1 = MultiLevelArrayKeepSize([
    DANDIArrayReader(spec1, level=2**l) for l in range(6)
])
array2 = MultiLevelArrayKeepSize([
    DANDIArrayReader(spec2, level=2**l) for l in range(6)
])


foo = 0

# centers = [s//2 for s in arrays[5].shape]
# slicer = tuple(slice(c-128, c+128) for c in centers)
plt.subplot(2, 1, 1)
dat = np.asarray(arrays1[5])
dat = np.sqrt(np.sum(dat*dat, 0))
plt.imshow(dat)
plt.subplot(2, 1, 2)
dat = np.asarray(arrays2[5])
dat = np.sqrt(np.sum(dat*dat, 0))
plt.imshow(dat)
plt.show()

foo = 0

