"""
Work in progress: on the fly registration

There's nothing much here yet.

I was building multilevel arrays that could be used for coarse to fine
registration. I was also trying to understand the coordinate system in
neuroglancer. There's still a lot to do.
"""

import numpy as np
import math
from .sliceable import SliceableArray
from . import indexing as idx


def default_affine_ng(shape, scale=1):
    """Default affine for NeuroGlancer"""
    ndim = len(shape)
    if not isinstance(scale, (list, tuple)):
        scale = [scale]
    scale = list(scale)
    scale += max(0, ndim - len(scale)) * scale[-1:]
    scale += [1.]
    aff = np.diag(scale)
    return aff


class MultiLevelArray(SliceableArray):
    """
    Multi-level array that keeps (roughly) the same field-of-view across
    levels when slicing.
    """

    def __init__(self, arrays, affine=None, names=None):
        pass

    def __new__(cls, arrays, affine=None, names=None):

        if not all(isinstance(a, SliceableArray) for a in arrays):

            if affine is None:
                if names is None:
                    shape = arrays[0].shape
                    names = 'XYZ'[:len(shape)]
                    names += '?' * max(0, len(shape) - len(names))
                spatial = [s for s, n in zip(shape, names) if n in 'XYZ']
                affine = default_affine_ng(spatial)
            if not isinstance(affine, (list, tuple)):
                # assumes neuroglancer coordinates
                affine = np.asanyarray(affine)
                affines = [affine]
                for _ in range(1, len(arrays)):
                    affine = np.copy(affine)
                    diag = np.copy(np.diag(affine))
                    affine -= np.diag(diag)
                    diag[:-1] *= 2
                    affine += np.diag(diag)
                    affines.append(affine)
                affine = affines

            arrays = [SliceableArray(arr, aff, names)
                      for arr, aff in zip(arrays, affine)]

        obj = super().__new__(cls)
        obj.copy_from(arrays[0])
        obj.levels = arrays
        return obj

    @property
    def nb_levels(self):
        return len(self.levels)

    def permute(self, permutation):
        levels = [lvl.permute(permutation) for lvl in self.levels]
        return type(self)(levels)

    def flip(self, *args, **kwargs):
        levels = [lvl.flip(*args, **kwargs) for lvl in self.levels]
        return type(self)(levels)

    def flipud(self):
        levels = [lvl.flipud() for lvl in self.levels]
        return type(self)(levels)

    def fliplr(self):
        levels = [lvl.fliplr() for lvl in self.levels]
        return type(self)(levels)

    def slice(self, index):

        index = tuple(idx.expand_index(index, self.shape))
        indices = [index]
        for _ in range(1, self.nb_levels):

            old_index = index
            index = []
            names = list(self.names)
            slicer = list(self._slicer)
            shape = list(self.shape)
            for i in old_index:
                if i is None:
                    index.append(None)
                    continue
                while isinstance(slicer[0], int):
                    slicer.pop(0)
                    names.pop(0)
                n = names.pop(0)
                s = shape.pop(0)
                if n in 'XYZ':
                    if isinstance(i, int):
                        i = idx.neg2pos(i, s)
                        i = int(math.floor(i / 2 - 1/4))
                    else:
                        assert isinstance(i, slice)
                        i = idx.neg2pos(i, s)
                        start = i.start
                        stop = i.stop
                        step = i.step
                        if step and step < 0:
                            if start is None:
                                start = s - 1
                            if stop is None:
                                stop = -1
                            start = int(math.ceil(start / 2 - 1/4))
                            stop = int(math.ceil(stop / 2 - 1/4))
                            if stop < 0:
                                stop = None
                        else:
                            if start is None:
                                start = 0
                            if stop is None:
                                stop = s
                            start = int(math.floor(start / 2 - 1/4))
                            stop = int(math.floor(stop / 2 - 1/4))
                        i = slice(start, stop, step)
                index.append(i)
            indices.append(tuple(index))

        levels = [lvl.slice(ind) for lvl, ind in zip(self.levels, indices)]
        return type(self)(levels)


class MultiLevelArrayKeepSize(MultiLevelArray):
    """
    Multi-level array that keeps (roughly) the same size, in voxels,
    across levels.
    """

    def slice(self, index):

        index = tuple(idx.expand_index(index, self.shape))
        indices = [index]
        for _ in range(1, self.nb_levels):
            old_index = index
            index = []
            names = list(self.names)
            shape = list(self.shape)
            slicer = list(self._slicer)
            for i in old_index:
                if i is None:
                    index.append(None)
                    continue
                while isinstance(slicer[0], int):
                    slicer.pop(0)
                    names.pop(0)
                s = shape.pop(0)
                n = names.pop(0)
                if n in 'XYZ':
                    if isinstance(i, int):
                        i = i * 2
                    else:
                        assert isinstance(i, slice)
                        i = idx.neg2pos(i, s)
                        start, stop, step = i.start, i.stop, i.step

                        step = step or 1
                        sign = (-1 if step < 0 else 1)
                        if step < 0:
                            start = s-1 if start is None else start
                            stop = -1 if stop is None else stop
                        else:
                            start = 0 if start is None else start
                            stop = s if stop is None else stop

                        # current scale
                        length, center = abs(stop - start), (start + stop) / 2
                        # next scale (take into account half voxel shift)
                        center = center/2 - 1/4
                        start = center - length * sign / 2
                        stop = center + length * sign / 2

                        if sign < 0:
                            start = int(math.ceil(start))
                            stop = int(math.ceil(stop))
                            start = min(start, s-1)
                            stop = max(stop, -1)
                            if stop < 0:
                                stop = None
                        else:
                            start = int(math.floor(start))
                            stop = int(math.floor(stop))
                            start = max(start, 0)
                            stop = min(stop, s)
                        i = slice(start, stop, step)
                index.append(i)
            indices.append(tuple(index))

        levels = [lvl.slice(ind) for lvl, ind in zip(self.levels, indices)]
        return type(self)(levels)
