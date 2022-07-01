"""
A lot of this stuff is ported from https://github.com/balbasty/nitorch
Specifically:
    - https://github.com/balbasty/nitorch/blob/master/nitorch/io/utils/indexing.py
    - https://github.com/balbasty/nitorch/blob/master/nitorch/io/volumes/mapping.py

TODO:
    [x] Propagate slicing to the affine matrix (I forgot to do it)
    [ ] Port `CatArray` as well (virtual concatenation of `SliceableArray`s)
    [ ] Option to handle subarrays that know how to do strided slicing
    [ ] Make it a standalone package?
"""
import copy
import math
from typing import List, Tuple, Optional, Sequence, Any
import numpy as np
from numpy.typing import ArrayLike
from . import indexing as idx

SlicerType = Tuple[int, slice, type(Ellipsis), type(None)]

try:
    import torch
    TensorLike = torch.Tensor
except ImportError:
    torch = None
    TensorLike = Any


class SliceableArray:
    """
    A wrapper around an array that can only be sliced using
    non-strided contiguous indices.

    This wrapper adds the ability to permute, add axes, drop axes, stride,
    track dimensions and orientation matrices, etc.
    """

    def __init__(self, array,
                 affine: Optional[ArrayLike] = None,
                 names: Optional[Sequence[str]] = None):
        """

        Parameters
        ----------
        array : object that implements property `shape` and
            method `__getitem__`, which will only be called
            with unstrided slices.
        affine : a voxel-to-world matrix
        names : a list of names for the input dimensions.
        """
        self._array = array
        if affine is not None:
            affine = np.asanyarray(affine)
        self._affine = affine
        if affine is not None and names is None:
           names = self._default_names()
        self._names = tuple(names) if names else None
        self._permutation = tuple(range(self._dim))
        self._slicer = idx.expand_index([Ellipsis], self._shape)

    def copy_from(self, other):
        self._array = other._array
        self._affine = other._affine
        self._names = other._names
        self._permutation = other._permutation
        self._slicer = other._slicer

    def _default_names(self):
        ndim = self._affine.shape[-1] - 1
        names = 'XYZ'[:ndim]
        if len(names) < self._dim:
            names += 'C'
        names += '?' * max(0, self._dim - len(names))
        return names

    def __len__(self):
        if self.shape:
            return self.shape[0]
        else:
            raise TypeError('Scalar objet has no len()')

    @property
    def _shape(self) -> Tuple[int]:
        """Original shape of the array"""
        # convert each element to make sure the tuple is clean
        # (no weird data types)
        return tuple(int(s) for s in self._array.shape)

    @property
    def _dim(self) -> int:
        """Orginal dimension of the array"""
        return len(self._shape)

    @property
    def shape(self) -> Tuple[int]:
        """Shape of the sliced array"""
        perm_shape = [self._shape[p] if p is not None else 1
                      for p in self._permutation]
        shp: List[int] = []
        for s, d in zip(self._slicer, perm_shape):
            if isinstance(s, slice):
                shp.append(idx.slice_length(s, d))
            elif s is None:
                shp.append(1)
            else:
                assert isinstance(s, int), f'wrong index type: {type(s)}'
        return tuple(shp)

    @property
    def size(self) -> int:
        """Number of elements in the sliced array"""
        return math.prod(self.shape)

    @property
    def dim(self) -> int:
        """Dimension of the sliced array"""
        return sum(isinstance(s, slice) or s is None for s in self._slicer)

    @property
    def spatial_shape(self) -> Tuple[int]:
        """Spatial shape (keeping only dimensions named X, Y or Z)"""
        if not self._names or not any(N in self._names for N in 'XYZ'):
            return tuple()
        else:
            shape = [s for s, n in zip(self.shape, self.names) if n in 'XYZ']
            return tuple(shape)

    @property
    def spatial_dim(self) -> int:
        """Dimension (keeping only dimensions named X, Y or Z)"""
        if not self._names or not any(N in self._names for N in 'XYZ'):
            return 0
        return sum(N in 'XYZ' for N in self.names)

    @property
    def spatial_names(self) -> Optional[Tuple[str]]:
        if not self._names:
            return None
        return tuple(N for N in self.names if N in 'XYZ')

    @property
    def _spatial_names(self) -> Optional[Tuple[str]]:
        if not self._names:
            return None
        return tuple(N for N in self._names if N in 'XYZ')

    @property
    def names(self) -> Optional[Tuple[str]]:
        if not self._names:
            return None
        return [self._names[i] if i is not None else '?'
                for i, s in zip(self._permutation, self._slicer)
                if not idx.is_droppedaxis(s)]

    @property
    def affine(self) -> Optional[ArrayLike]:
        if self._affine is None:
            return None
        spatial_indices = ['XYZ'.index(N) for N in self.spatial_names]
        spatial_indices = [*spatial_indices, -1]
        slicer = [0] * self._dim
        for p, s in zip(self._permutation, self._slicer):
            if p is not None:
                slicer[p] = s
        affine, _ = idx.affine_sub(self._affine, self._shape, tuple(slicer))
        affine = affine[:, spatial_indices]
        return affine

    def clone(self, **kwargs) -> "SliceableArray":
        """Make a copy of the Array"""
        new = copy.deepcopy(self)
        for k, v in kwargs.items():
            setattr(new, k, v)
        return new

    def permute(self, permutation: Sequence[int]) -> "SliceableArray":
        """Permute dimensions

        Parameters
        ----------
        permutation : list[int]
            Permutation indices
        """
        permutation = list(permutation)
        if len(permutation) != self.dim:
            raise ValueError('Permutation should be of length `dim`.')
        permutation = [self.dim + p if p < 0 else p
                       for p in permutation]
        if any(p < 0 or p >= self.dim for p in permutation):
            raise ValueError('Permutation indices should be in `(0, dim-1)`.')
        if len(set(permutation)) != len(permutation):
            raise ValueError('Permutation indices should be unique')

        # remove dropped axis
        _permutation = [p for p, s in zip(self._permutation, self._slicer)
                        if not idx.is_droppedaxis(s)]
        _slicer = [s for s in self._slicer if not idx.is_droppedaxis(s)]
        # apply permutation
        _permutation = [_permutation[p] for p in permutation]
        _slicer = [_slicer[p] for p in permutation]
        # put back dropped axis
        _permutation += [p for p, s in zip(self._permutation, self._slicer)
                         if idx.is_droppedaxis(s)]
        _slicer += [s for s in self._slicer if idx.is_droppedaxis(s)]

        kwargs = {'_permutation': _permutation, '_slicer': _slicer}
        return self.clone(**kwargs)

    def flip(self, *args, **kwargs) -> "SliceableArray":
        """Flip along one (or all) dimensions

        Parameters
        ----------
        dim : int or list[int], default=all
            Dimensions to flip
        """
        dims = None
        if args:
            dims, *args = args
        else:
            for key in ('dim', 'dims', 'axis', 'axes'):
                if key in kwargs:
                    dims = kwargs.pop(key)
                    break
        if args:
            raise ValueError(f'Too many positional arguments.')
        if kwargs:
            raise ValueError(f'Unknown (or superfluous) keywords '
                             f'{list(kwargs.keys())}')
        if dims is None:
            dims = list(range(self.dim))
        dims = make_list(dims)

        _slicer = []
        cnt = -1
        for s in self._slicer:
            if s is None or isinstance(s, slice):
                cnt += 1
            if cnt in dims and isinstance(s, slice):
                s = idx.invert_slice(s, self.shape[cnt])
            _slicer.append(s)
        kwargs = {'_slicer': _slicer}
        return self.clone(**kwargs)

    def flipud(self) -> "SliceableArray":
        """Flip the first (vertical) dimension."""
        return self.flip(0)

    def fliplr(self) -> "SliceableArray":
        """Flip the second (horizontal) dimension."""
        return self.flip(1)

    def slice(self, index: SlicerType) -> "SliceableArray":
        """Extract a sub-part of the array.

        Indices can only be slices, ellipses, integers or None.

        Parameters
        ----------
        index : tuple[slice or ellipsis or int or None]

        """
        index = idx.expand_index(index, self.shape)
        if any(isinstance(i, list) for i in index):
            raise ValueError('List indices not currently supported '
                             '(otherwise we enter advanced indexing '
                             'territory and it becomes too complicated).')

        # update permutation
        new_permutation = []
        old_permutation = list(self._permutation)
        old_slicer = list(self._slicer)
        new_slicer = list(index)
        while old_permutation and new_slicer:
            # get rid of dropped axes
            while old_slicer and idx.is_droppedaxis(old_slicer[0]):
                new_permutation.append(old_permutation.pop(0))
                old_slicer.pop(0)
            s = new_slicer.pop(0)
            if s is None:
                # add a new axis
                new_permutation.append(None)
            elif isinstance(s, slice):
                # preserve an existing axis
                new_permutation.append(old_permutation.pop(0))
                old_slicer.pop(0)
            else:
                # delete an axis
                if old_permutation[0] is None:
                    # drop a fake axis
                    old_permutation.pop(0)
                    old_slicer.pop(0)
                else:
                    # drop a real axis (so don't drop it really)
                    new_permutation.append(old_permutation.pop(0))
                    old_slicer.pop(0)
        # finish unrolling dropped axis
        while old_slicer and idx.is_droppedaxis(old_slicer[0]):
            new_permutation.append(old_permutation.pop(0))
            old_slicer.pop(0)
        _permutation = new_permutation

        # permuted full-sized shape
        perm_shape = [self._shape[d] for d in self._permutation
                      if d is not None]

        # compose slicers
        _slicer = idx.compose_index(self._slicer, index, perm_shape)

        kwargs = {'_slicer': _slicer, '_permutation': _permutation}
        return self.clone(**kwargs)

    def __getitem__(self, index: SlicerType) -> "SliceableArray":
        return self.slice(index)

    def get(self) -> ArrayLike:
        # compute permutation of original dimensions
        iperm = list(range(self._dim))
        for d, p in enumerate(self._permutation):
            if p is not None:
                iperm[p] = d

        # compute slicer of original dimensions
        slicer = tuple(self._slicer[p] for p in iperm)
        slicer = tuple(s if isinstance(s, slice) else slice(s, s+1)
                       for s in slicer if s is not None)

        # ensure slicer does not have negative strides
        perm_shape = [self._shape[d] for d in self._permutation
                      if d is not None]
        nostride_slicer = tuple(idx.slice_no_stride(s, d)
                                for s, d in zip(slicer, perm_shape))

        # extract chunk
        dat = self._array[nostride_slicer]

        # apply strides
        strides = tuple(slice(None, None, s.step) if s.step not in (1, None)
                        else slice(None) for s in slicer)
        if any(s != slice(None) for s in strides):
            dat = dat[strides]

        # permute original dimensions
        perm = [p for p in self._permutation if p is not None]
        dat = np.transpose(dat, perm)

        # add/drop axes
        slicer = tuple(None if s is None
                       else slice(None) if isinstance(s, slice)
                       else 0 for s in self._slicer)
        return dat[slicer]

    def __array__(self) -> ArrayLike:
        """Convert to numpy array"""
        return np.asanyarray(self.get())

    def __str__(self) -> str:
        args = [f'shape={self.shape}']
        if self.names:
            args += [f'names={self.names}']
        args = ', '.join(args)
        return f'{type(self).__name__}({args})'

    def __repr__(self) -> str:
        return self.__str__()


class ListWrapper:
    """Demo class to show how to wrap a container"""

    def __init__(self, dat):
        self.dat = dat

    @property
    def shape(self):
        return [len(self.dat)]

    def __getitem__(self, item):
        return self.dat[item]


class SliceableList(SliceableArray):
    def __init__(self, dat):
        super().__init__(ListWrapper(dat))

