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
import itertools
import math
from typing import List, Tuple, Optional, Sequence, Any
import numpy as np
from numpy.typing import ArrayLike

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
        self._slicer = expand_index([Ellipsis], self._shape)

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
            raise TypeError('Scalar object has no len()')

    @property
    def _shape(self) -> Tuple[int]:
        """Original shape of the array"""
        # convert each element to make sure the tuple is clean
        # (no weird data types)
        return tuple(int(s) for s in self._array.shape)

    @property
    def _dim(self) -> int:
        """Original dimension of the array"""
        return len(self._shape)

    @property
    def shape(self) -> Tuple[int]:
        """Shape of the sliced array"""
        perm_shape = [self._shape[p] if p is not None else 1
                      for p in self._permutation]
        shp: List[int] = []
        for s, d in zip(self._slicer, perm_shape):
            if isinstance(s, slice):
                shp.append(slice_length(s, d))
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
                if not is_droppedaxis(s)]

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
        affine, _ = affine_sub(self._affine, self._shape, tuple(slicer))
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
                        if not is_droppedaxis(s)]
        _slicer = [s for s in self._slicer if not is_droppedaxis(s)]
        # apply permutation
        _permutation = [_permutation[p] for p in permutation]
        _slicer = [_slicer[p] for p in permutation]
        # put back dropped axis
        _permutation += [p for p, s in zip(self._permutation, self._slicer)
                         if is_droppedaxis(s)]
        _slicer += [s for s in self._slicer if is_droppedaxis(s)]

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
                s = invert_slice(s, self.shape[cnt])
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
        index = expand_index(index, self.shape)
        if any(isinstance(idx, list) for idx in index):
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
            while old_slicer and is_droppedaxis(old_slicer[0]):
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
        while old_slicer and is_droppedaxis(old_slicer[0]):
            new_permutation.append(old_permutation.pop(0))
            old_slicer.pop(0)
        _permutation = new_permutation

        # permuted full-sized shape
        perm_shape = [self._shape[d] for d in self._permutation
                      if d is not None]

        # compose slicers
        _slicer = compose_index(self._slicer, index, perm_shape)

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
        nostride_slicer = tuple(slice_no_stride(s, d)
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


def make_list(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def affine_sub(affine, shape, indices):
    """Update an affine matrix according to a sub-indexing of the lattice.

    Notes
    -----
    .. Only sub-indexing that *keep an homogeneous voxel size* are allowed.
       Therefore, indices must be `None` or of type `int`, `slice`, `ellipsis`.

    Parameters
    ----------
    affine : (..., ndim_out[+1], ndim_in+1) tensor
        Input affine matrix.
    shape : (ndim_in,) sequence[int]
        Input shape.
    indices : tuple[slice or ellipsis]
        Subscripting indices.

    Returns
    -------
    affine : (..., ndim_out[+1], ndim_new+1) tensor
        Updated affine matrix.
    shape : (ndim_new,) tuple[int]
        Updated shape.

    """
    def is_int(elem):
        if torch and torch.is_tensor(elem):
            return elem.dtype in (torch.int32, torch.int64)
        elif np and isinstance(elem, np.ndarray):
            return elem.dtype in (np.int32, np.int64) and not elem.shape
        elif isinstance(elem, int):
            return True
        else:
            return False

    def to_int(elem):
        if torch and torch.is_tensor(elem):
            return elem.item()
        else:
            assert isinstance(elem, int)
            return elem

    # check types
    nb_dim = affine.shape[-1] - 1
    if len(shape) != nb_dim:
        raise ValueError(f'Expected shape of length {nb_dim}. Got {len(shape)}')
    if not isinstance(indices, tuple):
        raise TypeError('Indices should be a tuple.')
    indices = list(indices)

    # compute the number of input dimension that correspond to each index
    #   > slice index one dimension but eliipses index multiple dimension
    #     and their number must be computed.
    nb_dims_in = []
    ind_ellipsis = None
    for n_ind, ind in enumerate(indices):
        if isinstance(ind, slice):
            nb_dims_in.append(1)
        elif ind is Ellipsis:
            if ind_ellipsis is not None:
                raise ValueError('Cannot have more than one ellipsis.')
            ind_ellipsis = n_ind
            nb_dims_in.append(-1)
        elif is_int(ind):
            nb_dims_in.append(1)
        elif ind is None:
            nb_dims_in.append(0)
        else:
            raise TypeError('Indices should be None, integers, slices or '
                            'ellipses. Got {}.'.format(type(ind)))
    nb_known_dims = sum(nb_dims for nb_dims in nb_dims_in if nb_dims > 0)
    if ind_ellipsis is not None:
        nb_dims_in[ind_ellipsis] = max(0, nb_dim - nb_known_dims)

    # transform each index into a slice
    # note that we don't need to know "stop" to update the affine matrix
    nb_ind = 0
    indices0 = indices
    indices = []
    for d, ind in enumerate(indices0):
        if isinstance(ind, slice):
            start = ind.start
            step = ind.step
            step = 1 if step is None else step
            start = 0 if (start is None and step > 0) else \
                    shape[nb_ind] - 1 if (start is None and step < 0) else \
                    shape[nb_ind] + start if start < 0 else \
                    start
            indices.append(slice(start, None, step))
            nb_ind += 1
        elif ind is Ellipsis:
            for dd in range(nb_ind, nb_ind + nb_dims_in[d]):
                start = 0
                step = 1
                indices.append(slice(start, None, step))
                nb_ind += 1
        elif is_int(ind):
            indices.append(to_int(ind))
        elif ind is None:
            assert (ind is None), "Strange index of type {}".format(type(ind))
            indices.append(None)

    # Extract shift and scale in each dimension
    shifts = []
    scales = []
    slicer = []
    shape_out = []
    for d, ind in enumerate(indices):
        # translation + scale
        if isinstance(ind, slice):
            shifts.append(ind.start)
            scales.append(ind.step)
            shape_out.append(shape[d] // abs(ind.step))
            slicer.append(slice(None))
        elif isinstance(ind, int):
            scales.append(0)
            shifts.append(ind)
            slicer.append(0)
        else:
            slicer.append(None)
            assert (ind is None), "Strange index of type {}".format(type(ind))

    # build voxel-to-voxel transformation matrix
    lin = np.diag(np.asarray(scales))
    if any(not isinstance(s, slice) for s in slicer):
        # drop/add columns
        lin = [l[..., -1] for l in lin]
        zero = np.zeros(len(shifts))
        new_lin = []
        for s in slicer:
            if isinstance(s, slice):
                col, *lin = lin
                new_lin.append(col)
            elif isinstance(s, int):
                col, *lin = lin
            elif s is None:
                new_lin.append(zero)
        lin = np.stack(new_lin, axis=-1) if new_lin else []
    trl = np.asarray(shifts)[:, None]
    trf = np.concatenate((lin, trl), axis=1) if len(lin) else trl
    trf = np.concatenate((trf, np.zeros([1, trf.shape[1]])), axis=0)
    trf[-1, -1] = 1

    # compose
    affine = affine @ trf
    return affine, tuple(shape_out)


def cumprod(sequence, reverse=False, exclusive=False):
    """Perform the cumulative product of a sequence of elements.

    Parameters
    ----------
    sequence : any object that implements `__iter__`
        Sequence of elements for which the `__mul__` operator is defined.
    reverse : bool, default=False
        Compute cumulative product from right-to-left:
        `cumprod([a, b, c], reverse=True) -> [a*b*c, b*c, c]`
    exclusive : bool, default=False
        Exclude self from the cumulative product:
        `cumprod([a, b, c], exclusive=True) -> [1, a, a*b]`

    Returns
    -------
    product : list
        Product of the elements in the sequence.

    """
    if reverse:
        sequence = reversed(sequence)
    accumulate = None
    seq = [1] if exclusive else []
    for elem in sequence:
        if accumulate is None:
            accumulate = elem
        else:
            accumulate = accumulate * elem
        seq.append(accumulate)
    if exclusive:
        seq = seq[:-1]
    if reverse:
        seq = list(reversed(seq))
    return seq


class oob_slice:
    """Describe an out-of-bound slice (with output length 0)"""
    newaxis = False  # True if it is a slice into a virtual axis

    def __init__(self, newaxis=False):
        self.newaxis = newaxis

    def __repr__(self):
        if self.newaxis:
            return 'oob_slice(newaxis=True)'
        else:
            return 'oob_slice'

    __str__ = __repr__


def is_newaxis(slicer):
    """Return True if the index represents a new axis"""
    return slicer is None or (isinstance(slicer, oob_slice) and slicer.newaxis)


def is_droppedaxis(slicer):
    """Return True if the index represents a dropped axis"""
    return isinstance(slicer, int)


def is_sliceaxis(slicer):
    """Return True if the index is a slice"""
    return isinstance(slicer, (slice, oob_slice))


def neg2pos(index, shape):
    """Make a negative index (that counts from then end) positive

    .. warning:: this function should be called only once -- always on
        user inputs -- otherwise we risk transforming back stuff that
        should be kept negative.
        Ex: neg2pos(-5, 3) = -2                              # correct
            neg2pos(neg2pos(-5, 3), 3) = neg2pos(-2, 3) = 1  # wrong

    Parameters
    ----------
    index : index_like or sequence[index_like]
    shape : int or sequence[int]

    Returns
    -------
    index : index_like or tuple[index_like]

    Raises
    ------
    ValueError
        * if (en element of) shape is negative
        * if the lengths of index and shape are not consistent
    TypeError
        * if index is not an index_like or sequence[index_like]
        * if shape is not an int or sequence[int]
    """

    accepted_types = (slice, int, oob_slice, type(None), type(Ellipsis))
    if not isinstance(index, accepted_types):
        # add `None`s to shape to match new axes
        shape0 = shape
        shape = []
        for d, idx in enumerate(index):
            if idx is None:
                shape.append(None)
            else:
                shp, *shape0 = shape0
                shape.append(shp)
        index = list(index)
        if len(shape0) > 0 or len(index) != len(shape):
            raise ValueError('shape and index vectors not consistent.')
        # recursive call
        return tuple(neg2pos(idx, shp) for idx, shp in zip(index, shape))

    # sanity checks
    try:
        shape0 = shape
        shape = int(shape0)
        if shape != shape0:
            raise TypeError('Shape should be an integer')
    except TypeError:
        raise TypeError('Shape should be an integer')
    if shape < 0:
        raise ValueError('Shape should be a nonnegative integer')

    # deal with accepted types
    if isinstance(index, slice):
        return slice(neg2pos(index.start, shape),
                     neg2pos(index.stop, shape),
                     index.step)
    elif isinstance(index, int):
        if index is not None and index < 0:
            index = shape + index
        return index
    elif isinstance(index, (oob_slice, type(None), type(Ellipsis))):
        return index
    raise TypeError('Index should be an int, slice, Ellipsis or None')


def is_fullslice(index, shape, do_neg2pos=True):
    """Check if an index spans the full dimension.

    An index spans the full dimension if:
    * it is a new axis or a oob slice into a new axis
    * it is a slice equivalent to `:` or `::-1`
    * it is `0` along a singleton dimension

    Parameters
    ----------
    index : index_like or sequence[index_like]
        Index should have been expanded beforehand.
        That is, index_like includes (slice, int, None)
    shape : int or sequence[int]
    neg2pos : bool, default=True

    Returns
    -------
    bool or tuple[bool]

    """
    if index is None:
        return True
    elif isinstance(index, slice):
        index = simplify_slice(index, shape, do_neg2pos=do_neg2pos)
        return (index.start is None and index.stop is None
                and index.step in (None, 1, -1))
    elif isinstance(index, int):
        if do_neg2pos:
            index = neg2pos(index, shape)
        return index == 0 and shape == 1
    elif isinstance(index, oob_slice):
        return oob_slice.newaxis
    else:
        # expand
        index = expand_index(index, shape)
        # add `None`s to shape to match new axes
        shape0 = shape
        shape = []
        for d, idx in enumerate(index):
            if idx is None:
                shape.append(None)
            else:
                shp, *shape0 = shape0
                shape.append(shp)
        return tuple(is_fullslice(idx, shp) for idx, shp in zip(index, shape))


def slice_length(index, shape, do_neg2pos=True):
    """Compute the effective length (output number of elements) of a slice.

    ..warning:: `neg2pos` should *not* have been called on `index` before
        unless `do_neg2pos is False`.

    Parameters
    ----------
    index : slice
    shape : int
    do_neg2pos : bool, default=True

    Returns
    -------
    length : int

    """

    def sign(x):
        return 1 if x > 0 else -1 if x < 0 else 0

    if do_neg2pos:
        index = neg2pos(index, shape)
    start = index.start
    stop = index.stop

    # explicit step
    step = index.step
    step = 1 if step is None else step

    if step < 0:
        if stop is None or stop < 0:
            stop = -1
        if start is None or start >= shape:
            start = shape - 1
    else:
        if stop is None or stop > shape:
            stop = shape
        if start is None or start < 0:
            start = 0
    return max(1 + (stop - start - sign(step)) // step, 0)


def simplify_slice(index, shape, do_neg2pos=True):
    """Replace start/stop/step by `None`s when it is equivalent.

    ..warning:: `neg2pos` should *not* have been called on `index` before
        unless `do_neg2pos is False`.

    Parameters
    ----------
    index : slice
    shape : int

    Returns
    -------
    slice

    """

    length = slice_length(index, shape)
    if do_neg2pos:
        index = neg2pos(index, shape)
    start = index.start

    step = index.step or 1
    if step < 0:
        if start is None or start >= shape - 1:
            start = shape - 1
        stop = start + length * step
        if stop >= start:
            return oob_slice()
        if stop < 0:
            stop = None
        if start >= shape - 1:
            start = None
    else:
        if start is None or start <= 0:
            start = 0
        stop = start + length * step
        if stop <= start:
            return oob_slice()
        if stop >= shape:
            stop = None
        if start <= 0:
            start = None
        if step == 1:
            step = None

    return slice(start, stop, step)


def invert_slice(index, shape, do_neg2pos=True):
    """Compute a slice that is equivalent but with inverse stride.

    Parameters
    ----------
    index : slice
    shape : int
    do_neg2pos : bool, default=True

    Returns
    -------
    inv_index : slice
        with `inv_index.step == -index.step`

    """

    def sign(x):
        return 1 if x > 0 else -1 if x < 0 else 0

    start, step, length = slice_navigator(index, shape, do_neg2pos)

    # invert
    #   new start: last reachable element
    #   new step: negative step
    #   new stop: value just before/after last reachable element
    start = start + (length - 1) * step
    step = -step
    stop = start + (length - 1) * step + sign(step)  # value
    return simplify_slice(slice(start, stop, step), shape, do_neg2pos=False)


def slice_no_stride(index, shape, do_neg2pos=True):
    """Return a slice with the same coverage (min/max indices) but
    without strides

    Parameters
    ----------
    index : slice
    shape : int
    do_neg2pos : bool, default=True

    Returns
    -------
    nostride_index : slice
        with `nostride_index.step == 1`

    """
    if index.step is not None and index.step < 0:
        index = invert_slice(index, shape, do_neg2pos)
    if index.step not in (1, None):
        start, step, length = slice_navigator(index, shape, do_neg2pos)
        stop = start + (length - 1) * step + 1
        index = simplify_slice(slice(start, stop, 1), shape, do_neg2pos=False)
    return index


def slice_navigator(index, shape, do_neg2pos=True):
    """Return explicit start and step values from a slice

    Parameters
    ----------
    index : slice
    shape : int
    do_neg2pos : bool, default=False

    Returns
    -------
    start : int
        First index
    step : int
        Step between elements of the slice
    length : int > 0
        Length of the slice

    """
    length = slice_length(index, shape, do_neg2pos)
    index = simplify_slice(index, shape, do_neg2pos)
    start = index.start
    step = index.step or 1

    # compute explicit start
    if step < 0:
        if start is None:
            start = shape - 1
    else:
        if start is None:
            start = 0

    return start, step, length


def is_slice_equivalent(index1, index2, shape, same_sign=True, do_neg2pos=True):
    """Check that two slices describe the same chunk of data

    Parameters
    ----------
    index1 : slice
    index2 : slice
    shape : int
    same_sign : bool, default=True
        If True, requires that the steps have the same sign
    do_neg2pos : bool, default=True

    Returns
    -------
    bool

    """
    if do_neg2pos:
        index1 = neg2pos(index1, shape)
        index2 = neg2pos(index2, shape)
    if not same_sign:
        if index1.step is not None and index1.step < 0:
            index1 = invert_slice(index1, shape, False)
        if index2.step is not None and index2.step < 0:
            index2 = invert_slice(index2, shape, False)
    start1, step1, length1 = slice_navigator(index1, shape, False)
    start2, step2, length2 = slice_navigator(index2, shape, False)
    return (start1, step1, length1) == (start2, step2, length2)


def guess_shape(index, shape):
    """Guess the output shape obtained by indexing a volume.

    Parameters
    ----------
    index : sequence[index_like]
        Index.
    shape : sequence[int]
        Input shape

    Returns
    -------
    shape : tuple[int]
        Output shape

    Raises
    ------
    ValueError
        If the length of `shape` is not consistent with `index`.

    """
    index = expand_index(index, shape)

    output_shape = []
    while len(index) > 0:
        # pop first index
        idx, *index = index
        if idx is None:
            # new axis
            output_shape.append(1)
            continue
        if isinstance(idx, oob_slice):
            # out-of-bound slice
            output_shape.append(0)
            if not idx.newaxis:
                _, *shape = shape
            continue
        sz, *shape = shape
        if isinstance(idx, int):
            # dropped axis
            continue
        if isinstance(idx, slice):
            # kept axis
            output_shape.append(slice_length(idx, sz))
            continue

    if len(shape) > 0:
        raise ValueError("Shape to long for this index vector")

    return tuple(output_shape)


def expand_index(index, shape):
    """Expand indices in a tuple.

    * Ellipses are replaced with slices
    * Slices are simplified
    * Implicit slices are appended on the right
    * Negative indices (that count from then end) are made positive.

    This function is related to `nibabel.fileslice.canonical_slicers`,
    but with a few differences:
    * Floating point indices are *not* accepted
    * Negative steps are *not* made positive

    Parameters
    ----------
    index : index_like or sequence of index_like
        A tuple of indices with value types in
        {None, int, slice, oob_slice, ellipsis}
    shape : sequence of int
        The input shape

    Returns
    -------
    index : tuple of index_like
        A tuple of indices with value types in {None, int, slice, oob_slice}

    Raises
    ------
    TypeError
        * If the input is not a sequence of index_like
    ValueError
        * If more than one ellipsis is present
        * If a tensor or array that does not represent a scalar is present
    IndexError
        * If a scalar index falls out-of-bound

    """
    index = make_list(index)
    shape = list(shape)
    nb_dim = len(shape)

    def is_int(elem):
        if torch and torch.is_tensor(elem):
            return elem.dtype in (torch.int32, torch.int64) and not elem.shape
        elif np and isinstance(elem, np.ndarray):
            return elem.dtype in (np.int32, np.int64) and not elem.shape
        elif isinstance(elem, int):
            return True
        else:
            return False

    # compute the number of input dimension that correspond to each index
    # an individual index can be a slice, int, ellipsis, or None.
    #
    #    type        | in             | out         | supported
    #   --------------------------------------------------------
    #    None        | 0              | 1           | yes
    #    slice       | 1              | 1           | yes
    #    int         | 1              | 0           | yes
    #    ellipsis    | (dim - others) | same        | yes
    #    list[int]   | 1              | 1           | no
    #    list[bool]  | 1              | 1           | no
    #    array[int]  | 1              | array.dim() | no
    #    array[bool] | array.dim()    | 1           | no
    #
    # I've always found numpy's advanced indexing a bit weird (basically,
    # array-like indices must be broadcastable) and less intuitive than
    # Matlab's, although it's probably a bit more flexible.
    # Anyway, we choose -- like nibabel -- to not support advanced
    # indexing.

    nb_dim_in = []
    nb_dim_out = []
    ind_ellipsis = None
    for n_ind, ind in enumerate(index):
        if ind is None:
            nb_dim_in.append(0)
            nb_dim_out.append(1)
        elif isinstance(ind, slice):
            nb_dim_in.append(1)
            nb_dim_out.append(1)
        elif isinstance(ind, oob_slice):
            nb_dim_in.append(0 if ind.newaxis else 1)
            nb_dim_out.append(1)
        elif ind is Ellipsis:
            if ind_ellipsis is not None:
                raise ValueError('Cannot have more than one ellipsis.')
            ind_ellipsis = n_ind
            nb_dim_in.append(-1)
            nb_dim_out.append(-1)
        elif is_int(ind):
            ind = np.asanyarray(ind)
            if ind.ndim > 0:
                raise ValueError('Integer indices should be scalars '
                                 'Got array with shape {}.'.format(ind.shape))
            nb_dim_in.append(1)
            nb_dim_out.append(0)
            index[n_ind] = ind.item()
        else:
            raise TypeError('Indices should be integers, slices '
                            'or ellipses. Got {}.'.format(type(ind)))

    # deal with ellipsis
    nb_known_dims = sum(n for n in nb_dim_in if n > 0)
    if ind_ellipsis is not None:
        nb_dim_in[ind_ellipsis] = max(0, nb_dim - nb_known_dims)
        nb_dim_out[ind_ellipsis] = nb_dim_in[ind_ellipsis]
    else:
        index.append(Ellipsis)
        nb_dim_in.append(max(0, nb_dim - nb_known_dims))
        nb_dim_out.append(nb_dim_in[-1])

    # transform each index into a slice
    nb_ind = 0
    index0 = index
    index = []
    for d, ind in enumerate(index0):
        if ind is None:
            # new axis (0 -> 1)
            nb_ind += nb_dim_in[d]
            index.append(None)
        elif isinstance(ind, slice):
            # slice (1 -> 1)
            index.append(simplify_slice(ind, shape[nb_ind]))
            nb_ind += nb_dim_in[d]
        elif isinstance(ind, oob_slice):
            index.append(ind)
            nb_ind += nb_dim_in[d]
        elif ind is Ellipsis:
            # ellipsis (... -> ...)
            #   we replace one ellipsis with a series of slices
            for dd in range(nb_ind, nb_ind + nb_dim_in[d]):
                index.append(slice(None))
                nb_ind += 1
        else:
            # scalar (1 -> 0)
            assert isinstance(ind, int)  # already checked
            ind = neg2pos(ind, shape[nb_ind])
            if ind < 0 or ind >= shape[nb_ind]:
                raise IndexError('Out-of-bound index in dimension {} '
                                 '({} not in [0, {}])'
                                 .format(nb_ind, ind, shape[nb_ind] - 1))
            index.append(ind)
            nb_ind += nb_dim_in[d]

    return tuple(index)


def compose_index(parent, child, full_shape):
    """Compose two sub-indexing

    Parameters
    ----------
    parent : sequence[index_like]
    child : sequence[index_like]
    full_shape : sequence[int]
        Shape of the original volume, that `parent` indexes.

    Returns
    -------
    index : tuple[index_like]

    Raises
    ------
    IndexError
        If a scalar index falls out-of-bounds.
    ValueError, TypeError
        Raised by `expand_index` when called on `parent` and `child`

    """

    def oob(i):
        """Out-of-bound error."""
        raise IndexError('Index out-of-bound in parent dimension '
                         '{}.'.format(i))

    parent = expand_index(parent, full_shape)
    sub_shape = guess_shape(parent, full_shape)
    child = list(expand_index(child, sub_shape))

    i_parent = -1
    new_parent = []
    while parent:
        # copy leading `None`s
        while child and child[0] is None:
            new_parent = [*new_parent, None]
            child = child[1:]

        # if no more children, just keep the remaining parents
        # (I don't think this should ever happen)
        if not child:
            new_parent += parent
            break

        # extract leading dimension
        p, *parent = parent
        i_parent += 1

        if isinstance(p, int):
            # dropped axis
            # pop original dimension
            sz0, *full_shape = full_shape
            new_parent.append(p)
            continue

        # pop sub dimension
        c, *child = child
        sz, *sub_shape = sub_shape

        if p is None:
            # virtual axis
            if isinstance(c, int):
                if c != 0:
                    # out-of-bound
                    oob(i_parent)
                continue
            if isinstance(c, slice):
                # keep the new axis
                if slice_length(c, 1) == 0:
                    # out-of-bound
                    new_parent.append(oob_slice(newaxis=True))
                else:
                    new_parent.append(None)
                continue
            if isinstance(c, oob_slice):
                new_parent.append(oob_slice(newaxis=True))
            assert False, "p is None and c is {}".format(c)

        if isinstance(p, oob_slice):
            if not p.newaxis:
                sz0, *full_shape = full_shape
            # out-of-bound slice into a new axis
            if isinstance(c, int):
                # out-of-bound
                oob(i_parent)
                continue
            if isinstance(c, (slice, oob_slice)):
                # keep the axis
                new_parent.append(p)
                continue
            assert False, "p is oob_slice(newaxis=True) and c is {}".format(c)

        # pop original dimension
        sz0, *full_shape = full_shape

        if isinstance(p, slice):
            # slice
            if isinstance(c, int):
                # convert to scalar
                if c < 0 or c >= sz:
                    oob(i_parent)
                if p.step is not None and p.step < 0:
                    start = sz0 - 1 if p.start is None else p.start
                    new_parent.append(start + c * p.step)
                else:
                    new_parent.append((p.start or 0) + c * (p.step or 1))
                continue
            if isinstance(c, slice):
                # merge slices
                length = slice_length(c, sz)
                if length == 0:
                    # out-of-bound
                    new_parent.append(oob_slice())
                    continue
                # pre-compute child's start and step
                if c.step is not None and c.step < 0:
                    start = sz - 1 if c.start is None else c.start
                    step = c.step
                else:
                    start = 0 if c.start is None else c.start
                    step = 1 if c.step is None else c.step
                # pre-compute parent's start and step
                if p.step is not None and p.step < 0:
                    start0 = sz0 - 1 if p.start is None else p.start
                    step0 = p.step
                else:
                    start0 = 0 if p.start is None else p.start
                    step0 = 1 if p.step is None else p.step
                start = start0 + start * step0
                step = step0 * step
                stop = start + length * step
                if step < 0 and stop < 0:
                    # need to simplify this here because
                    # simplify_slice fails otherwise
                    stop = None
                new_slice = simplify_slice(slice(start, stop, step), sz0,
                                           do_neg2pos=False)
                new_parent.append(new_slice)
                continue
            if isinstance(c, oob_slice):
                new_parent.append(c)
            assert False, "p is slice and c is {}".format(c)

    while child:
        c, *child = child
        if c is not None:
            raise IndexError('More indices than dimensions')
        new_parent.append(c)

    return tuple(new_parent)


def split_operation(perm, slicer, direction):
    """Split the operation `slicer of permutation` into subcomponents.

    Symbolic slicing is encoded by a permutation and an indexing operation.
    The operation `sub.data()` where `sub` has been obtained by applying
    a succession of permutations and sub-indexings on `full` should be
    equivalent to
    ```python
    >>> full_data = full.data()
    >>> sub_data = full_data.permute(sub.permutation)[sub.slicer]
    ```
    However, the slicer may create new axes or drop (i.e., index with
    a scalar) original axes. This function splits `slicer of permutation`
    into more sub-components.
    If the direction is 'read'
        * `slicer_sub`  : unpermuted slicer without new axes
        * `perm`        : permutation without dropped axes
        * `slicer_add`  : slicer that adds new axes
    ```python
    >>> dat = full[slicer_sub].transpose(perm)[slicer_add]
    ```
    If the direction is 'write':
        * `slicer_drop` : slicer that drops new axes
        * `perm`        : inverse permutation without dropped axes
        * `slicer_sub`  : unpermuted slicer without
    ```python
    >>> full[slicer_sub] = dat[slicer_drop].transpose(perm)
    ```

    Parameters
    ----------
    perm : sequence[int]
        Permutation of `range(self._dim)`
    slicer : sequence[index_like]
        Sequence of indices that slice into `self._shape`
    direction : {'r', 'w'}
        Either split for reading ('r') or writing ('w') a chunk.

    Returns
    -------
    slicer_sub ('r') or slicer_drop ('w') : tuple
    perm : tuple
    slicer_add ('r') or slicer_sub ('w') : tuple

    """

    def remap(perm):
        """Re-index dimensions after some have been dropped"""
        remaining_dims = sorted(perm)
        dim_map = {}
        for new, old in enumerate(remaining_dims):
            dim_map[old] = new
        remapped_dim = [dim_map[d] for d in perm]
        return remapped_dim

    def select(index, seq):
        """Select elements into a sequence using a list of indices"""
        return [seq[idx] for idx in list(index)]

    slicer_nonew = list(filter(lambda x: not is_newaxis(x), slicer))
    slicer_nodrop = filter(lambda x: not is_droppedaxis(x), slicer)
    slicer_sub = select(invert_permutation(perm), slicer_nonew)
    perm_nodrop = [d for d, idx in zip(perm, slicer_nonew)
                   if not is_droppedaxis(idx)]
    perm_nodrop = remap(perm_nodrop)

    if direction.lower().startswith('r'):
        slicer_add = map(lambda x: slice(None) if x is not None else x,
                         slicer_nodrop)
        return tuple(slicer_sub), tuple(perm_nodrop), tuple(slicer_add)
    elif direction.lower().startswith('w'):
        slicer_drop = map(lambda x: 0 if x is None else slice(None),
                          slicer_nodrop)
        inv_perm_nodrop = invert_permutation(perm_nodrop)
        return tuple(slicer_drop), tuple(inv_perm_nodrop), tuple(slicer_sub)
    else:
        raise ValueError("direction should be in ('read' ,'write') "
                         "but got {}.".format(direction))


def invert_permutation(perm):
    """Return the inverse of a permutation

    Parameters
    ----------
    perm : sequence[int]
        Permutations. A permutation is a shuffled set of indices.

    Returns
    -------
    iperm : list[int]
        Inverse permutation.

    """
    iperm = [0] * len(perm)
    for i, p in enumerate(perm):
        iperm[p] = i
    return iperm


def slicer_sub2ind(slicer, shape):
    """Convert a multi-dimensional slicer into a linear slicer.

    Parameters
    ----------
    slicer : sequence[slice or int]
        Should not have new axes.
        Should have only positive strides.
    shape : sequence[int]
        Should have the same length as slicer

    Returns
    -------
    index : slice or int or list[int]

    """

    slicer = expand_index(slicer, shape)
    shape_out = guess_shape(slicer, shape)
    if any(isinstance(idx, slice) and idx.step and idx.step < 0
           for idx in slicer):
        raise ValueError('sub2ind does not like negative strides')
    if any(is_newaxis(idx) for idx in slicer):
        raise ValueError('sub2ind does not like new axes')

    slicer0 = slicer
    shape0 = shape

    # 1) collapse slices
    slicer = list(reversed(slicer))
    shape = list(reversed(shape))
    new_slicer = slice(None)
    new_shape = 1
    while len(slicer) > 0:
        idx, *slicer = slicer
        shp, *shape = shape

        if isinstance(idx, slice):
            if idx == slice(None):
                # merge full slices
                new_shape *= shp
                continue
            else:
                # stop trying to merge
                if idx.step in (1, None):
                    start = idx.start or 0
                    stop = idx.stop or shp
                    new_slicer = slice(start * new_shape, stop * new_shape)
                    new_shape *= shp
                    new_slicer = simplify_slice(new_slicer, new_shape)
                    new_slicer = [new_slicer] + slicer
                    new_shape = [new_shape] + shape
                else:
                    if new_shape != 1:
                        new_slicer = [new_slicer, idx] + slicer
                        new_shape = [new_shape, shp] + shape
                    else:
                        new_slicer = [idx] + slicer
                        new_shape = [shp] + shape
                break

        elif isinstance(idx, int):
            if shp == 1:
                continue
            else:
                new_slicer = slice(idx * new_shape, (idx + 1) * new_shape)
                new_shape *= shp
                new_slicer = simplify_slice(new_slicer, new_shape)
                if new_shape != 1:
                    new_slicer = [new_slicer] + slicer
                    new_shape = [new_shape] + shape
                else:
                    new_slicer = [idx] + slicer
                    new_shape = [shp] + shape
                break

    new_slicer = make_list(new_slicer)
    new_shape = make_list(new_shape)

    assert math.prod(shape0) == math.prod(new_shape), \
        "Oops: lost something: {} vs {}".format(math.prod(shape0),
                                                math.prod(new_shape))

    # 2) If we have a unique index, we can stop here
    if len(new_slicer) == 1:
        return new_slicer[0]

    # 3) Extract linear indices
    strides = [1] + list(cumprod(new_shape[1:]))
    new_index = []
    for idx, shp, stride in zip(new_slicer, new_shape, strides):
        if isinstance(idx, slice):
            start = idx.start or 0
            stop = idx.stop or shp
            step = idx.step or 1
            idx = list(range(start, stop, step))
        else:
            idx = [idx]
        idx = [i * stride for i in idx]
        if new_index:
            new_index = list(itertools.product(idx, new_index))
            new_index = [sum(idx) for idx in new_index]
        else:
            new_index = idx

    assert len(new_index) == math.prod(shape_out), \
        "Oops: lost something: {} vs {}".format(len(new_index),
                                                math.prod(shape_out))

    return new_index











