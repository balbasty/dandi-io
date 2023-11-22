"""
This module contains utilities specific to Dandiset 000026
"""
import json
import math
import tifffile
import numpy as np
from .dandifs import RemoteDandiFileSystem


class StitchedLSFM:
    """
    Stitch LSFM tiles saved as individual TIFF files.

    This class assumes that the "ChunkTransformationMatrixAxis" of each
    tile only contains a translation, and that the translation parameters
    are integral (in voxels).

    We also assume (for now) that the "ChunkTransformationMatrixAxis" of
    all tiles is ["X", "Y", "Z"]. We further assume that overlap only
    happens in the XY plane.

    ```python
    glob = 'dandi://dandi/000026/sub-I48/ses-SPIM/micr/*stain-Calretinin*.ome.tif'
    stitched_array = StitchedLSFM(glob)
    loaded_array = stitched_array[:128, :128, :128]
    ```
    """  # noqa: E501

    def __init__(self, glob):
        """
        Parameters
        ----------
        glob : str
            Globbing pattern that returns all tiles of the
            volume to stitch
        """
        self.glob = glob
        self._parsed = False

    def get_info(self):
        """
        Return the info JSON for neuroglancer's precomputed format

        See:
        https://github.com/google/neuroglancer/blob/master
            /src/neuroglancer/datasource/precomputed
            /volume.md#info-json-file-specification
        """
        if not self._parsed:
            self._parse_files()

        unitscale = self.unit2scale[self.infos[0]['unit']]
        vxsize = self.infos[0]['vx']

        scales = []
        maxlevel = int(math.floor(math.log2(max(self.shape) // 64)))
        for level in range(maxlevel):
            mult = 2**level
            subvx = [float(v * (2**level) * unitscale) for v in vxsize]
            subshape = [s // mult for s in self.shape]
            scales.append(dict(
                chunk_sizes=[[64, 64, 64]],
                encoding="raw",
                key=str(level),
                resolution=subvx,
                size=subshape,
                voxel_offset=[0, 0, 0],
            ))

        info = dict(
            data_type="float32",
            num_channels=1,
            type="image",
            scales=scales,
        )
        return info

    unit2scale = {
        'm': 1E9,
        'dm': 1E8,
        'cm': 1E7,
        'mm': 1E6,
        'um': 1E3,
        'nm': 1,
        'pm': 1E-3,
    }

    def _parse_files(self):
        fs = RemoteDandiFileSystem.for_url(self.glob)

        # find json files
        vpaths = fs.glob(self.glob)
        vpaths = [f for f in vpaths if not f.endswith('.json')]
        jpaths = []
        for path in vpaths:
            jpath = None
            for _ in range(3):
                if fs.exists(path + '.json'):
                    jpath = path + '.json'
                    break
                path = '.'.join(path.split('.')[:-1])
            jpaths.append(jpath)

        # read useful metadatas
        defmat = [[0]*4]*4
        defax = ['X', 'Y', 'Z']
        defpix = [1., 1., 1.]
        defunit = 'mm'

        infos = []
        for vpath, jpath in zip(vpaths, jpaths):
            info = {}
            if jpath:
                with fs.open(jpath) as f:
                    meta = json.load(f)
                info['shift'] = meta.get('ChunkTransformationMatrix', defmat)
                info['shift'] = np.asarray(info['shift'])[:3, -1].tolist()
                info['vx'] = meta.get('PixelSize', defpix)
                info['unit'] = meta.get('PixelSizeUnits', defunit)
                info['axes'] = meta.get('ChunkTransformationMatrixAxis', defax)
            with fs.open(vpath) as f:
                with tifffile.open(f):
                    info['shape'] = f.series[0].levels[0].shape[:3]
            infos.append(info)

        self.vpaths = vpaths
        self.jpaths = jpaths
        self.infos = infos
        self._compute_fov()
        self._compute_overlap_size()
        self._parsed = True

    def _compute_fov(self):
        """Compute the shape of the full stitched array"""
        mins = np.full([3], np.iinfo('int64').max, dtype='int64')
        maxs = np.full([3], np.iinfo('int64').min, dtype='int64')
        for info in self.infos:
            cmin = np.asarray(info['shifts'], dtype='int64')
            cmax = np.asarray(info['shape'], dtype='int64') + cmin
            mins = np.minimum(cmin,  mins)
            maxs = np.maximum(cmax,  maxs)
            info['min'] = cmin.tolist()
            info['max'] = cmax.tolist()
        self.mins = mins
        self.maxs = maxs
        self.shape = (maxs - mins).tolist()

    def _compute_overlap_size(self):
        """Compute average overlap along X and Y"""
        xoverlap, xcount = 0, 0
        yoverlap, ycount = 0, 0
        for i, info1 in enumerate(self.infos):
            for _, info2 in enumerate(self.infos[i+1:]):
                if (info1['min'][0] < info2['max'][0] and
                        info2['min'][0] < info1['max'][0]):
                    # X overlap
                    overlap = max(info2['max'][0] - info1['min'][0],
                                  info1['max'][0] - info2['min'][0])
                    xoverlap += overlap
                    xcount += 1
                if (info1['min'][1] < info2['max'][1] and
                        info2['min'][1] < info1['max'][1]):
                    # Y overlap
                    overlap = max(info2['max'][1] - info1['min'][1],
                                  info1['max'][1] - info2['min'][1])
                    yoverlap += overlap
                    ycount += 1
        xoverlap /= xcount
        yoverlap /= ycount
        self.xoverlap = xoverlap
        self.yoverlap = yoverlap

    def fix_slicers(self, *slicers, shape=None):
        """
        Normalize slicers so that start/stop/step are all positive
        integers, and stop is the first non-sampled index (this
        matters when step > 1)
        """
        if len(slicers) > 1:
            if shape is None:
                shape = self.shape
            return [self._fix_slicer(slicer, shape=length)
                    for slicer, length in zip(slicers, shape)]
        else:
            slicer = slicers[0]
            if shape is None:
                shape = self.shape[0]
            return self._fix_slicer(slicer, shape)

    @staticmethod
    def _fix_slicer(slicer, length):
        if not isinstance(slicer, slice):
            raise TypeError('Slice must be of type `slice`')
        if (slicer.step or 1) < 0:
            raise ValueError('Slice must have positive step')
        start = slicer.start or 0
        start = (length + start) if start < 0 else start
        step = slicer.step or 1
        stop = slicer.stop if slicer.stop is not None else length
        stop = (length + stop) if stop < 0 else stop
        length = int(math.ceil((stop - start) / step))
        stop = start + length*step + 1
        return slice(start, stop, step)

    def compute_subfov(self, slicex, slicey, slicez):
        """Compute the shape of the queried FOV"""
        return [int(math.ceil((slc.stop - slc.start) / slc.step))
                for slc in (slicex, slicey, slicez)]

    @staticmethod
    def isin(info, slicer):
        """Check if a tile is in the queried FOV"""
        mn, mx = info['min'], info['max']
        for i in range(3):
            if not (mn[i] < slicer[i].stop and mx[i] > slicer[i].start):
                return False
        return True

    def compute_weights(self, imin, imax, cmin, cmax, step):
        """Compute cosine weights
        imin : tile's left border in global frame
        imax : tile's right border (+1) in global frame
        cmin : patch's left border in global frame
        cmax : patch's right border (+1) in global frame
        step : skip every `step` of tile's voxels

        returns: [nx, ny] weight map
        """
        xidx = np.arange(cmin[0], cmax[0], step[0]) - imin[0]
        yidx = np.arange(cmin[1], cmax[1], step[1]) - imin[1]
        xwgt, ywgt = np.ones(xidx.shape), np.ones(yidx.shape)

        xmsk_low = xidx < self.xoverlap
        xmsk_upp = xidx >= imax[0] - self.xoverlap
        xwgt[xmsk_low] = np.sin(
            (xidx[xmsk_low] + 0.5) / (self.xoverlap - 0.5)
        )
        xwgt[xmsk_upp] = np.sin(
            (imax[0] - xidx[xmsk_upp] - 0.5) / (self.xoverlap - 0.5)
        )

        ymsk_low = yidx < self.yoverlap
        ymsk_upp = yidx >= imax[1] - self.yoverlap
        ywgt[ymsk_low] = np.sin(
            (yidx[ymsk_low] + 0.5) / (self.yoverlap - 0.5)
        )
        ywgt[ymsk_upp] = np.sin(
            (imax[1] - yidx[ymsk_upp] - 0.5) / (self.yoverlap - 0.5)
        )

        return xwgt[:, None] * ywgt[None, :]

    def load(self, slicex, slicey, slicez):
        """Load and stitch a subvolume"""
        if not self._parsed:
            self._parse_files()
        slicex, slicey, slicez = self.fix_slicers(slicex, slicey, slicez)
        subshape = self.compute_subfov(slicex, slicey, slicez)

        start = [slicex.start, slicey.start, slicez.start]
        stop = [slicex.stop, slicey.stop, slicez.stop]
        step = [slicex.step, slicey.step, slicez.step]

        voldata = np.zeros(subshape, dtype='float32')
        weights = np.zeros(subshape[:2], dtype='float32')

        for vpath, info in zip(self.vpaths, self.infos):
            if not self.isin(info, [slicex, slicey, slicez]):
                # compute coordinate (in full frame) of the
                # first and voxels that belong to both the queried
                # patch and the current tile (cmin), and the first voxel
                # that does not belong to them (xmax)
                cmin = np.maximum(info['min'], start) - start
                cmin = start + (cmin / step).ceil().astype(int) * step
                cmax = np.minimum(info['max'], stop) - start - 1
                cmax = 1 + start + (cmax / step).ceil().astype(int) * step

                # compute cosine weight
                wgt = self.compute_weights(
                    info['min'], info['max'], cmin, cmax, step
                )

                # compute slicer
                slicein = [slice(cmin[i].item() - info['min'][i],
                                 cmax[i].item() - info['min'][i],
                                 step[i])
                           for i in range(3)]
                sliceout = [slice((cmin[i].item() - start[i]) // step[i],
                                  (cmax[i].item() - start[i]) // step[i] + 1)
                            for i in range(3)]

                # (partially) read tile and accumulate
                with RemoteDandiFileSystem.for_url(vpath).open(vpath) as f:
                    with TiffView(f) as view:
                        subdat = view[tuple(slicein)]
                        voldata[tuple(sliceout)] += subdat * wgt[:, :, None]
                weights += wgt

        weights[weights == 0] = 1
        voldata /= weights[:, :, None]

        return voldata

    def _getitem__(self, slicer):
        if not isinstance(slicer, tuple):
            slicer = (slicer,)
        if len(slicer) != 3:
            raise ValueError('StitchedLSFM objects must be sliced using '
                             'exactly 3 slicers (no advanced indexing, '
                             'no new axis, no ellipsis, etc.)')
        return self.load(*slicer)


class TiffView:
    """Class that loads a partial view into a tiff file

    ```python
    with TiffView(fileobj) as obj:
        dat = obj[128::2, 128::2, 128::2]
    ```
    """

    def __init__(self, fileobj):
        """
        fileobj : str or path or file-like
        """
        self.fileobj = fileobj
        self.tiff = None
        level = self.tiff.series[0].levels[0]
        self.shape = level.shape
        self.shape_page = level.pages[0].shape
        self.shape_stack = self.shape[:-len(self.shape_page)]

    def __del__(self):
        getattr(self.tiff, 'close', lambda x: None)()

    def open(self, *args, **kwargs):
        self.close()
        self.tiff = tifffile.TiffFile(self.fileobj, *args, **kwargs)

    @property
    def closed(self):
        return getattr(getattr(self.tiff, 'filehandle', 0), 'closed', False)

    def close(self):
        if not self.closed:
            getattr(self.tiff, 'close', lambda x: None)()
        self.tiff = None

    def __enter__(self):
        if self.closed:
            self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def load(self, slicex, slicey, slicez):
        dat = self.tiff.asarray(key=slicez)
        dat = dat[:, slicey, slicex]
        dat = dat.transpose()           # reorder a X, Y, Z
        return dat

    def _getitem__(self, slicer):
        if not isinstance(slicer, tuple):
            slicer = (slicer,)
        if len(slicer) != 3:
            raise ValueError('TiffView objects must be sliced using '
                             'exactly 3 slicers (no advanced indexing, '
                             'no new axis, no ellipsis, etc.)')
        return self.load(*slicer)
