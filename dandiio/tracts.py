import numpy as np
from types import GeneratorType as generator
from nibabel.streamlines.trk import TrkFile
import random


MAX_TRACTS = 10000


class TractsTRK:
    """
    This class reads a TRK stremalines file (using nibabel)
    and implements methods that allow serving and/or converting the
    streamlines data into neuroglancer's precomputed skeleton format.

    The skeleton format was originally implemented to render skeletonized
    volumetric segmentations, with a relatively small number of
    individual objects. Since tractography (especially high-resolution
    tractography) can generate millions of streamlines, saving each
    streamline as an individual skeleton is very ineficient -- both for
    querying and rendering.

    Instead, we combine all streamlines into a single skeleton. Still,
    the large number of streamlines slows down rendering quite a lot, so
    I currently sample `MAX_TRACTS` streamlines from the file to display
    (currently set to 10,000).

    There is no "multi-scale" representation of the tracts in neuroglancer
    (i.e., generate tracts with a smaller or larger number of edges
    based on the current rendering scale). Maybe this is something that
    we should ask be added to neuroglancer.

    If a segmentation of the tractogram is available, it would probably
    make sense to save the tracts belonging to each segment in a
    different skeleton. This would allow switching each segment on and
    off, and would allow segments to be displayed in different colors.

    I also save a unit-norm orientation vector for each vertex. Saving
    this information as `vertex_attribute` allows using it for rendering
    (e.g., it can be used to render orientation-coded tracts).

    The specification of the precomputed skeleton format is available here:
        https://github.com/google/neuroglancer/blob/master/
        src/neuroglancer/datasource/precomputed/skeletons.md
    """

    def __init__(self, fileobj):
        self.fileobj = fileobj
        self.trkfile = None

    def _load(self, lazy=False):
        self.trkfile = TrkFile.load(self.fileobj, lazy_load=lazy)

    def _is_fully_loaded(self):
        return self.trkfile and not isinstance(self.trkfile.streamlines, generator)

    def __getitem__(self, id):
        if not self._is_fully_loaded():
            self._load()
        return self.trkfile.streamlines[id]

    def __len__(self):
        if not self._is_fully_loaded():
            self._load()
        return len(self.trkfile.streamlines)

    def precomputed_prop_info(self, combined=False):
        """
        Return the segment properties 'info' JSON

            https://github.com/google/neuroglancer/blob/master/
            src/neuroglancer/datasource/precomputed/segment_properties.md
        """
        if not self._is_fully_loaded():
            self._load()

        num_tracts = min(MAX_TRACTS, len(self.trkfile.streamlines))

        # if not combined, each tract correspond to a different "segment"
        # if combined, all tracts are merged into a single skeleton
        ids = ["1"] if combined else [str(i) for i in range(num_tracts)]

        info = {
            "@type": "neuroglancer_segment_properties",
            "inline": {
                "ids": ids,
                "properties": [{
                    "id": "label",
                    "type": "label",
                    "values": ids,
                }]
            }
        }
        return info

    def precomputed_skel_info(self):
        """
        Return the skeleton 'info' JSON

            https://github.com/google/neuroglancer/blob/master/
            src/neuroglancer/datasource/precomputed/skeletons.md
        """
        if not self.trkfile:
            self._load(lazy=True)

        # No need for a transform, as nibabel automatically converts
        # TRK coordinates to mm RAS+
        # All we need to do is convert mm to nm (we do this in the
        # track serving functions)

        info = {
            "@type": "neuroglancer_skeletons",
            "vertex_attributes": [
                {
                    "id": "orientation",
                    "data_type": "float32",
                    "num_components": 3,
                },
            ],
            "segment_properties": "prop",
        }

        return info

    def precomputed_skel_tract_combined(self, id=1):
        """
        Return all tracts encoded in a single skeleton.

        `id` should alway be `1` (since there is only one skeleton)

        TODO: combine tracts per segment if a tractogram segmentation
        is available

            https://github.com/google/neuroglancer/blob/master/
            src/neuroglancer/datasource/precomputed/skeletons.md
            #encoded-skeleton-file-format
        """
        if id != 1:
            return b''

        if not self._is_fully_loaded():
            self._load()

        num_tracts = len(self.trkfile.streamlines)
        ids = list(range(num_tracts))
        random.seed(1234)
        random.shuffle(ids)

        num_vertices = num_edges = 0
        vertices = b''
        edges = b''
        orientations = b''

        for id in range(min(num_tracts, MAX_TRACTS)):
            id = ids[id]
            tract = self.trkfile.streamlines[id] * 1E6  # nanometer

            # --- compute orientations ---------------------------------
            # 1) compute directed orientation of each edge
            orient = tract[1:] - tract[:-1]
            # 2) compute directed orientation of each vertex as the
            #    length-weighted average of the orientations of its edges
            #    (we just add them, and normalize later)
            orient = np.concatenate([
                orient[:1],                 # first vertex:   only one edge
                orient[:-1] + orient[1:],   # other vertices: two edges
                orient[-1:],                # last vertex:    only one edge
            ], 0)
            # 3) make orientations unit-length
            length = np.sqrt((orient * orient).sum(-1, keepdims=True))
            length = np.clip(length, 1e-12, None)
            orient /= length
            # ----------------------------------------------------------

            # vertex_positions: [num_vertices, 3] float32le (C-order)
            vertices += np.asarray(tract, dtype='<f4').tobytes()
            # edges: [num_edges, 2] uint32le (C-order)
            edges += np.stack([
                np.arange(len(tract) - 1, dtype='<u4') + num_vertices,
                np.arange(1, len(tract), dtype='<u4') + num_vertices,
            ], -1).tobytes()
            # orientations: [num_vertices, 3] float32le (C-order)
            orientations += np.asarray(orient, dtype='<f4').tobytes()
            # increase counters
            num_vertices += len(tract)
            num_edges += len(tract) - 1

        bintract = b''
        # num_vertices: uint32le
        bintract += np.asarray(num_vertices, dtype='<u4').tobytes()
        # edges: uint32le
        bintract += np.asarray(num_edges, dtype='<u4').tobytes()
        # vertex_positions: [num_vertices, 3] float32le (C-order)
        bintract += vertices
        # edges: [num_edges, 2] uint32le (C-order)
        bintract += edges
        # attributes | orientation: [num_vertices, 3] float32le (C-order)
        bintract += orientations

        return bintract

    def precomputed_skel_tract(self, id):
        """
        Return a single tract

        This function is used in the case where each streamline is
        encoded by a single skeleton. Note that this is very inefficient.

        TODO: add orientation attribute
        """
        if not self._is_fully_loaded():
            self._load()

        num_tracts = len(self.trkfile.streamlines)
        ids = list(range(num_tracts))
        random.seed(1234)
        random.shuffle(ids)
        id = ids[id]

        tract = self.trkfile.streamlines[id] * 1E6

        bintract = b''
        # num_vertices: uint32le
        bintract += np.asarray(len(tract), dtype='<u4').tobytes()
        # edges: uint32le
        bintract += np.asarray(len(tract) - 1, dtype='<u4').tobytes()
        # vertex_positions: [num_vertices, 3] float32le (C-order)
        bintract += np.asarray(tract, dtype='<f4').tobytes()
        # edges: [num_edges, 2] uint32le (C-order)
        bintract += np.stack([
            np.arange(len(tract) - 1, dtype='<u4'),
            np.arange(1, len(tract), dtype='<u4')
        ], -1).tobytes()

        print('serve tract', id, '/', num_tracts)
        return bintract
