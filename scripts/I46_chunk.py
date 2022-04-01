"""
Try to export a slab of SPIM data chunked across multiple ome.tif files.
The 000026/I48 data is chunked across a single dimensions, and chunks are
in order, so for now I am not using the json metadata at all.
I read the OME-XML metadata of one of the chunks to get the chunk size.
"""
import tifffile
from dandiio import RemoteDandiFileSystem
from dandiio.tiffio import get_ome_xml
import fsspec
from fsspec.implementations.memory import MemoryFileSystem
import imagecodecs.numcodecs
import xarray
import re
import numpy as np
import matplotlib.pyplot as plt


# Setup a DandiFS with root at dandiset/000026
dandifs = RemoteDandiFileSystem('000026')

# Get all files
#   Directly calling get_assets_with_path_prefix is way faster than
#   using dandifs.glob (because glob uses walk, and therefore requests
#   all files under the deepest directory)
files = dandifs.dandiset.get_assets_with_path_prefix(
    'rawdata/sub-I48/ses-SPIM/microscopy/'
    'sub-I48_ses-SPIM_sample-BrocaAreaS01_stain-Calretinin_chunk')

# Split json/tif
files = list(files)
json_files = list(sorted(f.path for f in files if f.path.endswith('.json')))
files = list(sorted(f.path for f in files if f.path.endswith('.ome.tif')))

# Read metadata of first chunk
with dandifs.open(files[0]) as f:
    ome = get_ome_xml(f)
shape = {
    axis: ome['ome:Image'][0]['ome:Pixels']['@Size' + axis]
    for axis in 'XYZ'
}
endian = '>' if ome['ome:Image'][0]['ome:Pixels']['@BigEndian'] else '<'
dtype = np.dtype(ome['ome:Image'][0]['ome:Pixels']['@Type'])
if dtype.str[0] not in (endian, '|'):
    dtype = endian + dtype.str[1:]
stain = re.search(r'stain-([^_\.]+)', files[0]).group(1)
print(stain)

# Generate the TiffSequence and use it to make a zarr store
# Files are not actually read at that point
tiffseq = tifffile.TiffSequence(files)
store = tiffseq.aszarr(
    dtype=dtype,
    chunkshape=[shape['X'], shape['Y'], shape['Z']],
    fillvalue=0,
    axestiled={0: 1},  # axis 0 of TiffSequence maps to axis 1 of chunks
    zattrs={
        '_ARRAY_DIMENSIONS': ['X', 'Y', 'Z']
    },
)

# Write a ReferenceFileSystem in json format
#   It seems that currently, all files need to live under the same directory.
#   It works if we use Dandi paths, but would not work with pure S3 URLs?
mem_fs = MemoryFileSystem()
with mem_fs.open('ref_fs.json', 'w',  newline='\n') as f:
    store.write_fsspec(
        f,
        '/'.join(files[0].split('/')[:-1]),
        groupname=stain,
        codec_id='imagecodecs_tiff',
        _append=False,
    )

# Look at the reference JSON file
with fsspec.open('ref_fs.json', 'r', protocol='memory') as f:
    buf = f.read()
    print(buf)

imagecodecs.numcodecs.register_codecs()

# Instantiate the reference file system
mapper = fsspec.get_mapper(
    'reference://',
    fo='ref_fs.json',
    target_protocol='memory',
    remote_protocol='local',
    fs=dandifs,  # The DandiFS will convert dandi paths to S3 URLs
)

# Create the array
dataset = xarray.open_dataset(
    mapper,
    engine='zarr',
    mask_and_scale=False,
    backend_kwargs={'consolidated': False},
)
print(dataset)

sub = dataset.sel({'X': slice(1024, 2048),
                   'Y': slice(1024, 2048),
                   'Z': dataset.dims['Z']//2})
image = sub[stain]
xarray.plot.imshow(image, size=6, aspect=1)
plt.show()