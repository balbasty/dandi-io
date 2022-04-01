from . import tiffio_utils
import numpy as np
import warnings
import os
import fsspec
import xmlschema
from typing import Optional, Literal
import tifffile


class _TiffReader:

    file = None                                     # File handle
    is_mine = False                                 # Do we own the file?
    endian: Literal['little', 'big'] = 'little'     # Endianness
    count_size: int = 4                             # Nb bytes to store counts
    offset_size: int = 4                            # Nb bytes to store offsets
    bigTIFF: bool = False                           # Classic or Big TIFF?

    def __init__(self, file):

        if isinstance(file, (str, os.PathLike)):
            self.fname = file
            self.file = fsspec.open(file, 'rb')
            self.is_mine = True
        else:
            if not hasattr(file, 'seek'):
                raise ValueError('Expected file_like object but got', type(file))
            self.fname = getattr(file, 'name', None)
            self.file = file
            self.is_mine = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_if_mine()

    def close_if_mine(self):
        if not hasattr(self.file, 'close'):
            return
        if self.is_mine and not self.file.closed:
            self.file.close()

    def close(self):
        if not hasattr(self.file, 'close'):
            return
        self.file.close()

    def read_header(self):
        buffer = self.file.read(16)
        endian = buffer[:2].decode('ascii')
        if endian == 'II':
            endian = 'little'
        elif endian == 'MM':
            endian = 'big'
        else:
            raise ValueError('Expected "II" or "MM" for endianness but got', endian)
        self.endian = endian
        magic = self.int_from_bytes(buffer[2:4])
        if magic == 43:
            # BigTIFF
            self.count_size = 8
            self.offset_size = self.int_from_bytes(buffer[4:6])
            sanity = self.int_from_bytes(buffer[6:8])
            if sanity:
                warnings.warn(f'BigTIFF sanity word is nonzero: {sanity}')
            self.first_fid = self.int_from_bytes(buffer[8:16])
            self.bigTIFF = True
        elif magic == 42:
            # classic TIFF
            self.count_size = 4
            self.offset_size = 4
            self.first_fid = self.int_from_bytes(buffer[4:8])
        else:
            raise ValueError('Expected magic number 42 but got', magic)
        self.entry_size = 4 + self.count_size + self.offset_size
        return self.first_fid

    def int_from_bytes(self, x):
        return int.from_bytes(x, byteorder=self.endian)

    def read_fid_size(self, fid):
        self.file.seek(fid)
        nb_dir = self.file.read(2)
        nb_dir = self.int_from_bytes(nb_dir)
        return nb_dir

    def next_fid(self, fid):
        self.file.seek(fid)
        nb_dir = self.int_from_bytes(self.file.read(2))
        self.file.seek(fid + 2 + self.entry_size * nb_dir)
        fid = self.int_from_bytes(self.file.read(self.offset_size))
        return fid

    def read_fid_header(self, fid):
        nb_dir = self.read_fid_size(fid)

        # Read all headers at once
        bo = tiffio_utils.bo_short(self.endian)
        dir_dtype = np.dtype([
            ('tag', f'{bo}u2'),
            ('type', f'{bo}u2'),
            ('count', f'{bo}u{self.count_size}'),
            ('value_or_offset', f'b', self.offset_size),
        ])
        buffer = self.file.read(self.entry_size * nb_dir)
        raw_headers = np.frombuffer(buffer, dtype=dir_dtype, count=nb_dir)

        # Parse `value_or_offset`
        headers = dict()
        for raw_header in raw_headers:
            count = raw_header['count']
            tag, _ = tiffio_utils.int_to_tag(raw_header['tag'])
            typename, val_dtype = tiffio_utils.int_to_type(raw_header['type'])
            val_dtype = val_dtype['numpy'](bo)
            nb_bytes = count * val_dtype.itemsize
            if nb_bytes <= self.offset_size:
                value = raw_header['value_or_offset']
                value = value[:nb_bytes].view(val_dtype)
            else:
                offset_dtype = f'{bo}u{self.offset_size}'
                offset = raw_header['value_or_offset'].view(offset_dtype)
                self.file.seek(offset)
                value = self.file.read(count * val_dtype.itemsize)
                value = np.frombuffer(value, dtype=val_dtype)
            if typename.endswith('RATIONAL'):
                value = tiffio_utils.rational_to_float(value)
            if value.dtype == np.dtype('S1'):
                value = value.view('S' + str(count)).item()
            else:
                value = value.tolist()
                if count == 1:
                    value = value[0]
            print(tag, value)
            headers[tag] = value

        return headers


class _OMETiffReader(_TiffReader):

    @staticmethod
    def get_ome_xml(headers, validation='strict'):
        if 'ImageDescription' not in headers:
            return None
        buffer = headers['ImageDescription']
        if not buffer[:5] == b'<?xml':
            return None
        # drop trailing characters
        buffer = buffer.decode('utf-8')
        while buffer[-1] != '>':
            buffer = buffer[:-1]
        schema = 'http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd'
        with fsspec.open(schema) as schema_file:
            schema = xmlschema.XMLSchema(schema_file)
        ome = schema.to_dict(buffer, validation=validation)
        return ome


def get_ome_xml(file, backend='tifffile'):
    if backend == 'tifffile':
        with tifffile.TiffFile(file) as f:
            ome = f.ome_metadata
        schema = 'http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd'
        with fsspec.open(schema) as schema_file:
            schema = xmlschema.XMLSchema(schema_file)
        ome = schema.to_dict(ome)
    else:
        with _OMETiffReader(file) as f:
            fid = f.read_header()
            headers = f.read_fid_header(fid)
        ome = f.get_ome_xml(headers)
    return ome