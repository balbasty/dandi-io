import numpy as np
import struct
import enum


def bo_long(bo):
    if bo == '<':
        return 'little'
    if bo == '>':
        return 'big'
    if bo in '=|':
        return 'native'
    return bo


def bo_short(bo):
    if bo == 'little':
        return '<'
    if bo == 'big':
        return '>'
    if bo == 'native':
        return '='
    return bo


def rational(bo):
    return np.dtype([
        ('numerator', f'{bo}u4'),
        ('denominator', f'{bo}u4'),
    ])


def srational(bo):
    return np.dtype([
        ('numerator', f'{bo}s4'),
        ('denominator', f'{bo}s4'),
    ])


def rational_to_float(x):
    return x['numerator'] / x['denominator']


types = {
    'BYTE': dict(tag=1, size=1,
                 python=lambda x, _: bytes(x),
                 numpy=lambda _: np.bytes),
    'ASCII': dict(tag=2, size=1,
                  python=lambda x, _: x.decode('ascii'),
                  numpy=lambda _: np.dtype('S1')),
    'SHORT': dict(tag=3, size=2,
                  python=lambda x, bo: int.from_bytes(x, byteorder=bo_long(bo)),
                  numpy=lambda bo: np.dtype(bo_short(bo) + 'u2')),
    'LONG': dict(tag=4, size=4,
                 python=lambda x, bo: int.from_bytes(x, byteorder=bo_long(bo)),
                 numpy=lambda bo: np.dtype(bo_short(bo) + 'u4')),
    'RATIONAL': dict(tag=5, size=4,
                     python=None, numpy=rational),
    'SBYTE': dict(tag=6, size=1,
                  python=lambda x, _: bytes(x),
                  numpy=lambda _: np.dtype('s1')),
    'UNDEFINED': dict(tag=7, size=1,
                      python=lambda x, _: bytes(x),
                      numpy=lambda _: np.bytes),
    'SSHORT': dict(tag=8, size=2,
                   python=lambda x, bo: int.from_bytes(x, byteorder=bo_long(bo)),
                   numpy=lambda bo: np.dtype(bo_short(bo) + 's2')),
    'SLONG': dict(tag=9, size=4,
                  python=lambda x, bo: int.from_bytes(x, byteorder=bo_long(bo)),
                  numpy=lambda bo: np.dtype(bo_short(bo) + 's4')),
    'SRATIONAL': dict(tag=10, size=4,
                      python=None, numpy=srational),
    'FLOAT': dict(tag=11, size=4,
                  python=lambda x, bo: struct.unpack(x, bo_short(bo) + 'f')[0],
                  numpy=lambda bo: np.dtype(bo_short(bo) + 'f4')),
    'DOUBLE': dict(tag=12, size=8,
                   python=lambda x, bo: struct.unpack(x, bo_short(bo) + 'd')[0],
                   numpy=lambda bo: np.dtype(bo_short(bo) + 'f8')),
    'IFD': dict(tag=13, size=4,
                python=lambda x, bo: int.from_bytes(x, byteorder=bo_long(bo)),
                numpy=lambda bo: np.dtype(bo_short(bo) + 'u4')),
    # BigTIFF
    'LONG8': dict(tag=16, size=8,
                  python=lambda x, bo: int.from_bytes(x, byteorder=bo_long(bo)),
                  numpy=lambda bo: np.dtype(bo_short(bo) + 'u8')),
    'SLONG8': dict(tag=17, size=8,
                   python=lambda x, bo: int.from_bytes(x, byteorder=bo_long(bo)),
                   numpy=lambda bo: np.dtype(bo_short(bo) + 's8')),
    'IFD8': dict(tag=18, size=8,
                 python=lambda x, bo: int.from_bytes(x, byteorder=bo_long(bo)),
                 numpy=lambda bo: np.dtype(bo_short(bo) + 'u8')),
}

field_type = enum.Enum('field_type',
                       {key: value['tag'] for key, value in types.items()},
                       module=__name__)


def int_to_type(x):
    try:
        name = field_type(x).name
        return name, types[name]
    except ValueError:
        # We don't know this tag but should not fail
        return x, None


tags = {
    # ------------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------------
    'NewSubfileType': dict(tag=254, type=field_type.LONG, count=1, default=0),
    'SubfileType': dict(tag=255, type=field_type.SHORT, count=1),
    'ImageWidth': dict(tag=256, type=(field_type.SHORT, field_type.LONG), count=1),
    'ImageLength': dict(tag=257, type=(field_type.SHORT, field_type.LONG), count=1),
    'BitsPerSample': dict(tag=258, type=field_type.SHORT, default=1),
    'Compression': dict(tag=259, type=field_type.SHORT, count=1, default=1),
    'PhotometricInterpretation': dict(tag=262, type=field_type.SHORT, count=1),
    'Threshholding': dict(tag=263, type=field_type.SHORT, count=1, default=1),
    'CellWidth': dict(tag=264, type=field_type.SHORT, count=1),
    'CellLength': dict(tag=265, type=field_type.SHORT, count=1),
    'FillOrder': dict(tag=266, type=field_type.SHORT, count=1, default=1),
    'ImageDescription': dict(tag=270, type=field_type.ASCII),
    'Make': dict(tag=271, type=field_type.ASCII),
    'Model': dict(tag=272, type=field_type.ASCII),
    'StripOffsets': dict(tag=273, type=(field_type.SHORT, field_type.LONG, field_type.LONG8)),
    'Orientation': dict(tag=274, type=field_type.SHORT, count=1, default=1),
    'SamplesPerPixel': dict(tag=277, type=field_type.SHORT, count=1, default=1),
    'RowsPerStrip': dict(tag=278, type=(field_type.SHORT, field_type.LONG), count=1, default=2**32-1),
    'StripByteCounts': dict(tag=279, type=(field_type.SHORT, field_type.LONG, field_type.LONG8)),
    'MinSampleValue': dict(tag=281, type=field_type.SHORT, default=0),
    'MaxSampleValue': dict(tag=281, type=field_type.SHORT),
    'XResolution': dict(tag=282, type=field_type.RATIONAL, count=1),
    'YResolution': dict(tag=283, type=field_type.RATIONAL, count=1),
    'PlanarConfiguration': dict(tag=284, type=field_type.SHORT, count=1, default=1),
    'FreeOffsets': dict(tag=288, type=field_type.LONG),
    'FreeByteCounts': dict(tag=289, type=field_type.LONG),
    'GrayResponseUnit': dict(tag=290, type=field_type.SHORT, default=2),
    'GrayResponseCurve': dict(tag=291, type=field_type.SHORT),
    'ResolutionUnit': dict(tag=296, type=field_type.SHORT, count=1, default=2),
    'Software': dict(tag=305, type=field_type.ASCII),
    'DateTime': dict(tag=306, type=field_type.ASCII, count=20),
    'Artist': dict(tag=315, type=field_type.ASCII),
    'HostComputer': dict(tag=316, type=field_type.ASCII),
    'ColorMap': dict(tag=320, type=field_type.SHORT, count=1),
    'ExtraSamples': dict(tag=338, type=field_type.SHORT),
    'Copyright': dict(tag=33432, type=field_type.ASCII),
    # ------------------------------------------------------------------
    # Extension
    # ------------------------------------------------------------------
    'DocumentName': dict(tag=269, type=field_type.ASCII),
    'PageName': dict(tag=285, type=field_type.ASCII),
    'XPosition': dict(tag=286, type=field_type.RATIONAL, count=1),
    'YPosition': dict(tag=287, type=field_type.RATIONAL, count=1),
    'T4Options': dict(tag=292, type=field_type.LONG, count=1, default=0),
    'T6Options': dict(tag=293, type=field_type.LONG, count=1, default=0),
    'PageNumber': dict(tag=297, type=field_type.SHORT, count=2),
    'TransferFunction': dict(tag=301, type=field_type.SHORT),
    'Predictor': dict(tag=317, type=field_type.SHORT, count=1, default=1),
    'WhitePoint': dict(tag=318, type=field_type.RATIONAL, count=2),
    'PrimaryChromaticities': dict(tag=319, type=field_type.RATIONAL, count=6),
    'HalftoneHints': dict(tag=321, type=field_type.SHORT, count=2),
    'TileWidth': dict(tag=322, type=(field_type.SHORT, field_type.LONG), count=1),
    'TileLength': dict(tag=323, type=(field_type.SHORT, field_type.LONG), count=1),
    'TileOffsets': dict(tag=324, type=(field_type.LONG, field_type.LONG8)),
    'TileByteCounts': dict(tag=325, type=(field_type.SHORT, field_type.LONG, field_type.LONG8)),
    'BadFaxLines': dict(tag=326, type=(field_type.SHORT, field_type.LONG), count=1),
    'CleanFaxData': dict(tag=327, type=field_type.SHORT, count=1),
    'ConsecutiveBadFaxLines': dict(tag=328, type=(field_type.SHORT, field_type.LONG), count=1),
    'SubIFDs': dict(tag=330, type=(field_type.LONG, field_type.IFD, field_type.LONG8, field_type.IFD8)),
    'InkSet': dict(tag=332, type=field_type.SHORT, count=1, default=1),
    'InkNames': dict(tag=333, type=field_type.ASCII),
    'NumberOfInks': dict(tag=334, type=field_type.SHORT, count=1, default=4),
    'DotRange': dict(tag=336, type=(field_type.BYTE, field_type.SHORT)),
    'TargetPrinter': dict(tag=337, type=field_type.ASCII),
    'SampleFormat': dict(tag=339, type=field_type.SHORT, default=1),
    'SMinSampleValue': dict(tag=340, type=(field_type.BYTE, field_type.SHORT, field_type.LONG, field_type.RATIONAL, field_type.DOUBLE)),
    'SMaxSampleValue': dict(tag=341, type=(field_type.BYTE, field_type.SHORT, field_type.LONG, field_type.RATIONAL, field_type.DOUBLE)),
    'TransferRange': dict(tag=342, type=field_type.SHORT, count=6),
    'ClipPath': dict(tag=343, type=field_type.BYTE),
    'XClipPathUnits': dict(tag=344, type=field_type.LONG, count=1),
    'YClipPathUnits': dict(tag=345, type=field_type.LONG, count=1),
    'Indexed': dict(tag=346, type=field_type.SHORT, count=1, default=0),
    'JPEGTables': dict(tag=347, type=field_type.UNDEFINED),
    'OPIProxy': dict(tag=351, type=field_type.SHORT, count=1, default=0),
    'GlobalParametersIFD': dict(tag=400, type=(field_type.LONG, field_type.IFD)),
    'ProfileType': dict(tag=401, type=field_type.LONG, count=1),
    'FaxProfile': dict(tag=402, type=field_type.BYTE, count=1),
    'CodingMethods': dict(tag=403, type=field_type.LONG, count=1),
    'VersionYear': dict(tag=404, type=field_type.BYTE, count=4),
    'ModeNumber': dict(tag=405, type=field_type.BYTE, count=1),
    'Decode': dict(tag=433, type=field_type.SRATIONAL),
    'DefaultImageColor': dict(tag=434, type=field_type.SHORT),
    'JPEGProc': dict(tag=512, type=field_type.SHORT, count=1),
    'JPEGInterchangeFormat': dict(tag=513, type=field_type.LONG, count=1),
    'JPEGInterchangeFormatLength': dict(tag=514, type=field_type.LONG, count=1),
    'JPEGRestartInterval': dict(tag=515, type=field_type.SHORT, count=1),
    'JPEGLosslessPredictors': dict(tag=517, type=field_type.SHORT),
    'JPEGPointTransforms': dict(tag=518, type=field_type.SHORT),
    'JPEGQTables': dict(tag=519, type=field_type.LONG),
    'JPEGDCTables': dict(tag=520, type=field_type.LONG),
    'JPEGACTables': dict(tag=521, type=field_type.LONG),
    'YCbCrCoefficients': dict(tag=529, type=field_type.RATIONAL, count=3),
    'YCbCrSubSampling': dict(tag=530, type=field_type.SHORT, count=2, default=2),
    'YCbCrPositioning': dict(tag=531, type=field_type.SHORT, count=1, default=1),
    'ReferenceBlackWhite': dict(tag=532, type=field_type.RATIONAL, count=6),
    'StripRowCounts': dict(tag=559, type=field_type.LONG),
    'XMP': dict(tag=700, type=field_type.BYTE),
    'ImageID': dict(tag=32781, type=field_type.ASCII),
    'ImageLayer': dict(tag=34732, type=(field_type.SHORT, field_type.LONG), count=2),
    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------
    # TODO: some of the GeoTIFF tags may be useful
    #   https://www.awaresystems.be/imaging/tiff/tifftags/private.html
}

field_tag = enum.Enum('field_tag',
                      {key: value['tag'] for key, value in tags.items()},
                      module=__name__)


def int_to_tag(x):
    try:
        name = field_tag(x).name
        return name, tags[name]
    except ValueError:
        # We don't know this tag but should not fail
        return x, None


compressions = {
    'No': dict(tag=1, codec=None),
    'HUFF': dict(tag=2, codec=None),
    'T4': dict(tag=3, codec=None),
    'T6': dict(tag=4, codec=None),
    'LZW': dict(tag=5, codec=None),
    'OJPEG': dict(tag=6, codec=None),
    'JPEG': dict(tag=7, codec=None),
    'ADOBE_DEFLATE': dict(tag=8, codec=None),
    'JBIGBW': dict(tag=9, codec=None),
    'JBIGC': dict(tag=10, codec=None),
    'LEW': dict(tag=32771, codec=None),
    'PackBits': dict(tag=32773, codec=None),
    'THUNDERSCAN': dict(tag=32809, codec=None),
    'IT8CTPAD': dict(tag=32895, codec=None),
    'IT8LW': dict(tag=32896, codec=None),
    'IT8P': dict(tag=32897, codec=None),
    'IT8BL': dict(tag=32898, codec=None),
    'PIXARFILM': dict(tag=32908, codec=None),
    'PIXARLOG': dict(tag=32909, codec=None),
    'DEFLATE': dict(tag=32946, codec=None),
    'DCS': dict(tag=32947, codec=None),
    'JBIG': dict(tag=34661, codec=None),
    'SGILOG': dict(tag=34676, codec=None),
    'SGILOG24': dict(tag=34677, codec=None),
    'JP2000': dict(tag=34712, codec=None),
    'LOSSYJPEG': dict(tag=34892, codec=None),
}
