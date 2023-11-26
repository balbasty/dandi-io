import re
import json
from urllib.parse import quote
from . import utils
from ..dandifs import RemoteDandiFileSystem
from ..dandi26 import StitchedLSFM


DEFAULT_DANDISET = '000026'
DEFAULT_PREFIX = f'dandi://dandi/{DEFAULT_DANDISET}'
DEFAULT_NGV = 'https://neuroglancer-demo.appspot.com'
DEFAULT_IP = '127.0.0.1'
DEFAULT_PORT = '8000'
DEFAULT_SRV = f'http://{DEFAULT_IP}:{DEFAULT_PORT}'

READERS = {}


class CONFIG:
    BASE_URL = DEFAULT_SRV
    NGV_URL = DEFAULT_NGV


def get_reader(fname):
    if fname not in READERS:
        reader = StitchedLSFM(f'{DEFAULT_PREFIX}/{fname}')
        READERS[fname] = reader
        # TODO: implement LRU cache ejection here
    else:
        reader = READERS[fname]
        # TODO: update cache entry age here
    return reader


def is_json(fname):
    return fname.endswith('.json')


def isnot_json(fname):
    return not is_json(fname)


def make_index(start_response):
    fs = RemoteDandiFileSystem('000026')

    result = '<html><body>\n<ul>\n'
    for sub in fs.ls('/'):
        if not sub.startswith('sub-'):
            continue
        result += f'\t<li>{sub}\n'
        result += '\t<ul>\n'
        for ses in fs.ls(sub):
            ses0 = ses.split('/')[-1]
            result += f'\t\t<li>{ses0}\n'
            result += '\t\t<ul>\n'
            for mod in fs.ls(ses):
                mod0 = mod.split('/')[-1]
                result += f'\t\t\t<li>{mod0}\n'
                result += '\t\t\t<ul>\n'
                all_files = list(sorted(filter(isnot_json, fs.ls(mod))))
                all_files = [re.sub(r'chunk-\d\d', r'chunk-*', fname)
                             for fname in all_files]
                for fname in all_files:
                    fname0 = fname.split('/')[-1]
                    url = make_url(fname)
                    if not url:
                        pass
                    result += f'\t\t\t\t<li><a href="{url}">{fname0}</a></li>\n'
                result += '\t\t\t</ul>\n'
                result += '\t\t\t</li>\n'
            result += '\t\t</ul>\n'
            result += '\t\t</li>\n'
        result += '\t</ul>\n'
        result += '\t</li>\n'
    result += '</ul>\n</body></html>'

    data = result.encode("ascii")
    start_response(
        "200 OK",
        [
            ("Content-type", "text/html"),
            ("Content-Length", str(len(data))),
            ("Access-Control-Allow-Origin", "*"),
        ],
    )
    return [data]


def make_url(path):
    if path.endswith('tif'):
        layer = dict(
            source=f"precomputed://{CONFIG.BASE_URL}/{path}",
            type="image",
            shader=utils.ubehelix_template % 40,
            name=path.split('/')[-1]
        )
    elif path.endswith('zarr'):
        url = RemoteDandiFileSystem(DEFAULT_DANDISET).s3_url(path)
        layer = dict(
            source=f"zarr://{url}",
            type="image",
            shader=utils.ubehelix_template % 40,
            name=path.split('/')[-1]
        )
    elif path.endswith(['nii', 'nii.gz']):
        url = RemoteDandiFileSystem(DEFAULT_DANDISET).s3_url(path)
        layer = dict(
            source=f"nifti://{url}",
            type="image",
            shader=utils.ubehelix_template % 40,
            name=path.split('/')[-1]
        )
    else:
        # format not handled for now
        return None

    ng_str = json.dumps(dict(layers=[layer]))
    url = f'{CONFIG.NGV_URL}/#!{quote(ng_str)}'
    return url


def serve_precomputed(environ, start_response):
    path_info = environ["PATH_INFO"]

    if path_info == "/":
        return make_index(start_response)

    if not path_info.startswith("/"):
        return utils.file_not_found(path_info, start_response)

    if path_info.split('/')[-1] == 'info':
        fname = '/'.join(path_info.split('/')[:-1])
        return serve_info(environ, start_response, fname)

    try:
        fname = path_info.split('/')
        fov_info = '/'.join(fname[-2:])
        fname = '/'.join(fname[:-2])
        level, x0, x1, y0, y1, z0, z1 = utils.parse_filename(fov_info)
    except utils.ParseFileException:
        return utils.file_not_found(path_info, start_response)

    step = 2**level
    reader = get_reader(fname)
    data = reader[x0:x1:step, y0:y1:step, z0:z1:step]

    data = data.tobytes('F')
    start_response(
        "200 OK",
        [
            ("Content-type", "application/octet-stream"),
            ("Content-Length", str(len(data))),
            ("Access-Control-Allow-Origin", "*"),
        ],
    )
    return [data]


def serve_info(environ, start_response, path):
    reader = get_reader(path)
    info = reader.get_info()
    data = json.dumps(info, indent=2, ensure_ascii=True).encode("ascii")

    start_response(
        "200 OK",
        [
            ("Content-type", "application/json"),
            ("Content-Length", str(len(data))),
            ("Access-Control-Allow-Origin", "*"),
        ],
    )
    return [data]


def webserver(argv=None):
    import argparse
    import sys
    import os
    from wsgiref.simple_server import make_server

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ip",
        help="IP address or dns name of interface to bind to",
        default=DEFAULT_IP,
    )
    parser.add_argument(
        "--port",
        help="Port to bind to",
        default=DEFAULT_PORT,
        type=int,
    )
    parser.add_argument(
        "--proxy",
        help="Use hub proxy",
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "--no-proxy",
        help="Do not use hub proxy",
        action='store_false',
        default=False,
        dest='proxy',
    )
    parser.add_argument(
        "--neuroglancer",
        help="Launch a neuroglancer instance",
        action='store_true',
        default=False,
    )
    opts = parser.parse_args((argv or sys.argv)[1:])

    CONFIG.BASE_URL = f"http://{opts.ip}:{opts.port}"
    proxy = f"https://hub.dandiarchive.org/user/{os.environ['GITHUB_USER']}/proxy"
    if opts.proxy:
        CONFIG.BASE_URL = f"{proxy}/{opts.port}"
    print('file index:', CONFIG.BASE_URL)
    if opts.neuroglancer:
        import neuroglancer
        ngv = neuroglancer.Viewer()
        CONFIG.NGV_URL = ngv.get_viewer_url()
        if opts.proxy:
            CONFIG.NGV_URL = CONFIG.NGV_URL.replace("http://127.0.0.1:", proxy)
        print('neuroglancer:', CONFIG.NGV_URL)

    def application(environ, start_response):
        return serve_precomputed(environ, start_response, opts.config_filename)

    httpd = make_server(opts.ip, opts.port, application)
    httpd.serve_forever()


if __name__ == "__main__":
    webserver()
