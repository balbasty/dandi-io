import json
import fsspec
import os
from . import utils
from ..dandifs import RemoteDandiFileSystem
from ..tracts import TractsTRK


class CONFIG:
    class DEFAULT:
        NGV = 'https://neuroglancer-demo.appspot.com'
        IP = '127.0.0.1'
        PORT = '8000'
        SRV = f'http://{IP}:{PORT}'
    BASE_URL = DEFAULT.SRV
    NGV_URL = DEFAULT.NGV


READERS = {}


def get_reader(fname):
    """Get or instantiate a TRK reader"""
    if fname not in READERS:
        if fname.startswith('dandi://'):
            with RemoteDandiFileSystem().open(fname) as f:
                reader = TractsTRK(f)
        elif fname.startswith(('http://', 'https://')):
            with fsspec.open(fname) as f:
                reader = TractsTRK(f)
        else:
            reader = TractsTRK(fname)
        READERS[fname] = reader
        # TODO: implement LRU cache ejection here
    else:
        reader = READERS[fname]
        # TODO: update cache entry age here
    return reader


def serve_precomputed(environ, start_response):
    # We expect PATH_INFO to be a path to a file on disk.
    # If that file is a TRK file, the path can be appended with:
    #   /path/to/file.trk/info      -> we will generate a NG precomp skeleton JSON
    #   /path/to/file.trk/prop/info -> we will generate a NG precomp propertiy JSON
    #   /path/to/file.trk/<int>     -> we will generate the corresponding NG precomp skeleton
    path_info = environ["PATH_INFO"]
    print(path_info)

    if path_info == "/":
        # TODO:
        #   Generate an index page with links to neuroglancer instances.
        #   For example, we could browse through the dandi instance
        #   to find files that live in the same space and generate a link
        #   that loads them and renders them nicely in neuroglancer.
        return utils.file_not_found(path_info, start_response)

    if not path_info.startswith("/"):
        # we expect a fullpath to a file on disk
        return utils.file_not_found(path_info, start_response)

    if path_info.split('/')[-1] == 'info':
        if path_info.split('/')[-2] == 'prop':
            fname = '/'.join(path_info.split('/')[:-2])
            return serve_prop_info(environ, start_response, fname)
        else:
            fname = '/'.join(path_info.split('/')[:-1])
            return serve_skel_info(environ, start_response, fname)

    if path_info.endswith(('.nii', '.nii.gz')):
        # TODO:
        #   Any format that neuroglancer knows how to render  natively
        #   should be returned as is.
        #   Right now we only do this for niftis.
        #   Note that zarr will probably need to be handled by our own
        #   (TODO) wrapper, because the format currently does not handle
        #   advanced orientation metadata (only translations and scales).
        #   We'll have to read the full transform from the sidecar JSON.
        with open(path_info, 'rb') as f:
            data = f.read()
    else:
        try:
            fname = path_info.split('/')
            tract_id = int(fname[-1])
            fname = '/'.join(fname[:-1])
            while fname.endswith('/'):
                fname = fname[:-1]
            if not os.path.exists(fname):
                raise FileNotFoundError
        except Exception:
            return utils.file_not_found(path_info, start_response)

        reader = get_reader(fname)
        data = reader.precomputed_skel_tract_combined(tract_id)

    start_response(
        "200 OK",
        [
            ("Content-type", "application/octet-stream"),
            ("Content-Length", str(len(data))),
            ("Access-Control-Allow-Origin", "*"),
        ],
    )
    return [data]


def serve_skel_info(environ, start_response, path):
    reader = get_reader(path)
    info = reader.precomputed_skel_info()
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


def serve_prop_info(environ, start_response, path):
    reader = get_reader(path)
    info = reader.precomputed_prop_info(combined=True)
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
        default=CONFIG.DEFAULT.IP,
    )
    parser.add_argument(
        "--port",
        help="Port to bind to",
        default=CONFIG.DEFAULT.PORT,
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
        "--ng",
        help="Launch a neuroglancer instance",
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "--ng-token",
        help="Neuroglancer token (default: random)",
        default=None,
    )
    parser.add_argument(
        "--ng-port",
        help="Neuroglancer token (default: random)",
        default=0,
        type=int,
    )
    opts = parser.parse_args((argv or sys.argv)[1:])

    CONFIG.BASE_URL = f"http://{opts.ip}:{opts.port}"
    proxy = f"https://hub.dandiarchive.org/user/{os.environ.get('GITHUB_USER', 'anon')}/proxy"
    if opts.proxy:
        CONFIG.BASE_URL = f"{proxy}/{opts.port}"
    print('file index:', CONFIG.BASE_URL)
    if opts.ng:
        import neuroglancer
        from neuroglancer.server import global_server_args
        global_server_args['bind_port'] = opts.ng_port
        ngv = neuroglancer.Viewer(token=opts.ng_token)
        CONFIG.NGV_URL = ngv.get_viewer_url()
        if opts.proxy:
            CONFIG.NGV_URL = CONFIG.NGV_URL.replace("http://127.0.0.1:", proxy)
        print('neuroglancer:', CONFIG.NGV_URL)

    def application(environ, start_response):
        return serve_precomputed(environ, start_response)

    httpd = make_server(opts.ip, opts.port, application)
    httpd.serve_forever()


if __name__ == "__main__":
    webserver()
