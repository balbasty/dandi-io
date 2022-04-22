"""Webserver module for mod_wsgi in Apache (e.g.) serving DANDI volumes

To use, create a virtualenv with precomputed-tif installed. Create a json
file containing a list of sources to serve. Each source should have a
"name" key, a "directory" key and a "format" key, e.g.:
[
   {
       "name": "expt1-dapi",
       "urls": [
          "file://path-to/foo_chunk-1_spim.ngff",
          "file://path-to/foo_chunk-2_spim.ngff"
            ]
   },
   {
       "name": "expt1-phalloidin",
       "urls": [
          "file://path-to/bar_chunk-1_spim.ngff",
          "file://path-to/bar_chunk-2_spim.ngff"
            ]
   }
]

Create a Python file for to serve via wsgi, e.g.

from precomputed_tif.dandi_webserver import serve_precomputed

CONFIG_FILE = "/etc/precomputed.config"

def application(environ, start_response):
    return serve_precomputed(environ, start_response, config_file)

"""

import json
import math
import urllib
from urllib.parse import quote

from dandiio.array import DANDIArrayReader


class ParseFileException(BaseException):
    pass


readers = {}


def file_not_found(dest, start_response):
    start_response(
        "404 Not found",
        [("Content-type", "text/html"), ("Access-Control-Allow-Origin", "*")],
    )
    return [("<html><body>%s not found</body></html>" % dest).encode("utf-8")]


shader_template = """
#uicontrol float brightness slider(min=0.0, max=100.0, default=%f)
void main() {
    float x = clamp(toNormalized(getDataValue()) * brightness, 0.0, 1.0);
    float angle = 2.0 * 3.1415926 * (4.0 / 3.0 + x);
    float amp = x * (1.0 - x) / 2.0;
    vec3 result;
    float cosangle = cos(angle);
    float sinangle = sin(angle);
    result.r = -0.14861 * cosangle + 1.78277 * sinangle;
    result.g = -0.29227 * cosangle + -0.90649 * sinangle;
    result.b = 1.97294 * cosangle;
    result = clamp(x + amp * result, 0.0, 1.0);
    emitRGB(result);
}
"""


def list_one(key, ng_url, url):
    layer = dict(
        source=f"precomputed://{url}",
        type="image",
        shader=shader_template % 40,
        name=key,
    )
    ng_str = json.dumps(dict(layers=[layer]))
    url = f"{ng_url}#!%s" % quote(ng_str)

    return url  # '<li><a href="%s">%s</a></li>' % (url, key)


def neuroglancer_listing(start_response, config):
    result = "<html><body><ul>\n"

    def sort_fn(d):
        return d["name"]

    for d in sorted(config, key=sort_fn):
        result += list_one(d["name"]) + "\n"
    result += "</ul></body></html>"
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


def serve_precomputed(environ, start_response, config_file):
    with open(config_file) as fd:
        spec = json.load(fd)
    path_info = environ["PATH_INFO"]
    print(f"path: {path_info}")
    if path_info == "/":
        data = "Nothing here".encode("ascii")
        start_response(
            "200 OK",
            [
                ("Content-type", "text/html"),
                ("Content-Length", str(len(data))),
                ("Access-Control-Allow-Origin", "*"),
            ],
        )
        return [data]
    elif path_info.startswith("/"):
        filename = path_info.lstrip("/")
        print(f"file: {filename}")
        if filename == "info":
            return serve_info(environ, start_response, spec)
        else:
            try:
                level, x0, x1, y0, y1, z0, z1 = parse_filename(filename)
            except ParseFileException:
                return file_not_found(path_info, start_response)
            print(level, x0, x1, y0, y1, z0, z1)
            if level not in readers:
                ar = DANDIArrayReader(spec, level=level)
                readers[level] = ar
            else:
                ar = readers[level]
            img = ar[z0:z1, y0:y1, x0:x1]
            data = img.tobytes()  # tostring("C")
            start_response(
                "200 OK",
                [
                    ("Content-type", "application/octet-stream"),
                    ("Content-Length", str(len(data))),
                    ("Access-Control-Allow-Origin", "*"),
                ],
            )
            return [data]
    else:
        return file_not_found(path_info, start_response)


def serve_info(environ, start_response, spec):
    level = 1
    if level not in readers:
        ar = DANDIArrayReader(spec, level=level)
        readers[level] = ar
    else:
        ar = readers[level]
    info = ar.get_info()
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


def parse_filename(filename):
    try:
        level, path = filename.split("/")
        level = int(level.split("_")[0])
        xstr, ystr, zstr = path.split("_")
        x0, x1 = [int(x) for x in xstr.split("-")]
        y0, y1 = [int(y) for y in ystr.split("-")]
        z0, z1 = [int(z) for z in zstr.split("-")]
    except ValueError:
        raise ParseFileException()
    return level, x0, x1, y0, y1, z0, z1


if __name__ == "__main__":
    import argparse
    import sys
    from wsgiref.simple_server import make_server

    parser = argparse.ArgumentParser()
    parser.add_argument("config_filename", help="File with the DANDI sources")
    """
    parser.add_argument(
        "ng_url",
        help="Neuroglancer url on the hub"
    )
    """
    parser.add_argument(
        "--ip-address",
        help="IP address or dns name of interface to bind to",
        default="127.0.0.1",
    )
    parser.add_argument("--port", help="Port to bind to", default=8000, type=int)
    parser.add_argument(
        "--proxy",
        help="Whether to use hub proxy",
        action=argparse.BooleanOptionalAction,
        default=True,
        type=bool,
    )
    opts = parser.parse_args(sys.argv[1:])

    def application(environ, start_response):
        return serve_precomputed(environ, start_response, opts.config_filename)

    import os

    import neuroglancer

    url = f"http://{opts.ip_address}:{opts.port}/"
    if opts.proxy:
        url = f"https://hub.dandiarchive.org/user/{os.environ['GITHUB_USER']}/proxy/{opts.port}"
    ngv = neuroglancer.Viewer()
    ng_url = ngv.get_viewer_url()
    if opts.proxy:
        ng_url = ng_url.replace("http://127.0.0.1:", url)

    print(list_one("test", ng_url, url))
    httpd = make_server(opts.ip_address, opts.port, application)
    httpd.serve_forever()
