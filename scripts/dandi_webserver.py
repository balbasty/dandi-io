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

import typing

import requests

from dandiio.array import DANDIArrayReader
from dandiio.dandifs import RemoteDandiFileSystem

class ParseFileException(BaseException):
    pass


readers = {}
specs = {}
rfs:typing.Dict[typing.Tuple[str, str], RemoteDandiFileSystem] = {}


def file_not_found(dest, start_response):
    start_response(
        "404 Not found",
        [("Content-type", "text/html"), ("Access-Control-Allow-Origin", "*")],
    )
    return [("<html><body>%s not found</body></html>" % dest).encode("utf-8")]


cubehelix_template = """
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

# TODO: make these colorblind-aware
#       blue is pretty yucky
#
red_template = """
#uicontrol float brightness slider(min=0.0, max=100.0, default=%.2f)
void main() {
   emitRGB(vec3(brightness * toNormalized(getDataValue()), 0, 0));
}
"""

green_template = """
#uicontrol float brightness slider(min=0.0, max=100.0, default=%.2f)
void main() {
   emitRGB(vec3(0, brightness * toNormalized(getDataValue()), 0));
}
"""

blue_template = """
#uicontrol float brightness slider(min=0.0, max=100.0, default=%.2f)
void main() {
   emitRGB(vec3(0, 0, brightness * toNormalized(getDataValue())));
}
"""


def make_url(keys):
    layers = []
    if len(keys) == 1:
        colors = [cubehelix_template]
    else:
        colors = [red_template, green_template, blue_template] * ((len(keys) + 2) // 3)
        colors = colors[:len(keys)]

    for (subject, sample, stain), color in zip(keys, colors):
        layer = dict(
            source=f"precomputed://{base_url}{subject}/{sample}/{stain}",
            type="image",
            shader=color % 40,
            name=stain
        )
        layers.append(layer)
    ng_str = json.dumps(dict(layers=layers))
    url = f"{ng_url}#!%s" % quote(ng_str)
    return url  # '<li><a href="%s">%s</a></li>' % (url, key)


def neuroglancer_listing(start_response, config):
    result = "<html><body><ul>\n"

    def sort_fn(d):
        return d["name"]
    with open(config) as fd:
        tree = json.load(fd)["tree"]
    for subject in sorted(tree.keys()):
        result += f"  <li>subject\n    <ul>\n"
        for sample in sorted(tree[subject].keys()):
            result += f"      <li>{sample}\n        <ul>\n"
            stains = sorted(tree[subject][sample].keys())
            # i is a bit pattern where each bit is a stain present or absent
            # this lets us iterate through all of them.
            # obviously we want to add html magic to make the list auto expand and such
            for i in range(1, 2 ** len(stains)):
                keys = [(subject, sample, stain) for j, stain in enumerate(stains)
                        if 2 ** j & i]
                url = make_url(keys)
                result += f'          <li><a href="{url}">{"+".join([_[2] for _ in keys])}</a></li>\n'
            result += "        </ul>\n"
            result += "      </li>\n"
        result += "     </ul>\n"
        result += "    </li>\n"
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


def make_spec(config, subject, sample, stain):
    spec_key = (subject, sample, stain)
    if spec_key not in specs:
        print(f"Constructing spec for {spec_key}")
        dandiset = config["dandiset"]
        version = config["version"]
        if (dandiset, version) not in rfs:
            rfs[dandiset, version] = RemoteDandiFileSystem(dandiset, version)
        my_rfs = rfs[dandiset, version]
        sub_spec_in = config["tree"][subject][sample][stain]
        sub_spec_out = {}
        for key, value in sub_spec_in.items():
            try:
                # doesn't work sometimes.
                url = my_rfs.s3_url(value["path"])
            except:
                request_url = f"https://api.dandiarchive.org/api/dandisets/{dandiset}/versions/{version}/assets/{value}/"
                urls = requests.get(request_url).json()["contentUrl"]
                url = [_ for _ in urls if ".s3." in _].pop()
            sub_spec_out[key] = [url]
        specs[key] = sub_spec_out
    return specs[key]


def serve_precomputed(environ, start_response, config_file):
    with open(config_file) as fd:
        config = json.load(fd)
    spec = config["tree"]
    path_info = environ["PATH_INFO"]
    print(f"path: {path_info}")
    if path_info == "/":
        data = neuroglancer_listing(start_response, config_file)
        return data
    elif path_info.startswith("/"):
        try:
            subject, sample, stain, filename = path_info[1:].split("/", 3)
        except ValueError:
            return file_not_found(path_info, start_response)
        print(f"subject: {subject}, sample: {sample}, stain: {stain}, file: {filename}")
        try:
            sub_spec = make_spec(config, subject, sample, stain)
        except ValueError:
            return file_not_found(path_info, start_response)
        if filename == "info":
            return serve_info(environ, start_response, subject, sample, stain, sub_spec)
        else:
            try:
                level, x0, x1, y0, y1, z0, z1 = parse_filename(filename)
            except ParseFileException:
                return file_not_found(path_info, start_response)
            print(level, x0, x1, y0, y1, z0, z1)
            reader_key = (subject, sample, stain, level)
            if reader_key not in readers:
                ar = DANDIArrayReader(sub_spec, level=level)
                readers[reader_key] = ar
                # TODO: implement LRU cache ejection here
            else:
                ar = readers[reader_key]
                # TODO: update cache entry age here
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


def serve_info(environ, start_response, subject, sample, stain, spec):
    level = 1
    key = (subject, sample, stain, level)
    if key not in readers:
        ar = DANDIArrayReader(spec, level=level)
        readers[key] = ar
    else:
        ar = readers[key]
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

    import os

    import neuroglancer

    base_url = f"http://{opts.ip_address}:{opts.port}/"
    if opts.proxy:
        base_url = f"https://hub.dandiarchive.org/user/{os.environ['GITHUB_USER']}/proxy/{opts.port}/"
    ngv = neuroglancer.Viewer()
    ng_url = ngv.get_viewer_url()
    if opts.proxy:
        ng_url = ng_url.replace("http://127.0.0.1:", f"https://hub.dandiarchive.org/user/{os.environ['GITHUB_USER']}/proxy/")

    def application(environ, start_response):
        return serve_precomputed(environ, start_response, opts.config_filename)
    print(make_url((("MITU01", "125", "LEC"),)))
    print(base_url)
    httpd = make_server(opts.ip_address, opts.port, application)
    httpd.serve_forever()
