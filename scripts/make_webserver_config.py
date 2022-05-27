import argparse
import json

import requests
import re
import sys

# TODO: use RemoteDandiFileSystem when it supports regex
def parse_args(args = sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dandiset",
                        help="The dandiset to navigate",
                        required=True)
    parser.add_argument("--version",
                        help="The version of the dandiset",
                        default="draft")
    parser.add_argument("--output",
                        help="The output file for the spec",
                        required=True)
    return parser.parse_args(args)


def parse_path(path):
    pattern = "sub-(?P<subject>[^_]+)_ses-(?P<session>[^_]+)_sample-(?P<sample>[^_]+)_stain-(?P<stain>[^_]+)_run-[^_]+_chunk-(?P<chunk>[^_]+)"
    filename = path.split("/")[-1]
    return re.search(pattern, filename).groupdict()


def tadd(tree, key):
    if key not in tree:
        tree[key] = {}
    return tree[key]


def get_all_ngffs(dandiset, version):
    url = f"https://api.dandiarchive.org/api/dandisets/{dandiset}/versions/{version}/assets/?regex=.%2B%5C.ome.zarr"
    answer = requests.get(url).json()
    results = answer["results"]
    while "next" in answer and answer["next"]:
        answer = requests.get(answer["next"]).json()
        results += answer["results"]
    tree = {}
    for result in results:
        path = result["path"]
        try:
            keys = parse_path(path)
            ksubject = keys["subject"]
            subject = tadd(tree, ksubject)
            ksample = keys["sample"]
            sample = tadd(subject, ksample)
            kstain = keys["stain"]
            stain = tadd(sample, kstain)
            stain[f'{ksubject}/{ksample}/{kstain}/{keys["chunk"]}'] = result["asset_id"]
        except AttributeError:
            print(f"Failed to parse {path}", file=sys.stderr)
        except:
            print(f"Failed to look up asset for {result['asset_id']}")
    return tree


def main():
    opts = parse_args()
    tree = get_all_ngffs(opts.dandiset, opts.version)
    with open(opts.output, "w") as fd:
        json.dump(dict(dandiset=opts.dandiset,
                       version=opts.version,
                       tree=tree), fd, indent=2)


if __name__ == "__main__":
    main()
