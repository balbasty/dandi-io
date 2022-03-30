from fsspec.implementations.http import HTTPFileSystem
from fsspec.utils import stringify_path
from fsspec.asyn import sync_wrapper
from dandi.dandiapi import DandiAPIClient, RemoteDandiset, RemoteAsset
from dandi.utils import get_instance
import re
from urllib.parse import unquote as url_unquote
from typing import Optional, Union
import requests


class RemoteDandiFileSystem(HTTPFileSystem):
    """
    A file system that browses through a remote dandiset.

    Examples
    --------
    Load and parse a remote file
    >> from dandifs import RemoteDandiFileSystem
    >> import json
    >> fs = RemoteDandiFileSystem()
    >> with fs.open('dandi://dandi/000026/rawdata/sub-I38/ses-MRI/anat/'
    >>              'sub-I38_ses-MRI-echo-4_flip-4_VFA.json') as f:
    >>      info = json.load(f)

    Browse a dataset
    >> from dandifs import RemoteDandiFileSystem
    >> fs = RemoteDandiFileSystem('000026')
    >> fs.glob('**/anat/*.json')

    """

    def __init__(
            self,
            dandiset: Optional[Union[str, RemoteDandiset]] = None,
            version: Optional[str] = None,
            client: Optional[Union[str, "DandiInstance", DandiAPIClient]] = None,
            **kwargs):
        """
        Initialise a remote DANDI file system.
        The root of a DANDI file system is a dandiset at a given version.
        The file system can be initialized from
            - a `RemoteDandiset` instance; or
            - the name of a dandiset [+ version]; and
                . a DandiAPIClient instance
                . a DandiInstance instance
                . the name of a known DANDI instance
                . the url of a DANDI server
        """
        super().__init__(**kwargs)
        if not isinstance(dandiset, RemoteDandiset):
            if not isinstance(client, DandiAPIClient):
                client = DandiAPIClient(client)
            if dandiset:
                dandiset = client.get_dandiset(dandiset, version)
        self._dandiset = dandiset
        self._client = None if dandiset else client

    @property
    def dandiset(self):
        return self._dandiset

    @dandiset.setter
    def dandiset(self, x):
        if x:
            self._client = None
        elif self._dandiset:
            self._client = self._dandiset.client
        self._dandiset = x

    @property
    def client(self):
        return self.dandiset.client if self.dandiset else self._client

    @client.setter
    def client(self, x):
        if self.dandiset:
            raise ValueError('Cannot assign a DANDI client to a FileSystem '
                             'that is already linked to a dandiset. '
                             'Unassign the dandiset first.')
        self._client = x

    @classmethod
    def for_url(cls, url: str) -> "DandiFileSystem":
        """
        Instantiate a FileSystem that interacts with the correct
        DANDI instance for a given url
        """
        instance, dandiset, version, *_ = split_dandi_url(url)
        return cls(dandiset, version, instance)

    async def _walk(self, path, maxdepth=None, **kwargs):
        """
        Walk through all files under a path.

        The path can be
        - relative to the root of the dandiset; or
        - an absolute URL

        Yields: (path, list of dirs, list of files)
        """
        path = stringify_path(path).strip('/')
        detail = kwargs.pop('detail', False)
        assets = kwargs.pop('assets', None)
        if assets is None:
            dandiset = kwargs.pop('dandiset', None)
            if not dandiset:
                dandiset, path = self.get_dandiset(path)
            assets = dandiset.get_assets_with_path_prefix(path)

        files = {}
        dirs = {}
        full_dirs = set()

        pattern_file = path + r'/(?P<f>[^/]+)'
        patter_dir = path + r'/(?P<d>[^/])+/.+'

        depth = len(path.split('/'))
        assets, assets_in = [], assets
        for asset in assets_in:
            asset = getattr(asset, 'path', asset)
            # is the input path exactly this asset?
            if asset == path:
                name = asset.split('/')[-1]
                files[''] = {'name': name, 'size': None, 'type': 'file'}
                continue
            pathname = asset.rstrip('/')
            name = pathname.split('/')
            name = name[min(depth, len(name)-1)]
            # is this asset a file directly under `path`
            match = re.fullmatch(pattern_file, asset)
            if match:
                name = match.group('f')
                files[name] = {
                    'name': path + '/' + name,
                    'size': None,
                    'type': 'file',
                }
                continue
            # is this asset a file a few levels under `path`?
            match = re.fullmatch(patter_dir, asset)
            if match:
                dirs[name] = {
                    'name': path + '/' + name,
                    'size': None,
                    'type': 'directory',
                }
                full_dirs.add(pathname)
            assets.append(asset)

        if detail:
            yield path, dirs, files
        else:
            yield path, list(dirs), list(files)

        if maxdepth is not None:
            maxdepth -= 1
            if maxdepth == 0:
                return
        kwargs['maxdepth'] = maxdepth
        kwargs['assets'] = assets
        kwargs['detail'] = detail

        for directory in full_dirs:
            async for _ in self._walk(directory, **kwargs):
                yield _

    async def _ls(self, path, detail=True, **kwargs):
        async for path, dirs, files in self._walk(path, detail=detail, maxdepth=1):
            if detail:
                return [*dirs.values(), *files.values()]
            else:
                return dirs + files
        return []

    ls = sync_wrapper(_ls)

    async def _glob(self, path, **kwargs):
        dandiset = kwargs.pop('dandiset', None)
        if not dandiset:
            dandiset, path = self.get_dandiset(path)
        self.dandiset, dandiset0 = dandiset, self.dandiset
        result = await super()._glob(path)
        self.dandiset = dandiset0
        return result

    def get_dandiset(self, path):
        """
        If path is a relative path, return (self.dandiset, path)
        Else, the path is an absolute URL and we instantiate the correct
        remote dandiset and spit out the relative path.

        Returns: dandiset, path
        """
        dandiset = self.dandiset
        if path.startswith(('http://', 'https://', 'dandi://', 'DANDI:')):
            instance, dandiset_id, version_id, path, asset_id = split_dandi_url(path)
            api_url = get_instance(instance)
            if self.client.api_url == api_url.api:
                client = self.client
            else:
                client = DandiAPIClient.for_dandi_instance(instance)
                dandiset = None
            if not asset_id:
                if not dandiset or dandiset.identifier != dandiset_id:
                    dandiset = client.get_dandiset(dandiset_id, version_id)
                if not dandiset or dandiset.version_id != version_id:
                    dandiset = client.get_dandiset(dandiset_id, version_id)
            else:
                asset = client.get_asset(asset_id)
                return dandiset, asset
        elif not self.dandiset:
            raise ValueError('File system must be linked to a dandiset to '
                             'use relative paths.')
        return dandiset, path

    def open(self, path, *args, **kwargs):
        dandiset, asset = self.get_dandiset(path)
        if not isinstance(asset, RemoteAsset):
            asset = dandiset.get_asset_by_path(asset)
        s3_url = _get_s3_url(asset.download_url)
        return super().open(s3_url, *args, **kwargs)


def split_dandi_url(url):
    """
    Split a valid dandi url into its subparts.
    Returns: (instance, dandiset_id, version_id, path, asset_id)
    where instance can be an instance_id or an URL.
    """
    instance = None
    server = None
    dandiset_id = None
    version = None
    path = ''
    asset_id = None
    if url.startswith('dandi://'):
        # dandi://<instance name>/<dandiset id>[@<version>][/<path>]
        pattern = r'dandi://(?P<i>[^/]+)/(?P<d>\d+)(@(?P<v>[^/]+))?(?P<p>.*)'
        match = re.match(pattern, url)
        if not match:
            raise SyntaxError('Wrong dandi url')
        instance = match.group('i')
        dandiset_id = match.group('d')
        version = match.group('v')
        path = match.group('p')
    elif url.startswith(('DANDI:', 'https://identifiers.org/DANDI:')):
        # DANDI:<dandiset id>[/<version id>]
        # https://identifiers.org/DANDI:<dandiset id>[/<version id>]
        pattern = r'(https://identifiers.org/)?DANDI:(?P<d>\d+)(/(?P<v>[^/]+))?'
        match = re.match(pattern, url)
        if not match:
            raise SyntaxError('Wrong dandi url')
        dandiset_id = match.group('d')
        version = match.group('v')
        instance = 'DANDI'
    else:
        pattern = r'https://(?P<s>[^/]+)(/api)?(/#)?(?P<u>.*)'
        match = re.match(pattern, url)
        if not match:
            raise SyntaxError('Wrong dandi url')
        server = match.group('s')
        url = match.group('u')
        if url.startswith('/dandisets/'):
            # https://<server>[/api]/dandisets/<dandiset id>[/versions[/<version>]]
            # https://<server>[/api]/dandisets/<dandiset id>/versions/<version>/assets/<asset id>[/download]
            # https://<server>[/api]/dandisets/<dandiset id>/versions/<version>/assets/?path=<path>
            pattern = r'/dandisets/(?P<d>\d+)(/versions/(?P<v>[^/]+))?(?P<u>.*)'
            match = re.match(pattern, url)
            if not match:
                raise SyntaxError('Wrong dandi url')
            dandiset_id = match.group('d')
            version = match.group('v')
            url = match.group('u')
            pattern = r'/assets/((\?path=(?P<p>[.*]+))|(?P<a>[^/]+))'
            match = re.match(pattern, url)
            if match:
                path = match.group('p')
                asset_id = match.group('a')
        elif url.startswith('/dandiset/'):
            # https://<server>[/api]/[#/]dandiset/<dandiset id>[/<version>][/files[?location=<path>]]
            pattern = r'(/(?P<v>[^/]+))?/files(\?location=(?P<p>.*))?'
            pattern = r'/dandiset/(?P<d>\d+)' + pattern
            match = re.match(pattern, url)
            dandiset_id = match.group('d')
            version = match.group('v')
            path = match.group('p')
        elif url.startswith('/assets/'):
            # https://<server>[/api]/assets/<asset id>[/download]
            pattern = r'/assets/(?P<a>[^/]+)'
            match = re.match(pattern, url)
            if not match:
                raise SyntaxError('Wrong dandi url')
            asset_id = match.group('a')

    path = url_unquote(path)
    path = (path or '').strip('/')

    if instance is None:
        instance = 'https://' + server

    return instance, dandiset_id, version, path, asset_id


def _get_s3_url(url):
    url = requests.request(url=url, method='head').url
    if '?' in url:
        return url[:url.index('?')]
    return url
