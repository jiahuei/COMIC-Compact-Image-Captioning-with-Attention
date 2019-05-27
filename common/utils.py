# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 23:38:59 2019

@author: jiahuei

Utility functions.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, math, time
import requests
import tarfile
from zipfile import ZipFile
from tqdm import tqdm
#from PIL import Image
#Image.MAX_IMAGE_PIXELS = None
# By default, PIL limit is around 89 Mpix (~ 9459 ** 2)


_EXT = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']
pjoin = os.path.join

try:
    from natsort import natsorted, ns
except ImportError:
    natsorted = None


def maybe_download_from_url(url, dest_dir):
    """
    Downloads file from URL, streaming large files.
    """
    #url = 'http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz'
    response = requests.get(url, stream=True)
    #chunk_size = 1024 * 512         # 512 kB
    chunk_size = 1024 ** 2          # 1 MB
    fname = url.split('/')[-1]
    
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    fpath = pjoin(dest_dir, fname)
    if os.path.isfile(fpath):
        print('INFO: Found file `{}`'.format(fname))
        return fpath
    
    if response.ok:
        print('INFO: Downloading `{}`'.format(fname))
    else:
        print('ERROR: Download error. Server response: {}'.format(response))
        return False
    time.sleep(0.2)
    
    # Case-insensitive Dictionary of Response Headers.
    # The length of the request body in octets (8-bit bytes).
    file_size = int(response.headers['Content-Length'])
    num_iters = math.ceil(file_size / chunk_size)
    tqdm_kwargs = dict(desc = 'Download progress',
                       total = num_iters,
                       unit = 'MB')
    with open(fpath, 'wb') as handle:
        for chunk in tqdm(response.iter_content(chunk_size), **tqdm_kwargs):
            if not chunk: break
            handle.write(chunk)
    print('INFO: Download complete: `{}`'.format(fname))
    return fpath


def maybe_download_from_google_drive(id, fpath, file_size=None):
    URL = 'https://docs.google.com/uc?export=download'
    chunk_size = 1024 ** 2          # 1 MB
    fname = os.path.basename(fpath)
    out_path = os.path.split(fpath)[0]
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if os.path.isfile(fpath):
        print('INFO: Found file `{}`'.format(fname))
        return fpath
    print('INFO: Downloading `{}`'.format(fname))
    
    session = requests.Session()
    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)
    
    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)
    
    if file_size is not None:
        num_iters = math.ceil(file_size / chunk_size)
    else:
        num_iters = None
    tqdm_kwargs = dict(desc = 'Download progress',
                       total = num_iters,
                       unit = 'MB')
    with open(fpath, 'wb') as handle:
        for chunk in tqdm(response.iter_content(chunk_size), **tqdm_kwargs):
            if not chunk: break
            handle.write(chunk)
    print('INFO: Download complete: `{}`'.format(fname))
    return fpath


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def extract_tar_gz(fpath):
    """
    Extracts tar.gz file into the containing directory.
    """
    def progress(members):
        with tqdm(total=len(members)) as pbar:
            for tarinfo in members:
                yield tarinfo
                pbar.update()
    tar = tarfile.open(fpath, 'r:gz')
    
    opath = os.path.split(fpath)[0]
    tar.extractall(path=opath, members=progress(tar))   # members=None to extract all
    tar.close()


def extract_zip(fpath):
    """
    Extracts zip file into the containing directory.
    """
    with ZipFile(fpath, 'r') as zip_ref:
        for f in tqdm(iterable=zip_ref.namelist(),
                      total=len(zip_ref.namelist())):
            zip_ref.extract(member=f, path=os.path.split(fpath)[0])
        #zip_ref.extractall(os.path.split(fpath)[0])






