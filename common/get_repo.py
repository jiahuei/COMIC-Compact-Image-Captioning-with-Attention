#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 16:57:10 2019

@author: jiahuei
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
import utils
pjoin = os.path.join


common = CURR_DIR
scst = pjoin(CURR_DIR, 'scst')

##

print('\nINFO: Fetching `tylin/coco-caption` @ commit 3a9afb2 ...')
dest = common
zip_path = utils.maybe_download_from_url(
    r'https://github.com/tylin/coco-caption/archive/3a9afb2682141a03e1cdc02b0df6770d2c884f6f.zip',
    dest)
utils.extract_zip(zip_path)
os.remove(zip_path)
old_name = pjoin(dest, 'coco-caption-3a9afb2682141a03e1cdc02b0df6770d2c884f6f')
new_name = pjoin(dest, 'coco_caption')
os.rename(old_name, new_name)


print('\nINFO: Fetching `ruotianluo/cider` @ commit 77dff32 ...')
dest = scst
zip_path = utils.maybe_download_from_url(
    r'https://github.com/ruotianluo/cider/archive/dbb3960165d86202ed3c417b412a000fc8e717f3.zip',
    dest)
utils.extract_zip(zip_path)
os.remove(zip_path)
old_name = pjoin(dest, 'cider-dbb3960165d86202ed3c417b412a000fc8e717f3')
new_name = pjoin(dest, 'cider_ruotianluo')
os.rename(old_name, new_name)



