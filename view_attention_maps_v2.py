#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 19:24:37 2016

@author: jiahuei
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#from __future__ import unicode_literals

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt4')
#get_ipython().run_line_magic('matplotlib', 'qt')
from PIL import Image, ImageEnhance, ImageFont, ImageDraw, ImageOps
import os, json, sys, fcntl
import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle
fl = fcntl.fcntl(sys.stdin.fileno(), fcntl.F_GETFL)
fcntl.fcntl(sys.stdin.fileno(), fcntl.F_SETFL, fl | os.O_NONBLOCK)
font = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 30)
pjoin = os.path.join

data_root = '/home/jiahuei/Documents/3_Datasets/MSCOCO_captions'
eval_root = '/home/jiahuei/Documents/1_TF_files/caption_baseN/mscoco_v5/v5_1_icpV1_lnlstm_word_mixed4f_b32_r512_zeroInit_cnnFT_run_01/infer_test_beam_3_lpen_0.0'

#data_root = '/ext_hdd/3_Datasets/InstaPIC1M/data/images'
#eval_root = '/ext_hdd/1_TF_files/caption_baseN/insta_v3/v3_0_bN256_256_multi_8_reuse_run_01_eval_beam_1_batch_25'

with open(pjoin(eval_root, 'outputs___354216.pkl'), 'rb') as f:
    outputs = pickle.load(f)

with open(pjoin(eval_root, 'metric_scores_detailed.json'), 'r') as f:
    scores = json.load(f)


IMG_INSIZE = 224
IMG_SAVE_SIZE = min(512, IMG_INSIZE * 3)


fid = 0
img_plot = None
#for score in scores['425064']:
for score in scores['354216']:
    img_id = score['image_id']
    if type(img_id) == unicode:
        img_fname = img_id
    else:
        img_fname = 'val2014/COCO_val2014_{:012d}.jpg'.format(img_id)
    
    img = Image.open(pjoin(data_root, img_fname))
    img_big = img.copy()
    img = img.resize([IMG_INSIZE, IMG_INSIZE], Image.BILINEAR)
    img_big = img_big.resize([IMG_SAVE_SIZE, IMG_SAVE_SIZE], Image.BILINEAR)
    
    # Get attention maps
    attn_maps = outputs['attention'][img_fname]
    map_size = list(attn_maps.shape)
    attn_maps = np.reshape(attn_maps, [map_size[0], -1, 14, 14])
    
    # Apply attention map
    norm_maps = []
    bg = Image.new('RGB', [IMG_INSIZE, IMG_INSIZE])
    for head in range(8):
        m = attn_maps[head, :, :, :]
        m_max = m.max()
        if m_max < 0.01:
            m *= (255.0 / m_max / 5)
        else:
            m *= (255.0 / m_max)
        m = m.astype(np.uint8)
        
        time_series = []
        for t in m:
            t = Image.fromarray(t)
            t = t.convert('L')
            t = t.resize([IMG_INSIZE, IMG_INSIZE], Image.BILINEAR)
            comp = Image.composite(img, bg, t)
            comp = ImageEnhance.Brightness(comp).enhance(2.0)
            comp = ImageEnhance.Contrast(comp).enhance(1.5)
            time_series.append(comp)
        norm_maps.append(time_series)
    
    # Visualise
    vis_map = []
    for t in range(map_size[1]):
        vis_map.append([_[t] for _ in norm_maps])
    bg_big = Image.new('RGB', [IMG_INSIZE * 6, IMG_INSIZE * 4])
    bg_big.paste(img, (0, IMG_INSIZE * 2))
    draw = ImageDraw.Draw(bg_big)
    draw.text((10, int(IMG_INSIZE * 3.5)), outputs['captions'][img_fname], font=font)
    
    final_vis = []
    for m_list in vis_map:
        x = 0
        y = 0
        for i, m in enumerate(m_list):
            bg_big.paste(m, (x, y))
            x += IMG_INSIZE
            if i == 3:
                x = 0
                y = IMG_INSIZE
        final_vis.append(bg_big.copy())
    
    key_input = 'r'
    while key_input == 'r':
        for z in range(2):
            for v in final_vis:
                if img_plot is None:
                    img_plot = plt.imshow(v)
                    plt.show()
                    plt.pause(.15)
                else:
                    img_plot.set_data(v)
                    plt.show()
                    plt.pause(.15)
            
        key_input = raw_input('Press "a" to save, "r" to repeat, "e" to end, or other keys to continue next.\n')
        if key_input == 'a':
            # Save attention maps
            if type(img_id) == unicode:
                save_dir = pjoin(eval_root, '{}_c{}'.format(img_id, score['CIDEr']))
            else:
                save_dir = pjoin(eval_root, '{:012d}_c{}'.format(img_id, score['CIDEr']))
            
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            img_big.save(pjoin(save_dir, 'base.jpg'))
            for i, h in enumerate(norm_maps):
                for j, t in enumerate(h):
                    t.save(pjoin(save_dir, 'h{}_t{}.jpg'.format(i, j)))
            with open(pjoin(save_dir, 'caption.txt'), 'w') as f:
                f.write(outputs['captions'][img_fname])
            break
        elif key_input == 'r':
            pass
        else:
            break
    
    if key_input == 'e':
        break
    fid += 1


