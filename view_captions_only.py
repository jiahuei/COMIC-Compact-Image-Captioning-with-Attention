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
from PIL import Image, ImageEnhance, ImageOps, ImageFont, ImageDraw
import os, json, sys, fcntl, random
import matplotlib.pyplot as plt
fl = fcntl.fcntl(sys.stdin.fileno(), fcntl.F_GETFL)
fcntl.fcntl(sys.stdin.fileno(), fcntl.F_SETFL, fl | os.O_NONBLOCK)
pjoin = os.path.join
IMG_RESIZE = 512
IMG_CROP = int(224 / 256 * IMG_RESIZE)
font = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', int(IMG_RESIZE / 7))


randomise = False

ckpt = 425064
data_root = '/ext_hdd/3_Datasets/MSCOCO_captions/val2014'
model_cap_json = '/ext_hdd/1_TF_files/caption_baseN/mscoco_v3/v3_0_bN256_256_multi_8_reuse_run_01_eval_beam_3/captions_results___model-425064.json'
base_cap_json = '/ext_hdd/1_TF_files/caption_baseN/mscoco_v3/v3_0_word_256_single_RERUN_FINAL_run_01_eval_beam_3/captions_results___model-407353.json'
score_json = '/ext_hdd/1_TF_files/caption_baseN/mscoco_v3/v3_0_bN256_256_multi_8_reuse_run_01_eval_beam_3/metric_scores_detailed.json'
output_dir = '/home/jiahuei/Documents/supplemental/insta_bN256_beam_3'

#ckpt = 425064
#data_root = '/ext_hdd/3_Datasets/InstaPIC1M/data/images'
#model_cap_json = '/ext_hdd/1_TF_files/caption_baseN/insta_v3/v3_0_bN256_256_multi_8_reuse_run_01_eval_beam_3/captions_results___model-303180.json'
#base_cap_json = '/ext_hdd/1_TF_files/caption_baseN/insta_v3/v3_0_word_256_single_RERUN_run_01_eval_beam_3/captions_results___model-525512.json'
#score_json = '/ext_hdd/1_TF_files/caption_baseN/insta_v3/v3_0_bN256_256_multi_8_reuse_run_01_eval_beam_3/metric_scores_detailed.json'
#output_dir = '/home/jiahuei/Documents/supplemental/insta_bN256_beam_3'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(model_cap_json, 'r') as f:
    m_cap_list = json.load(f)
with open(base_cap_json, 'r') as f:
    b_cap_list = json.load(f)
m_captions = {}
for c in m_cap_list:
    m_captions[c['image_id']] = c['caption']
b_captions = {}
for c in b_cap_list:
    b_captions[c['image_id']] = c['caption']

with open(score_json, 'r') as f:
    scores = json.load(f)

out_str = '\r\nModel JSON: {}\r\n'.format(os.path.basename(model_cap_json))
out_str += '\r\nBaseline JSON: {}\r\n\r\n\r\n'.format(os.path.basename(base_cap_json))
with open(pjoin(output_dir, 'captions.txt'), 'a') as f:
    f.write(out_str)

img_plot = None
img_id_list = []

scores = scores[str(ckpt)]

if randomise:
    random.seed(4896)
    random.shuffle(scores)
raise SystemExit
for score in scores[:]:
    img_id = score['image_id']
    if type(img_id) == unicode:
        img_fname = img_id
    else:
        img_fname = 'COCO_val2014_{:012d}.jpg'.format(img_id)
    
    img = Image.open(pjoin(data_root, img_fname))
    img = ImageEnhance.Brightness(img).enhance(1.15)
    img = ImageEnhance.Contrast(img).enhance(1.075)
    
    # Resize to 512 x 512 instead of 256 x 256
    # Crop to 448 x 448 instead of 224 x 224
    img = img.resize([IMG_RESIZE, IMG_RESIZE], Image.BILINEAR)
    img = ImageOps.crop(img, (IMG_RESIZE - IMG_CROP) / 2)
    
    # Visualise
    model_cap = m_captions[img_id]
    base_cap = b_captions[img_id]
    bg_big = Image.new('RGB', [IMG_RESIZE * 6, IMG_RESIZE * 2])
    bg_big.paste(img, (0, 0))
    draw = ImageDraw.Draw(bg_big)
    draw.text((10, int(IMG_RESIZE * 1.25)), model_cap, font=font)
    draw.text((10, int(IMG_RESIZE * 1.5)), base_cap, font=font)
    
    if img_plot is None:
        img_plot = plt.imshow(bg_big)
        plt.show()
        plt.pause(0.1)
    else:
        img_plot.set_data(bg_big)
        plt.show()
        plt.pause(0.1)
    
    # Get key press
    key_input = raw_input('Press "a" to save, "e" to end, or other keys to continue next.\n')
    if key_input == 'a':
        # Save image
        cider = 'c{:1.3f}'.format(score['CIDEr']).replace('.', '-')
        if type(img_id) == unicode:
            fname = '{}_{}.jpg'.format(cider, img_id)
        else:
            fname = '{}_{:012d}.jpg'.format(cider, img_id)
        img.save(pjoin(output_dir, fname))
        img_id_list.append(img_id)
        
        # Write captions
        out_str = '{}\r\nBL: {}\r\n\r\n'.format(model_cap, base_cap)
        with open(pjoin(output_dir, 'captions.txt'), 'a') as f:
            f.write('{}\r\n{}'.format(fname, out_str))
        
        # Write captions in LATEX format
        out_str = '\t\t\\gph{{1.0}}{{images/xxx/{}}}\t\t'.format(fname)
        out_str += '& \\begin{tabular}{m{\\linewidth}}\n'
        out_str += '\t\t\t\\tit{{{}}}\t\t\t\t\\\\ \\\\\n'.format(base_cap)
        out_str += '\t\t\t\\tbf{{{}}}\t\t\t\\\\ \n'.format(model_cap)
        out_str += '\t\t\\end{tabular} \\\\\n\n'
        with open(pjoin(output_dir, 'captions_latex.txt'), 'a') as f:
            f.write(out_str)
    
    elif key_input == 'e':
        break

# Write the image ids
if type(img_id_list[0]) != unicode:
    img_id_list = [str(i) for i in img_id_list]
    out_str = '\r\n\r\n\r\nImage IDs\r\n\r\n{}'.format(',\r\n'.join(img_id_list))
else:
    out_str = '\r\n\r\n\r\nImage IDs\r\n\r\n\'{}\''.format('\',\r\n\''.join(img_id_list))
with open(pjoin(output_dir, 'captions.txt'), 'a') as f:
    f.write(out_str)


'''
Regular expression to replace file paths
(/home/jiahuei/Dropbox/@_PhD/Codes/TensorFlow_scripts/caption_baseN/report/images/insta_bN256_beam_3/c)[\d]*(-)[\d]*_
(/home/jiahuei/Dropbox/@_PhD/Codes/TensorFlow_scripts/caption_baseN/report_files/images/mscoco_bN256_beam_3/c)[\d]*(-)[\d]*_[0]*

'''


