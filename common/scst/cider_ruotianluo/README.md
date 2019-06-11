CIDEr for Self-Critical Sequence Training (SCST)
===================

This module is based on the repo `ruotianluo/cider` (with modifications).

It is **not compatible with COCO output format** due to dict key differences.

Code for Consensus-based Image Description Evaluation. Provides CIDEr as well as
CIDEr-D (CIDEr Defended) which is more robust to gaming effects.


## Important Note

CIDEr by default (with idf parameter set to "corpus" mode) computes IDF values 
using the reference sentences provided. 
Thus, CIDEr score for a reference dataset with only 1 image will be zero. 
When evaluating using one (or few) images, set idf to "coco-val-df" instead, 
which uses IDF from the MSCOCO Vaildation Dataset for reliable results.

To enable the IDF mode "coco-val-df":
1. Download the [IDF file](https://github.com/ruotianluo/cider/blob/dbb3960165d86202ed3c417b412a000fc8e717f3/data/coco-val.p)
1. Rename the file to `coco-val-df.p`
1. Place the file in `./cider_ruotianluo/data`


## Dependencies
- java 1.8.0
- python 2.7


## References & Acknowledgments
To see the code differences, refer to this fork:
- [[jiahuei/cider]](https://github.com/jiahuei/cider)

Thanks to the developers of:
- [[ruotianluo/cider]](https://github.com/ruotianluo/cider/tree/dbb3960165d86202ed3c417b412a000fc8e717f3)

