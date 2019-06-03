# COMIC: Towards a Compact Image Captioning Model with Attention

Released on June 03, 2019

## Description

This is the code repo of our TMM2019 work titled [COMIC: Towards a Compact Image Captioning Model with Attention](https://arxiv.org/abs/1903.01072). In this paper, we tackle the problem of compactness of image captioning models which is hitherto unexplored. We showed competitive results on both MS-COCO and InstaPIC-1.1M datasets despite having an embedding vocabularly size that is 39x-99x smaller.

<img src="TMM.png" height="200">

## Source Code
Pre-trained checkpoints for the COMIC models are available in [pretrained](https://github.com/jiahuei/COMIC-Towards-A-Compact-Image-Captioning-Model-with-Attention/tree/master/pretrained) folder. Details are provided in a separate README.

Note that, some parts of this code may be subject to change.

## Citation

Please cite the following paper if you use this repository in your reseach:

```
@article{tan2019comic,
  title={COMIC: Towards A Compact Image Captioning Model with Attention},
  author={Tan, Jia Huei and Chan, Chee Seng and Chuah, Joon Huang},
  journal={IEEE Transactions on Multimedia},
  year={in Press},
  publisher={IEEE}
}
```

## Dependencies
- tensorflow 1.9.0
- python 2.7
- java 1.8.0
- tqdm >= 4.24.0
- Pillow >= 3.1.2
- packaging >= 19.0
- requests >= 2.18.4


## Running the code
Assuming you are in the [src](https://github.com/jiahuei/COMIC-Towards-A-Compact-Image-Captioning-Model-with-Attention/tree/master/src) folder:

1. Run `setup.sh`. This will download the required Stanford models 
and run all the dataset pre-processing scripts.

2. Run the training script `python train.py`.

3. Run the inference and evaluation script 
`python infer.py --infer_checkpoints_dir mscoco/logdir`.

Examples are given in `example.sh`.

## Training the models (MS-COCO)
### COMIC-256
```bash
python train.py
```
### Baseline
```bash
python train.py  \
    --token_type 'word'  \
    --cnn_fm_projection 'none'  \
    --attn_num_heads 1
```
### Baseline-8
```bash
python train.py  \
    --token_type 'word'  \
    --cnn_fm_projection 'none'  \
    --attn_num_heads 8
```
### Baseline-SC
```bash
python train.py  \
    --token_type 'word'  \
    --cnn_fm_projection 'none'  \
    --rnn_size 160  \
    --rnn_word_size 128  \
    --attn_num_heads 1
```
### InstaPIC models
InstaPIC models can be trained by passing this additional argument:
```bash
    --dataset_file_pattern 'insta_{}_v25595_s15'
```

## Avoid re-downloading datasets
Re-downloading can be avoided by:
1. Editing `setup.sh`
2. Providing the path to the directory containing the dataset files

```bash
python coco_prepro.py --dataset_dir /path/to/coco/dataset
python insta_prepro.py --dataset_dir /path/to/insta/dataset
```

In the same way, both `train.py` and `infer.py` accept alternative dataset paths.

```bash
python train.py --dataset_dir /path/to/dataset
python infer.py --dataset_dir /path/to/dataset
```

This code assumes the following dataset directory structures:

### MS-COCO
```
{coco-folder}
+-- captions
|   +-- {folder and files generated by coco_prepro.py}
+-- test2014
|   +-- {image files}
+-- train2014
|   +-- {image files}
+-- val2014
    +-- {image files}
```

### InstaPIC-1.1M
```
{insta-folder}
+-- captions
|   +-- {folder and files generated by insta_prepro.py}
+-- images
|   +-- {image files}
+-- json
    +-- insta-caption-test1.json
    +-- insta-caption-train.json
```


## Differences compared to our paper
To match the settings as described in our TMM paper, 
set the `legacy` argument of `train.py` to `True` (the default is `False`). 
This will override some of the provided arguments.

When using the default arguments, the differences compared to our TMM paper settings are:
- Attention map dropout is set to `0.1`
- RNN init method is changed to `x_{t=-1} = W_I * CNN(I)`
from `h_{t=-1} = W_I tanh (LN (I_{embed} ))`
- Changed training scheme (learning rate, ADAM epsilon)

Changes that can be enabled:
- CNN fine-tuning via the `train_mode` flag. Model is initialised using the 
last training checkpoint of RNN training.
- RNN variational dropout 
[[arxiv]](https://arxiv.org/abs/1512.05287)
[[tf]](https://www.tensorflow.org/versions/r1.9/api_docs/python/tf/contrib/rnn/DropoutWrapper#methods)
- Context layer (linear projection after attention)
- SCST [[arxiv]](https://arxiv.org/abs/1612.00563) **(to be added soon)**

### Performance differences in MS-COCO

| Default mode                  | BLEU-4    | CIDEr     | SPICE     |
| -------------                 | --------- | --------- | --------- |
| Baseline                      | 0.311 (0.296)     | 0.937  (0.885)   | 0.174  (0.167)   |
| **COMIC-256**                 | 0.308 (0.292)    | 0.944   (0.881)  | 0.176    (0.164) |
| COMIC-256 (CNN fine-tune)     | 0.328     | 1.001     | 0.185     |

Note that scores in bracket () indicate original TMM paper results. Please see [pretrained](https://github.com/jiahuei/COMIC-Towards-A-Compact-Image-Captioning-Model-with-Attention/tree/master/pretrained) folder.

## Main arguments

### train.py
- `train_mode`: The training regime. Choices are `decoder`, `cnn_finetune`, `scst`. 
All training starts with `decoder` mode (freezing the CNN).
- `legacy`: If `True`, will match settings as described in paper.
- `token_type`: Language model. Choices are `radix`, `word`, `char`.
- `radix_base`: Base value for Radix models.
- `cnn_name`: CNN model name.
- `cnn_input_size`: CNN input size.
- `cnn_fm_attention`: End point name of feature map for attention.
- `cnn_fm_projection`: Feature map projection method. Choices are `none`, `independent`, `tied`.
    
- `rnn_name`: Type of RNN. Choices are `LSTM`, `LN_LSTM`, `GRU`.
- `rnn_size`: Number of RNN units.
- `rnn_word_size`: Size of word embedding.
- `rnn_init_method`: RNN init method. Choices are `project_hidden`, `first_input`.
- `rnn_recurr_dropout`: If `True`, enable variational recurrent dropout.
    
- `attn_num_heads`: Number of attention heads.
- `attn_context_layer`: If `True`, add linear projection after multi-head attention.
- `attn_alignment_method`: Alignment / composition method. Choices are `add_LN`, `dot`.
- `attn_probability_fn`: Attention map probability function. Choices are `softmax`, `sigmoid`.

### infer.py
- `infer_set`: The split to perform inference on. Choices are `test`, `valid`, `coco_test`, `coco_valid`.
`coco_test` and `coco_valid` are for inferencing on the whole 
`test2014` and `val2014` sets respectively. 
These are used for MS-COCO online server evaluation.
- `infer_checkpoints_dir`: Directory containing the checkpoint files.
- `infer_checkpoints`: Checkpoint numbers to be evaluated. Comma-separated.
- `annotations_file`: Annotations / reference file for calculating scores.

- `infer_beam_size`: Beam size of beam search. Pass `1` for greedy search.
- `infer_length_penalty_weight`: Length penalty weight used in beam search.
- `infer_max_length`: Maximum caption length allowed during inference.
- `batch_size_infer`: Inference batch size for parallelism.


## Microsoft COCO Caption Evaluation
This code uses the standard `coco-caption` code with *SPICE* metric
[[Link to repo]](https://github.com/tylin/coco-caption/tree/3a9afb2682141a03e1cdc02b0df6770d2c884f6f).


To perform online server evaluation:
1. Infer on `coco_test` (test2014), rename the JSON output file to `captions_test2014__results.json`.
2. Infer on `coco_valid` (val2014), rename the JSON output file to `captions_val2014__results.json`.
3. Zip the files and submit.


## Feedback
Suggestions and opinions on this dataset (both positive and negative) are greatly welcomed. Please contact the authors by sending an email to
`tan.jia.huei at gmail.com` or `cs.chan at um.edu.my`.

## License and Copyright
The project is open source under BSD-3 license (see the ``` LICENSE ``` file).

&#169;2019 Center of Image and Signal Processing, Faculty of Computer Science and Information Technology, University of Malaya.


