## Using the checkpoints
Just point `infer.py` to the directory containing the checkpoints. 
Model configurations are loaded from `config.pkl`.

```bash
python infer.py --infer_checkpoints_dir /path/to/checkpoint/dir
```


## Misc
The size of the checkpoints may be larger as it includes the CNN weights.
