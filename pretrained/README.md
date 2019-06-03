## Using the checkpoints
Just point `infer.py` to the directory containing the checkpoints. 
Model configurations are loaded from `config.pkl`.

For example:
```bash
python infer.py --infer_checkpoints_dir mscoco/radix_b256_add_LN_softmax_h8_tie_lstm_cnnFT_run_01
```


## Misc
The size of the checkpoints may be larger as they include CNN weights.
