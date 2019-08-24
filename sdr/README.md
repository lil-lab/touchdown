## Touchdown Spatial Description Resolution (SDR) Task

Before running the scripts, set paths for `data_dir`, `image_dir`, and `target_dir`.


To run model Concat:

```
python3 train.py --embed_dropout 0.5 --lr 0.001 --model concat --bidirectional True --log --name concat
```

To run model ConcatConv:

```
python3 train.py --embed_dropout 0.5 --lr 0.001 --model concat_conv --bidirectional True --num_conv_layers 1 --log --name concat_conv
```

To run model RNN2Conv:

```
python3 train.py --embed_dropout 0.5 --lr 0.001 --model rnn2conv --bidirectional True -- num_rnn2conv_layers 1 --log --name rnn2conv
```

To run model LingUNet:

```
python3 train.py --embed_dropout 0.5 --lr 0.0005 --model lingunet --bidirectional True --num_lingunet_layers 2 --log --summary --name lingunet
```

