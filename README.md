# HEP-GPT
GPT for High Energy Physics

## Installation

```
pip install -r requirements.txt
```

## Usage

```bash
python data/trackml/reader.py /global/cfs/cdirs/m3443/data/trackml-codalab/train_all data/trackml/ -w 10
```

And Training


```bash
python train.py
```

## To Do List

### Model exploration
- [ ] *data representation*, Preare the data with a *fixed* block size (18 + 2). In the tensor, rows are tracks and columns are the UMIDs. Then we *always* feed the model with complete tracks.
- [ ] *model architecture*, Do we need the positiional encoding?


### Physics performance
- [ ] Average the loss evaluation across batche and block sizes
- [ ] Calculate the accuracy of the model predicting the next hit position (aka. the next detector module)
- [ ] Determine how many detecot modules are needed to accurately (100%) predict the next hit position


### Computing performance
- [ ] Calculate the memory usage, flops, time per event, GPU utilization, power consumption
- [ ] Enable distributed training. Check the `fsdp` strategy.
- [ ] Use `torch.utils.checkpoint` to reduce the memory usage
- [ ] Use `torch.utils.bottleneck` to find the bottleneck

## Interesting findings

- [ ] Enable `compiling` reduces the time from 157.1 s to 88.3 s, 40% faster.