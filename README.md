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
or create each sequence padded to the fixed length of 20.

```bash
python data/trackml/reader.py /global/cfs/cdirs/m3443/data/trackml-codalab/train_all data/trackml_fixed_length/ -w 10 --num-train 1000 --num-val 100 --padding --prefix v2
```

### Training

```bash
python main.py experiment=test_trackml
```

### Obselete training script

```bash
python scripts/train.py -h
```

Train the model for fixed length sequences
```bash
python scripts/train.py max_epochs=10000 compile=True data.train_data=data/trackml_fixed_length/v1_evt10_train.bin data.val_data=data/trackml_fixed_length/v1_evt10_val.bin training.batch_size=1024 model.n_embd=1024
```