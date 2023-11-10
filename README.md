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
python data/trackml/reader.py /global/cfs/cdirs/m3443/data/trackml-codalab/train_all data/trackml_fixed_length/ -w 10 --num-train 10 --num-val 10 --padding --prefix v1
```

And Training

```bash
python train.py
```
