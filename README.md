# HEP-GPT
GPT for High Energy Physics

## Installation

```
pip install -r requirements.txt
```

## Usage

* Start the environment: `source $SCRATCH/venv/llm/bin/activate`
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

### Preprocessing

Convert the ACTS CSV files into one Parquet file.
```bash
cd /pscratch/sd/x/xju/LLMTracking/HEP-GPT

python scripts/convert_acts_to_hdf5.py /pscratch/sd/x/xju/LLMTracking/MCGenerators/acts/dask_v5_p500_try1 data/raw_parquets/v5/partition1 -w 10
```