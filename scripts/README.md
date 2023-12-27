# Instructions

## Create events for training from ACTS Open data detector
```bash
./run_create_events.sh
```

## Convert the CSV files to HDF5
```bash
time python scripts/convert_acts_to_hdf5.py /pscratch/sd/x/xju/LLMTracking/acts/run/v4 data/odd_raw_data/v4 -w 50
```