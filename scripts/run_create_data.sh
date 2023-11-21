#!/bin/bash

INDIR=/pscratch/sd/x/xju/LLMTracking/acts/run/v1_ttbar
OUTDIR=data/odd_fixed_length

python scripts/create_data.py $INDIR $OUTDIR -w 1 --num-train 10 --num-val 0 --num-test 0 \
  --outname-prefix v1 --with-padding
