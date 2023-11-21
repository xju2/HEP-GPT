#!/bin/bash

INDIR=/pscratch/sd/x/xju/LLMTracking/acts/run
OUTDIR=data/odd_fixed_length

# python scripts/create_data.py $INDIR/v1_ttbar $OUTDIR -w 1 --num-train 10 --num-val 0 --num-test 0 \
#   --outname-prefix v1 --with-padding

python scripts/create_data.py $INDIR/v0 $OUTDIR -w 1 --num-train 0 --num-val 10 --num-test 0 \
  --outname-prefix v1 --with-padding

