#!/bin/bash

INDIR=/pscratch/sd/x/xju/LLMTracking/acts/run
OUTDIR=data/odd_fixed_length

# python scripts/create_data.py $INDIR/v1_ttbar $OUTDIR -w 1 --num-train 10 --num-val 0 --num-test 0 \
#   --outname-prefix v0 --with-padding

# python scripts/create_data.py $INDIR/v0 $OUTDIR -w 1 --num-train 0 --num-val 10 --num-test 0 \
#   --outname-prefix v0 --with-padding

function create_v1() {
  python scripts/create_data.py $INDIR/v1_ttbar $OUTDIR --num-train 100 --num-val 0 --num-test 0 \
    --outname-prefix v1 --with-padding -w 10
}

#create_v1

# 1000 ttbar events in v2
python scripts/create_data.py $INDIR/v2 $OUTDIR -w 10 --num-train 800 --num-val 100 --num-test 100 --outname-prefix v2 --with-padding

# 10_000 ttbar events in v3
python scripts/create_data.py $INDIR/v3 $OUTDIR -w 10 --num-train 8000 --num-val 1000 --num-test 1000 --outname-prefix v3 --with-padding

# 100_000 ttbar events in v4
# python scripts/create_data.py $INDIR/v4 $OUTDIR -w 10 --num-train 80000 --num-val 10000 --num-test 10000 --outname-prefix v4 --with-padding

# 1_000_000 ttbar events in v5
# python scripts/create_data.py $INDIR/v5 $OUTDIR -w 10 --num-train 800000 --num-val 100000 --num-test 100000 --outname-prefix v5 --with-padding
