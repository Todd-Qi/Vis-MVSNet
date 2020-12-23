#!/bin/bash

for scene in Family Francis Horse Lighthouse M60 Panther Playground Train
do

DATA_PATH="/mnt/A/qiyh/2021/Github/Vis-MVSNet/tt/intermediate/${scene}"
PAIR_FILE="/mnt/B/qiyh/mvsnet/preprocessed_inputs/tt/intermediate/${scene}/pair.txt"

python fusion.py --data ${DATA_PATH} --pair ${PAIR_FILE} # ../fusion.py
done
