#!/bin/bash

#SUBSET="training"
SUBSET="test"

#for scene in courtyard delivery_area electro facade kicker meadow office pipes playground relief relief_2 terrace terrains
for scene in botanical_garden boulders bridge door exhibition_hall lecture_room living_room lounge observatory old_computer statue terrace_2
do

DATA_PATH="/mnt/A/qiyh/2021/Github/Vis-MVSNet/eth3d/num_src7/${SUBSET}/${scene}"
PAIR_FILE="/mnt/B/qiyh/ETH3D/${SUBSET}/${scene}/pair.txt"

python fusion.py --data ${DATA_PATH} --pair ${PAIR_FILE} # ../fusion.py

done
