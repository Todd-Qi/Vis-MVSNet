#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

NUM_SRC=7
SUBSET="test"

if [ "$SUBSET" == "training" ];then
scene_names=(courtyard delivery_area electro facade kicker meadow office pipes playground relief relief_2 terrace terrains)
fi

if [ "$SUBSET" == "test" ];then
scene_names=(door exhibition_hall lecture_room living_room lounge observatory old_computer statue terrace_2 botanical_garden boulders bridge)
fi

for((idx=0;idx<${#scene_names[@]};idx++))
do
scene=${scene_names[$idx]}
#echo $scene

python test.py \ # ../test.py
--load_path ./pretrained_model/vis \
--dataset_name eth3d_high_res \
--data_root /mnt/B/xxx/ETH3D/ \
--write_result --result_dir ./eth3d/num_src${NUM_SRC}_new/${SUBSET}/${scene} \
--resize 1920,1280  --crop 1920,1280 \
--num_src ${NUM_SRC} --interval_scale 1.0 \
--subset ${SUBSET} \
done

