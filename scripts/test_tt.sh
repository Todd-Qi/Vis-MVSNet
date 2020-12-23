#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

NUM_SRC=7
SUBSET="advanced"

if [ "$SUBSET" == "intermediate" ];then
scene_names=(Family Francis Horse Lighthouse M60 Panther Playground Train)
fi

if [ "$SUBSET" == "advanced" ];then
scene_names=(Auditorium Ballroom Courtroom Museum Palace Temple)
fi

scene_names=(Temple)

for((idx=0;idx<${#scene_names[@]};idx++))
do
scene=${scene_names[$idx]}
#echo $scene

python test.py \ # ../test.py
--load_path ./pretrained_model/vis \
--dataset_name tanksandtemples \
--data_root /mnt/B/xxx/mvsnet/preprocessed_inputs/tt/ \
--write_result --result_dir ./tt/${SUBSET}/${scene} \
--resize 1920,1080  --crop 1920,1056 \
--num_src ${NUM_SRC} --interval_scale 1.0 \
--subset ${SUBSET} \

done