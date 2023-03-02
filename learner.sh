#!/bin/bash

if [ $# != 3 ]
then
    echo -e "usage:$0 [path to train_dir & d5.py] [first_time==1] [training_session=999]"
    echo "$@"
    echo "$#"
    exit
fi

path=$1
first_time=$2
training_session=$3
##Bring up the learner:
if [ $first_time -eq 1 ];
then
    /users/`whoami`/venv/bin/python $path/d5.py --job_name=learner --task=0 --base_path=$path --training_session=${training_session} &
elif [ $first_time -eq 4 ]
then
    /users/`whoami`/venv/bin/python $path/d5.py --job_name=learner --task=0 --base_path=$path --load --eval --training_session=${training_session} &
else
    /users/`whoami`/venv/bin/python $path/d5.py --job_name=learner --task=0 --base_path=$path --load --training_session=${training_session} &
fi
