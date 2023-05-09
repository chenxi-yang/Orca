if [ $# -eq 3 ]
then
    source setup.sh

    first_time=$1
    port_base=$2
    training_session=$3
    cur_dir=`pwd -P`
    scheme_="cubic"
    max_steps=50000         #todo Run untill you collect 50k samples per actor 
    eval_duration=30
    num_actors=1
    memory_size=$((max_steps*num_actors))
    dir="${cur_dir}/rl_module"

    # sed "s/\"num_actors\"\: 1/\"num_actors\"\: $num_actors/" $cur_dir/params_base.json > "${dir}/params.json"
    # sed -i "s/\"memsize\"\: 5320000/\"memsize\"\: $memory_size/" "${dir}/params.json"

    sed -i "s/\"num_actors\": [[:digit:]]\+,/\"num_actors\": $num_actors,/" "${dir}/params.json"
    sed -i "s/\"memsize\": [[:digit:]]\+,/\"memsize\": $memory_size,/" "${dir}/params.json"

    sudo killall -s9 python client orca-server-mahimahi

    epoch=20
    act_port=$port_base
    
    # Start learning a model
    # Bring up the learner:
    echo "./learner.sh  $dir $first_time $training_session &"
    if [ $1 -eq 1 ];
    then
        # Start the learning from the scratch
        /users/`logname`/venv/bin/python ${dir}/main.py --job_name=learner --task=0 --base_path=${dir} --training_session=${training_session} &
        lpid=$!
    else
        # Continue the learning on top of previous model
        /users/`logname`/venv/bin/python ${dir}/main.py --job_name=learner --task=0 --base_path=${dir} --load --training_session=${training_session} &
        lpid=$!
    fi
    sleep 10

    #Bring up the actors:
    # Here, we go with single actor
    #TODO: add actor
    echo "start $num_actors actors"
    for i in `seq 0 $((num_actors-1))`
    do
        echo "actor_id start: $i"
        upl="wired48"
        downl="constant-mbps-$((i+1))" #1: 12mbps, 2: 24mbps, 3: 36mbps, 4: 48mbps

        # this part is fixed
        dl=48 # Mbps
        del=10 # delay in ms
        bdp=$((2*dl*del/12))      #12Mbps=1pkt per 1 ms ==> BDP=2*del*BW=2*del*dl/12
        qs=$((2*bdp))

        act_id=$((i))

        echo "starting actor $act_id with port $act_port on trace $downl"
        ./actor.sh ${act_port} $epoch ${first_time} $scheme_ $dir $act_id $downl $upl $del 0 $qs 0 ${training_session} & # > actor_$i.txt &
        act_port=$((act_port+1))
        pids="$pids $!"
    done

    for pid in $pids
    do
        echo "waiting for $pid"
        wait $pid
    done

    sleep 10

    #Kill the learner
    sudo kill -s15 $lpid

    #Wait if it needs to save somthing!
    sleep 30

    #Make sure all are down ...
    for i in `seq 0 $((num_actors))`
    do
        sudo killall -s15 python
        sudo killall -s15 orca-server-mahimahi
    done
    sudo killall -s9 client
else
    echo "usage: $0 [{Learning from scratch=1} {Continue your learning=0} {Just Do Evaluation=4}] [base port number ]"
fi

