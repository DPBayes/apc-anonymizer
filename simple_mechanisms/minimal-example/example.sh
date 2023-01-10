#!/bin/sh

## Create a conda environment if it does not exist
if [ -z "$(conda env list | grep waltti-minimal)" ]
then
    conda env create -f environment.yml
fi

## Activate conda environment
conda activate waltti-minimal

## Create SQL database to store the hyperparameter optimization results
initdb -D waltti_sqldb

## Start the database
pg_ctl -D waltti_sqldb -l logfile start

## Create an inner database
createdb hyperparameter_opt

## Run the hyperparameter optimization
CONFIG_FILE=../example_config_module.py
CONFIG_NAME=test_config
CONFIG_ARGS=( --n_seats 78 --n_cats 6 --cat_edges 5 40 50 65 72 78 --cat_names EMPTY MANY_SEATS_AVAILABLE FEW_SEATS_AVAILABLE STANDING_ROOM_ONLY CRUSHED_STANDING_ROOM_ONLY FULL )
NUM_TRIALS_PER_TASK=20
# We will paralellize this over 4 processes to speed-up the process
pids=()
for i in {1..4}
do
    python3 ../main.py $CONFIG_FILE $CONFIG_NAME infer --train_hyperparameters --num_trials=$NUM_TRIALS_PER_TASK $CONFIG_ARGS &
    pids+=($pids $!)
    sleep 1
done

for pid in "${pids[@]}"
do
    wait $pid
done

#python3 ../main.py $CONFIG_FILE $CONFIG_NAME infer --train_hyperparameters --num_trials=$NUM_TRIALS_PER_TASK $CONFIG_ARGS

### Run the inference once more with the best hyperparameters
python3 ../main.py $CONFIG_FILE $CONFIG_NAME infer $CONFIG_ARGS

## Sample from learned model
LATENT_COUNT=1 # the number of passengers in bus
python3 ../main.py $CONFIG_FILE $CONFIG_NAME sample $LATENT_COUNT $CONFIG_ARGS

## Stop DB
pg_ctl stop -D waltti_sqldb
