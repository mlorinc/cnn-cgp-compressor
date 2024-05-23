#!/bin/bash
#PBS -N $job_name
#PBS -l $machine
#PBS -l walltime=$time_limit 

JOB_NAME=$job_name
JOB_DIR=$job_dir
USERNAME=$username
SERVER=$server
HF_TOKEN=$hf_token
EXPERIMENT_WILDCARD=$experiment_wildcard
STATS_FORMAT=$stats_format
NUM_WORKERS=$num_workers
NUM_PROC=$num_proc
BATCH_SIZE=$batch_size

export DATA_DIR=$data_dir
export RESULT_DIR=$result_dir
export EXPERIMENT=$experiment
export SCRATCH_CWD=$cwd
export MODEL_NAME=$model_name
export MODEL_PATH=$model_path

export CGP_CPP_PROJECT=$cgp_cpp_project
export CGP_BINARY_SRC=$cgp_binary_src
export CGP_BINARY=$cgp_binary
export CGP=$SCRATCHDIR/$CGP_BINARY
export NO_CGP_CLEAN="yes"
export ERROR_DATATYPE=$error_t
export CXXFLAGS_EXTRA="$cflags"
export DATASET=$dataset
export MODULO=$modulo
export MODULO_GROUP=$modulo_group
# PYTHON TEMPLATE END

# test if scratch directory is set
# if scratch directory is not set, issue error message and exit
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

JOB_MESSAGE="$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR"
# append a line to a file "jobs_info.txt" containing the ID of the job, the hostname of node it is run on and the path to a scratch directory
# this information helps to find a scratch directory in case the job fails and you need to remove the scratch directory manually
echo $JOB_MESSAGE >> $JOB_NAME.jobs_info.txt

export OMP_NUM_THREADS=$PBS_NUM_PPN # set it equal to PBS variable PBS_NUM_PPN (number of CPUs in a chunk)
cd $SCRATCHDIR || { echo >&2 "Could not change directory to $SCRATCHDIR"; exit 1; }

ml add intelcdk mambaforge || { echo >&2 "Could not add intel or mamba"; exit 1; }

/storage/brno12-cerit/home/mlorinc/jobs/compile_cgp.pbs.sh || { echo >&2 "Could not compile C++ CGP"; exit 1; }

# Download python environment and activate it
cp /storage/brno12-cerit/home/mlorinc/python/pytorch_env.tar $SCRATCHDIR || { echo >&2 "Could not copy mamba env"; exit 1; }
tar -xvf $SCRATCHDIR/pytorch_env.tar || { echo >&2 "Could not untar the tar"; exit 1; }
mamba activate $SCRATCHDIR/masters || { echo >&2 "Could not activate python env"; exit 1; }

rm -rf $SCRATCHDIR/compress_py/data_store/$EXPERIMENT || { echo >&2 "Could not delete $SCRATCHDIR/compress_py/data_store/$EXPERIMENT"; exit 1; }
mkdir -p $SCRATCHDIR/compress_py/data_store/$EXPERIMENT || { echo >&2 "Could not mkdir $SCRATCHDIR/compress_py/data_store/$EXPERIMENT"; exit 1; }

if [ ! -z "$MODULO" ]; then
# Copy relevant experiments and extract them
for file in $(ls -A "$DATA_DIR" | awk "NR % $MODULO == $MODULO_GROUP" | tr '\n' ' '); do
    cp $DATA_DIR/$file $SCRATCHDIR/compress_py/data_store/$EXPERIMENT || { echo >&2 "Could not copy $file from $DATA_DIR/"; exit 4; }
    cd $SCRATCHDIR/compress_py/data_store/$EXPERIMENT && unzip $file || { echo >&2 "Could not unzip $file at $SCRATCHDIR/compress_py/data_store/$EXPERIMENT"; exit 4; }
    rm $file
done
else
    cp $DATA_DIR/*.zip $SCRATCHDIR/compress_py/data_store/$EXPERIMENT || { echo >&2 "Could not copy *.zip from $DATA_DIR/"; exit 4; }
    cd $SCRATCHDIR/compress_py/data_store/$EXPERIMENT
    for file in *.zip; do
        unzip $file
        rm $file
    done
fi

# Copy python worksapce and delete any remaining experiments
cp -r /storage/brno12-cerit/home/mlorinc/compress_py $SCRATCHDIR || { echo >&2 "Could not copy /storage/brno12-cerit/home/mlorinc/compress_py to $SCRATCHDIR"; exit 1; }
mkdir -p $RESULT_DIR/GROUP_$MODULO_GROUP
mkdir -p $RESULT_DIR/model_metrics
cp -r $RESULT_DIR/model_metrics $SCRATCHDIR/compress_py/data_store/$EXPERIMENT || { echo >&2 "Could not copy /storage/brno12-cerit/home/mlorinc/compress_py to $SCRATCHDIR"; exit 1; }

# move into scratch directory
cd $SCRATCHDIR/$SCRATCH_CWD || { echo >&2 "Error while moving to the experiment dir $SCRATCHDIR/compress_py!"; exit 3; }
echo -e "pbs_server=$SERVER\npbs_username=$USERNAME\ndatastore=$SCRATCHDIR/compress_py/data_store\ncgp=$SCRATCHDIR/$CGP_CPP_PROJECT/$CGP_BINARY_SRC\nhuggingface=$HF_TOKEN\nTQDM_DISABLE=1\n" > .env || { echo >&2 "Could not create .env file"; exit 3; }

python ./compress.py $EXPERIMENT:model-metrics $MODEL_NAME $MODEL_PATH --experiment $EXPERIMENT_WILDCARD  -s $STATS_FORMAT --top 1 --num-workers $NUM_WORKERS --num-proc $NUM_PROC --batch-size $BATCH_SIZE --include-loss 2>> stderr.log 1>> stdout.log || { echo >&2 "Calculation ended up erroneously (with a code $?) !!"; exit 3; }
# python ./compress.py $EXPERIMENT:model-metrics $MODEL_NAME $MODEL_PATH --experiment $EXPERIMENT_WILDCARD  -s $STATS_FORMAT --top 1 --num-workers 14 --num-proc 1 --batch-size 2048 --include-loss 2>> stderr.log 1>> stdout.log || { echo >&2 "Calculation ended up erroneously (with a code $?) !!"; exit 3; }
# move the output to user's DATADIR or exit in case of failure
cp -r $SCRATCHDIR/compress_py/data_store/$EXPERIMENT/* $RESULT_DIR/GROUP_$MODULO_GROUP || { echo >&2 "Result file(s) copying failed (with a code $?) !!"; exit 4; }
echo "copied files to $RESULT_DIR/GROUP_$MODULO_GROUP"
# clean the SCRATCH directory
clean_scratch
