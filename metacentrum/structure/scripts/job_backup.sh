#!/bin/bash
set -e

message="$(tail -1 jobs_info.txt)"

PBS_JOBID=$(echo "$message" | awk '{print $1}')
HOSTNAME=$(echo "$message" | awk '{print $6}')
SCRATCHDIR=$(echo "$message" | awk '{print $11}')
EXPERIMENT=$(echo "$message" | awk '{print $14}')
EXPERIMENT_FOLDER=$(echo "$message" | awk '{print $16}')
DATADIR=$(echo "$message" | awk '{print $20}')
TIMESTAMP=${TIMESTAMP:-$(date '+%d_%m_%Y_%H_%M_%S')}


if [ -z "$SCRATCHDIR" ]; then
    echo "could not find SCRATCHDIR"
    exit 1
fi

if [ -z "$EXPERIMENT_FOLDER" ]; then
    DATADIR=/storage/brno12-cerit/home/mlorinc/cgp_workspace
fi

if [ -z "$3" ]; then
    BACKUP_DIR=/storage/brno12-cerit/home/mlorinc/cgp_workspace/jobs_backups/
else
    BACKUP_DIR=$3
fi

echo "PBS_JOBID=$PBS_JOBID"
echo "HOSTNAME=$HOSTNAME"
echo "SCRATCHDIR=$SCRATCHDIR"
echo "DATADIR=$DATADIR"

if [ -z "$EXPERIMENT_FOLDER" ]; then
    EXPERIMENT_FOLDER="$(realpath -s --relative-base="$DATADIR" "$PWD")"
    EXPERIMENT=$(basename "$EXPERIMENT_FOLDER")
    EXPERIMENT_FOLDER=$(dirname "$EXPERIMENT_FOLDER")
fi

echo "EXPERIMENT=$EXPERIMENT"
echo "EXPERIMENT_FOLDER=$EXPERIMENT_FOLDER"

if [ -z "$EXPERIMENT_FOLDER" ] || [ -z "$EXPERIMENT" ]; then
    echo "either EXPERIMENT_FOLDER is empty or EXPERIMENT"
    exit 2
fi

echo "backing up into: $BACKUP_DIR/$EXPERIMENT_FOLDER/$1/$EXPERIMENT"
if [ ! -z "$2" ]; then
    ssh $HOSTNAME \
    "cd $SCRATCHDIR/$EXPERIMENT_FOLDER/$EXPERIMENT/train_statistics/fitness && \
    for file in *.csv; do head -1 \$file > \$file.short && awk -F ',' -v OFS=',' '{ print \$1, \$2, \$3, \$4, \$5, \$6, \$7, \$8, \$9, \$10, \$11, "\"\"" }' \$file |\
    tail -n +2 >> \$file.short &&\
    tail -$2 \$file >> \$file.short; done &&\
    cd $SCRATCHDIR/$EXPERIMENT_FOLDER/$EXPERIMENT && zip -r data.zip cgp_configs train_statistics/fitness/*.short weights *.*" || { echo "skipping $EXPERIMENT"; exit 0; }
else
    ssh $HOSTNAME "cd $SCRATCHDIR/$EXPERIMENT_FOLDER/$EXPERIMENT && zip -r data.zip *" || { echo "skipping $EXPERIMENT"; exit 0; }
fi

mkdir -p $BACKUP_DIR/$EXPERIMENT_FOLDER/$1/$EXPERIMENT
scp $HOSTNAME:$SCRATCHDIR/$EXPERIMENT_FOLDER/$EXPERIMENT/data.zip $BACKUP_DIR/$EXPERIMENT_FOLDER/$1/$EXPERIMENT/${TIMESTAMP}_data.zip
