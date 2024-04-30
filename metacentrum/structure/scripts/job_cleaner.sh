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

if [ -z "$3" ]; then
    BACKUP_DIR=~/cgp_workspace/failed_jobs_results
else
    BACKUP_DIR=$3
fi

~/scripts/jobs_backup.sh "$1" "$2" "$BACKUP_DIR"
ssh $HOSTNAME "rm -rf $SCRATCHDIR/*"
DIR=$(ssh $HOSTNAME "ls -A $SCRATCHDIR/* || echo ''")

if [ -z "$DIR" ]; then
   exit 0
else
   echo "could not clear scratch directory: $DIR"
   exit 3
fi
