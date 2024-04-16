#!/bin/bash
#PBS -N CGP_COMPILE
#PBS -l select=1:ncpus=2:mem=2gb:scratch_local=2gb
#PBS -l walltime=00:10:00

# define a DATADIR variable: directory where the input files are taken from and where output will be copied to
DATADIR=${DATADIR:-/storage/brno12-cerit/home/mlorinc/cgp_workspace}
CGP_CPP_PROJECT=${CGP_CPP_PROJECT:-cgp_cpp_project}
CGP_BINARY_SRC=${CGP_BINARY_SRC:-bin/cgp}
CGP_BIN_DST=${DESTINATION:-$DATADIR/$CGP_CPP_PROJECT/$CGP_BINARY_SRC}

# test if scratch directory is set
# if scratch directory is not set, issue error message and exit
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

JOB_MESSAGE="$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR"
# append a line to a file "jobs_info.txt" containing the ID of the job, the hostname of node it is run on and the path to a scratch directory
# this information helps to find a scratch directory in case the job fails and you need to remove the scratch directory manually 
echo $JOB_MESSAGE >> $DATADIR/$CGP_CPP_PROJECT/jobs_info.txt

ml add intelcdk

# if the copy operation fails, issue error message and exit
cp -r $DATADIR/$CGP_CPP_PROJECT $SCRATCHDIR || { echo >&2 "Error while copying CGP source file(s)!"; exit 2; }

cd $SCRATCHDIR/$CGP_CPP_PROJECT && make clean && make || { echo >&2 "Error while moving to the compilation dir!"; exit 3; }

mkdir -p $(dirname $CGP_BIN_DST) || { echo >&2 "Error while creating bin dir!"; exit 4; }

cp $SCRATCHDIR/$CGP_CPP_PROJECT/$CGP_BINARY_SRC $CGP_BIN_DST || { echo >&2 "Error while copying CGP binary!"; exit 5; }

test -n "$NO_CGP_CLEAN" || { echo "cleaning scratch dir"; clean_scratch; }
