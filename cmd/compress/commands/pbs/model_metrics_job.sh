#!/bin/bash
#PBS -N $job_name
#PBS -l $machine
#PBS -l walltime=$time_limit 

# define a DATADIR variable: directory where the input files are taken from and where output will be copied to
export COPY_SRC=$copy_src
export COPY_DST=$copy_dst
export END_JOB_COPY_SRC=$end_copy_src
export END_JOB_COPY_DST=$end_copy_dst
export MAIN_PROGRAM="$program"
export SCRATCH_CWD=$cwd

export CGP_CPP_PROJECT=$cgp_cpp_project
export CGP_BINARY_SRC=$cgp_binary_src
export CGP_BINARY=$cgp_binary
export CGP=$SCRATCHDIR/$CGP_BINARY
export NO_CGP_CLEAN="yes"
export ERROR_DATATYPE=$error_t
export CXXFLAGS_EXTRA="$cflags"
# PYTHON TEMPLATE END

# test if scratch directory is set
# if scratch directory is not set, issue error message and exit
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }
rm -f $END_JOB_COPY_DST/complete

JOB_MESSAGE="$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR"
# append a line to a file "jobs_info.txt" containing the ID of the job, the hostname of node it is run on and the path to a scratch directory
# this information helps to find a scratch directory in case the job fails and you need to remove the scratch directory manually
mkdir -p $END_JOB_COPY_DST
echo $JOB_MESSAGE >> $END_JOB_COPY_DST/jobs_info.txt

export OMP_NUM_THREADS=$PBS_NUM_PPN # set it equal to PBS variable PBS_NUM_PPN (number of CPUs in a chunk)

/storage/brno12-cerit/home/mlorinc/jobs/compile_cgp.pbs.sh

ml add intelcdk mambaforge
mamba activate /storage/brno12-cerit/home/mlorinc/python/masters
# ml add intelcdk cmake python python36-modules
#singularity shell --nv /cvmfs/singularity.metacentrum.cz/NGC/PyTorch\:21.03-py3.SIF

mkdir -p $SCRATCHDIR/$COPY_DST || { echo >&2 "Could not create experiments folder in $SCRATCHDIR!"; exit 1; }

# if the copy operation fails, issue error message and exit
cp -r $COPY_SRC $SCRATCHDIR/$COPY_DST || { echo >&2 "Error while copying experiment file(s)!"; exit 2; }
cp $DATADIR/$CGP_CPP_PROJECT/$CGP_BINARY_SRC $SCRATCHDIR/$CGP_BINARY || { echo >&2 "Error while copying C++ CGP file(s)!"; exit 2; }

# move into scratch directory
cd $SCRATCHDIR/$SCRATCH_CWD || { echo >&2 "Error while moving to the experiment dir!"; exit 3; }

$MAIN_PROGRAM 2>> stderr.log 1>> stdout.log || { echo >&2 "Calculation ended up erroneously (with a code $?) !!"; exit 3; }

# move the output to user's DATADIR or exit in case of failure
cp -r $SCRATCHDIR/$END_JOB_COPY_SRC $END_JOB_COPY_DST || { echo >&2 "Result file(s) copying failed (with a code $?) !!"; exit 4; }

touch $END_JOB_COPY_DST/complete
# clean the SCRATCH directory
clean_scratch
