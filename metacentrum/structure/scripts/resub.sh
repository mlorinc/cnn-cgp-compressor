#!/bin/bash
set -e
for file in *; do
        pushd . > /dev/null
        cd $file && { [ -f jobs_info.txt ] || qsub train.pbs.sh; }
        popd > /dev/null
done
