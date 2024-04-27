#!/bin/bash
set -e
export TIMESTAMP=$(date '+%d_%m_%Y_%H_%M_%S')
for file in *; do
        pushd . > /dev/null
        cd $file 
        if [ -f jobs_info.txt ] && [ -f complete ]; then
            ~/scripts/job_backup.sh "$1"
        fi        
        popd > /dev/null
done
unset TIMESTAMP
