#!/bin/bash
set -e
for file in *; do
        pushd . > /dev/null
        cd $file 
        if [ -f jobs_info.txt ] && [ ! -f complete ]; then
            ~/scripts/job_cleaner.sh
        fi        
        popd > /dev/null
done
