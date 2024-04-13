#!/bin/bash
set -e
for file in *; do
        pushd . > /dev/null
        cd $file && { [ -f jobs_info.txt ] || echo $file; }
        popd > /dev/null
done
