#!/bin/bash
set -e
for file in *; do
        pushd . > /dev/null
        cd $file && { [ ! -f complete ] && echo $file; }
        popd > /dev/null
done
