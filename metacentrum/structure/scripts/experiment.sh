#!/bin/bash
# /storage/brno12-cerit/home/mlorinc/
set -e

merge_experiments () {
    if [ "$3" = "all" ]; then
        mode="stats-all"
    elif [ "$3" = "lean" ]; then
        mode="stats"
    else
        echo "unknown compression mode $3"
        exit 1
    fi


    mkdir -p $1
    mkdir -p $1/cgp_configs
    mkdir -p $1/train_statistics/fitness
    for file in $1_batch_*; do
        echo "merging $file ..."
        pushd . > /dev/null
        cd $file
        TOP=$2 /storage/brno12-cerit/home/mlorinc/scripts/compress.sh "$mode" cgp_configs || { echo "ignoring $file"; popd > /dev/null; continue; }
        popd > /dev/null
        for x in $file/cgp_configs/*.zip; do [[ -e $x ]] && mv $x $1/cgp_configs; done
        mv $file/train_statistics/fitness/*.zip $1/train_statistics/fitness
        cp $file/*.* $1/
    done

    if [ "$3" = "all" ]; then
        echo "zipping $3 into $1"
        zip -r $1.all.zip $1/cgp_configs/*.zip $1/train_statistics/fitness/*.zip $1/*.*
    fi

    if [ "$3" = "lean" ]; then
        echo "zipping $3 into $1"
        zip -r $1.lean.zip $1/cgp_configs/*.zip $1/train_statistics/fitness/*.csv.chromosome.zip  $1/train_statistics/fitness/*.csv.chromosomeless.zip $1/*.*
    fi    
}

case "$1" in
merge)
    [ -z "$2" ] && echo "missing argument of what experiment to merge" && exit 1
    [ -z "$3" ] && echo "missing argument of how many chromosomes to keep" && exit 2
    [ -z "$4" ] && ZIP="lean" || ZIP="$4"
    merge_experiments $2 $3 $ZIP
    ;;
*)
    echo -n "unknown argument: $1"
    exit 1
    ;;
esac

