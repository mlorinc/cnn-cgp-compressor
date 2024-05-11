#!/bin/bash
set -e

compress_stats () {
    for file in train_statistics/fitness/*.csv; do
        rm -f $file.chromosomeless.zip $file.chromosome.zip $file.zip

        if [ "$1" = "all" ]; then
            echo "zipping stats file: $file"
            zip -j $file.zip $file
        else
            echo "generating short chromosomeless $file.chromosomeless"
            head -1 $file > $file.chromosomeless
            awk -F ',' -v OFS=',' '{ print $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, "\"\"" }' $file | tail -n +2 >> $file.chromosomeless
            echo "generating short chromosome $file.chromosome"
            head -1 $file > $file.chromosome
            tail -$TOP $file >> $file.chromosome
            echo "zipping the chromosomeless $file.chromosomeless"
            zip -j $file.chromosomeless.zip $file.chromosomeless
            echo "zipping the chromosome $file.chromosome"
            zip -j $file.chromosome.zip $file.chromosome
        fi
    done
}

compress_cgp_configs () {
    for file in cgp_configs/*.config; do
        if [ "$file" = "cgp_configs/*.config" ]; then
            break
        fi
        rm -f $file.zip
        echo "zipping configs file: $file"
        zip -j $file.zip $file
    done
}

for var in "$@"
do
    case $var in
    stats)
        compress_stats "lean"
        ;;
    stats-all)
        compress_stats "all"
        ;;
    cgp_configs)
        compress_cgp_configs
        ;;

    *)
        echo -n "unknown argument: $var"
        exit 1
        ;;
    esac
done
