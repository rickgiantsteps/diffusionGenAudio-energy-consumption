#!/bin/bash

SCRIPT_NAME=$(basename "$0" .sh)
OUTPUT_DIR="./output"
GLOBAL_OUTPUT_FILE="${OUTPUT_DIR}/${SCRIPT_NAME}_scores.csv"
BASELINE_DIR="../baseline"
GENAUDIO_DIR="../genaudios/${SCRIPT_NAME}"
BASELINES=("clotho_chosen" "audiocaps_chosen")

declare -A BASELINE_MAP
BASELINE_MAP["clotho_chosen"]="clotho"
BASELINE_MAP["audiocaps_chosen"]="audiocaps"

GENAUDIO_SUBFOLDERS=("10" "25" "50" "100" "150" "200")

mkdir -p "$OUTPUT_DIR"

first_file=1

for baseline in "${BASELINES[@]}"; do
    BASELINE_PATH="${BASELINE_DIR}/${baseline}"
    BASELINE_NAME="${BASELINE_MAP[$baseline]}"

    for subfolder in "${GENAUDIO_SUBFOLDERS[@]}"; do
        GENAUDIO_PATH="${GENAUDIO_DIR}/${BASELINE_NAME}/${subfolder}"
        echo "Processing: Baseline ${BASELINE_PATH} vs Generated ${GENAUDIO_PATH}"
        fadtk clap-laion-audio "$BASELINE_PATH" "$GENAUDIO_PATH" scores.csv --inf

        if [ ! -f scores.csv ]; then
            echo "No output file generated for baseline ${baseline} and subfolder ${subfolder}!"
            continue
        fi

        if [ $first_file -eq 1 ]; then
            mv scores.csv "$GLOBAL_OUTPUT_FILE"
            first_file=0
        else
            tail -n +2 scores.csv >> "$GLOBAL_OUTPUT_FILE"
            rm scores.csv
        fi
    done
done

echo "All processes completed! Combined results saved in ${GLOBAL_OUTPUT_FILE}"

TRANSFORMED_FILE="${OUTPUT_DIR}/${SCRIPT_NAME}_fad_scores.csv"
awk -F, 'BEGIN {
    OFS=",";
    print "model,steps,baseline,fad_score"
}
NR > 1 {
    n = split($3, parts, "/");
    print parts[4], parts[6], parts[5], $4
}' "$GLOBAL_OUTPUT_FILE" > "$TRANSFORMED_FILE"

rm "$GLOBAL_OUTPUT_FILE"
echo "Created file containing all FAD scores for the model: ${TRANSFORMED_FILE}"
