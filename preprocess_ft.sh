INPUT_DIR=$1
OUTPUT_DIR=$2
CUDA_VISIBLE_DEVICES=0 python preprocess.py -i $INPUT_DIR -o $OUTPUT_DIR --save_audio