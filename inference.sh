MODEL=$1
DATA_DIR=$2
OUTPUT_DIR=$3
EXT=$4

python preprocess_for_inference.py -d $DATA_DIR -o $OUTPUT_DIR -e  $EXT
CUDA_VISIBLE_DEVICES=0,1,2,3 python inference_from_audio.py -m $MODEL -d $DATA_DIR -o $OUTPUT_DIR



