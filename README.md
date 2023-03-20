## Preprocessing
### 1. Download dataset
[VCTK](https://datashare.ed.ac.uk/handle/10283/2651)  
[LibriTTS](https://www.openslr.org/60/)
### 2. Preprocessing for pre-training
```
INPUT_DIR=[Directory of LibriTTS]
OUTPUT_DIR=./dataset/LibriTTS
CUDA_VISIBLE_DEVICES=0,1 python preprocess.py -i $INPUT_DIR -o $OUTPUT_DIR
```
### 3. Preprocessing for fine-tuning
```
INPUT_DIR=[Directory of VCTK]
OUTPUT_DIR=./dataset/VCTK
CUDA_VISIBLE_DEVICES=0,1 python preprocess.py -i $INPUT_DIR -o $OUTPUT_DIR --save_audio
```
## Pre-training
```
PT_MODEL_NAME=pretrain_80
MASK_RATIO=0.8
CUDA_VISIBLE_DEVICES=0,1 python pretrain.py -m $MODEL_NAME -r $MASK_RATIO
```

## Fine-tuning
```
FT_MODEL_NAME=finetune_random
PT_MODEL_NAME=pretrain_80
CUDA_VISIBLE_DEVICES=0,1 python finetune.py -m $FT_MODEL_NAME -p $PT_MODEL_NAME
```
