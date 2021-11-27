CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu_finetune.py \
   -cfg='./configs/mae_base_patch16_224_finetune.yaml' \
   -dataset='imagenet2012' \
   -batch_size=32 \
   -data_path='/home/aistudio/data/ILSVRC2012mini' \
