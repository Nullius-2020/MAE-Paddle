CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu_pretrain.py \
-cfg='./configs/mae_base_patch16_224.yaml' \
-dataset='imagenet2012' \
-batch_size=8 \
-data_path='/home/aistudio/data/ILSVRC2012mini' 
