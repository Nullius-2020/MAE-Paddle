CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu_finetune.py \
-cfg='./configs/mae_base_patch16_224_finetune.yaml' \
-dataset='imagenet2012' \
-batch_size=128 \
-data_path='/home/aistudio/data/ILSVRC2012mini' \
-eval \
-pretrained='./output/train-20211123-17-27-45/MAE-Epoch-20-Loss-5.221875454711914'
