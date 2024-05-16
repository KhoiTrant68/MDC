#CUDA_VISIBLE_DEVICES=0 python training.py \
#-d ./datasets/imagenet \
#--checkpoint ./pretrained_models/mae_visualize_vit_large_ganloss.pth \
#--input_size 224 \
#--num_keep_patches 144 \
#--epochs 1000 \
#--batch_size 32 \
#--output_dir ./weights \
#--log_dir ./logs \
#--cuda

CUDA_VISIBLE_DEVICES=0 python training.py \
-d ./datasets/kodak_train \
--checkpoint ./pretrained_models/mae_visualize_vit_large_ganloss.pth \
--input_size 224 \
--num_keep_patches 144 \
--epochs 10 \
--batch_size 4 \
--output_dir ./weights \
--log_dir ./logs \
--cuda