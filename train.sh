python main_compress.py \
-d ./datasets/imagenet \
--checkpoint ./checkpoint/mae_pretrain_vit_base.pth \
--epochs 1000 \
--batch_size 32 \
--output_dir ./output_dir \
--log_dir ./logs \
--cuda