CUDA_VISIBLE_DEVICES=0 python testing.py \
-d ./datasets/kodak \
--checkpoint ./weights/best_model.pth \
--input_size 224 \
--num_keep_patches 144 \
--output_path ./results \
--cuda