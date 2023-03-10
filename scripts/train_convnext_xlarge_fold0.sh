python -m torch.distributed.run --nproc_per_node 2 main.py \
        --epochs 12 \
        --model convnext_xlarge \
        --model_ema True \
        --batch_size 32 \
        --input_size 128 \
        --data_set IMNET \
        --data_path train_images/3_frames/fold0 \
        --nb_classes 2 \
        --drop_path 0.2 \
        --num_workers 8 \
        --warmup_epochs 4 \
        --save_ckpt true \
        --output_dir checkpoints/convnext_xlarge_fold0_128_bs_32_ema \
        --finetune weights/convnext_xlarge_22k_1k_224_ema.pth\
        --cutmix 0 \
        --mixup 0 \
        --hflip 0.5 \
        --vflip 0.5 \
        --color_jitter 0 \
        --lr 1e-4 \
        --min_lr 1e-6 \
        --project convnext_xlarge_fold0_128_bs_32_ema \
        --seed 69