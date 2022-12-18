python -m torch.distributed.run --nproc_per_node 2 main.py \
        --epochs 12 \
        --model convnext_large \
        --model_ema True \
        --batch_size 32 \
        --input_size 384 \
        --data_set IMNET \
        --data_path ../data/3_frames/fold0_pseudo \
        --nb_classes 2 \
        --drop_path 0.2 \
        --num_workers 8 \
        --warmup_epochs 4 \
        --save_ckpt true \
        --output_dir checkpoints/convnext_large_fold0_384_bs_32_ema_pseudo \
        --finetune /home/nhatnt/quangminh/competition/zindi/cls/ConvNeXt/weights/convnext_large_22k_224.pth
        --cutmix 0 \
        --mixup 0 \
        --hflip 0.5 \
        --vflip 0.5 \
        --color_jitter 0 \
        --lr 1e-4 \
        --min_lr 1e-6 \
        --use_amp \
        --project convnext_large_fold0_384_bs_32_ema_pseudo \
        --seed 69