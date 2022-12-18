python main.py --predict \
    --data_predict infer_data/ \
    --model convnext_xlarge\
    --batch_size 32 \
    --input_size 128 \
    --data_set IMNET \
    --data_path train_images/3_frames/fold0 \
    --nb_classes 2 \
    --num_frame_pred 2 \
    --finetune ./checkpoints/convnext_xlarge_fold0_128_bs_32_ema/checkpoint-best.pth \
    --sample_submission_path sample_submission.csv \
    --name convnext_xlarge_fold0_128_bs_32_ema \


    
    
    

    
    
    
    
    