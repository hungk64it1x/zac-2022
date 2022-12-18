python main.py --predict \
    --data_predict ../data/extract_data/public_test_4 \
    --model convnext_large\
    --batch_size 32 \
    --input_size 224 \
    --data_set IMNET \
    --data_path ../data/3_frames/fold0 \
    --nb_classes 2 \
    --num_frame_pred 2 \
    --finetune ./checkpoints/convnext_large_fold2_224_bs_32_ema/checkpoint-best.pth \
    --sample_submission_path sample/SampleSubmission_old.csv \
    --name convnext_large_fold2_224_bs_32_ema \



    
    
    

    
    
    
    
    