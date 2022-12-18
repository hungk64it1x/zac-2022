import os
import shutil
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

src = 'train_images'
liveness_src = f'{src}/liveness/images'
spoof_src = f'{src}/spoof/images'
df = pd.read_csv(f'raw/label.csv')
N_FOLDS = 5
MULTI = True
NUM_FRAMES = 3

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
df['fold'] = -1

for i, (train_idx, val_idx) in enumerate(skf.split(df, df['liveness_score'])):
    df.loc[val_idx, 'fold'] = i 

print(df.groupby(['fold', 'liveness_score'])['fname'].size())

# public_test_src = f'{src}/extract_data/public_test'
# public_test_path = f'{src}/public/images'
# os.makedirs(public_test_path, exist_ok=True)

for fold in range(N_FOLDS):
    fold_path = f'{src}/{NUM_FRAMES}_frames/fold{fold}'
    train_path = f'{fold_path}/train'
    val_path = f'{fold_path}/val'
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(f'{train_path}/0', exist_ok=True)
    os.makedirs(f'{train_path}/1', exist_ok=True)
    os.makedirs(f'{val_path}/0', exist_ok=True)
    os.makedirs(f'{val_path}/1', exist_ok=True)
    os.makedirs(fold_path, exist_ok=True)
    train_df = df[df['fold'] != fold].reset_index(drop=True)
    val_df = df[df['fold'] == fold].reset_index(drop=True)
    
    if MULTI:
        for id, row in tqdm(train_df.iterrows()):
            image_id = row['fname'].split('.')[0]
            label = row['liveness_score']
    
            if label == 1:
                for i in range(NUM_FRAMES):
                    shutil.copy(f'{liveness_src}/{image_id}/{image_id}_{i}.jpg', f'{train_path}/{label}')
            if label == 0:
                for i in range(NUM_FRAMES):
                    shutil.copy(f'{spoof_src}/{image_id}/{image_id}_{i}.jpg', f'{train_path}/{label}')
        
        for id, row in tqdm(val_df.iterrows()):
            image_id = row['fname'].split('.')[0]
            label = row['liveness_score'] 
            if label == 1:
                for i in range(NUM_FRAMES):
                    shutil.copy(f'{liveness_src}/{image_id}/{image_id}_{i}.jpg', f'{val_path}/{label}')
            if label == 0:
                for i in range(NUM_FRAMES):
                    shutil.copy(f'{spoof_src}/{image_id}/{image_id}_{i}.jpg', f'{val_path}/{label}')
        # for image_id in os.listdir(public_test_src):
        #     for i in range(NUM_FRAMES):
        #         shutil.copy(f'{public_test_src}/{image_id}/{image_id}_{i}.jpg', f'{public_test_path}/{image_id}.jpg')
    else:
        for id, row in tqdm(train_df.iterrows()):
            image_id = row['fname'].split('.')[0]
            label = row['liveness_score']
            if label == 1:
                shutil.copy(f'{liveness_src}/{image_id}/{image_id}_0.jpg', f'{train_path}/{label}')
            if label == 0:
                shutil.copy(f'{spoof_src}/{image_id}/{image_id}_0.jpg', f'{train_path}/{label}')
        
        for id, row in tqdm(val_df.iterrows()):
            image_id = row['fname'].split('.')[0]
            label = row['liveness_score'] 
            if label == 1:
                shutil.copy(f'{liveness_src}/{image_id}/{image_id}_0.jpg', f'{val_path}/{label}')
            if label == 0:
                shutil.copy(f'{spoof_src}/{image_id}/{image_id}_0.jpg', f'{val_path}/{label}')
        for image_id in os.listdir(public_test_src):
            shutil.copy(f'{public_test_src}/{image_id}/{image_id}_0.jpg', f'{public_test_path}/{image_id}.jpg')

