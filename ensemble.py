from main import Model
from extract_frame import preprocess
import time
import pandas as pd
import os
from tqdm import tqdm
import cv2
import numpy as np

models = []

# weights = [0.2, 0.3, 0.2, 0.15, 0.15, 0.2, 0.3, 0.2, 0.15, 0.15]
weights = [0.4, 0.4, 0.2, 0.2, 0.4, 0.2, 0.2]
checkpoints = [
    'checkpoints/convnext_large_fold0_224_bs_32_ema/checkpoint-best.pth',
    'checkpoints/convnext_large_fold1_224_bs_32_ema/checkpoint-best.pth',
    'checkpoints/convnext_large_fold2_224_bs_32_ema/checkpoint-best.pth',
    # 'checkpoints/convnext_large_fold3_224_bs_32_ema/checkpoint-best.pth',
    # 'checkpoints/convnext_large_fold4_224_bs_32_ema/checkpoint-best.pth',

    'checkpoints/convnext_xlarge_fold0_128_bs_32_ema/checkpoint-best.pth',
    'checkpoints/convnext_xlarge_fold1_128_bs_32_ema/checkpoint-best.pth',
    'checkpoints/convnext_xlarge_fold2_128_bs_32_ema/checkpoint-best.pth',
    'checkpoints/convnext_xlarge_fold3_128_bs_32_ema/checkpoint-best.pth',
    # 'checkpoints/convnext_xlarge_fold4_128_bs_32_ema/checkpoint-best.pth',
]
sizes = [224, 224, 224, 128, 128, 128, 128]
# sizes = [256, 256, 256, 224, 224, 224, 224]
# names = ['convnext_large', 'convnext_large', 'convnext_large', 'convnext_large', 'convnext_large', 'convnext_xlarge', 'convnext_xlarge', 'convnext_xlarge', 'convnext_xlarge', 'convnext_xlarge']
names = ['convnext_large', 'convnext_large', 'convnext_large', 'convnext_xlarge', 'convnext_xlarge', 'convnext_xlarge', 'convnext_xlarge']
print('Loading model ...')
for cp, s, name in tqdm(zip(checkpoints, sizes, names)):
    models.append(Model(name=name, checkpoint_model=cp, model_prefix='', size=s))

src = '../data/public_test/videos'
df = pd.read_csv('sample/SampleSubmission_old.csv')
test_cases_list = df['fname'].values.tolist()
test_cases = [os.path.join(src, i) for i in test_cases_list]

sub_df = pd.DataFrame(columns=['fname', 'liveness_score'])
sub_df['fname'] = test_cases_list


liveness_scores = []
for test_case in tqdm(test_cases):
    input = preprocess(test_case)
    outputs = []
    for i, model in enumerate(models):
        output = model.predict(input)

        outputs.append(weights[i] * output)
    score = 0.4 * sum(outputs[:3]) + 0.6 * sum(outputs[3:])
    liveness_scores.append(score[1])
    t2 = time.time()

sub_df['liveness_score'] = liveness_scores
sub_df.to_csv('submission/submission.csv', index=False)