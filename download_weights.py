import gdown
import os

model_folder = 'checkpoints'
model_names = [
    'convnext_large_fold0_224_bs_32_ema', 
    'convnext_large_fold1_224_bs_32_ema', 
    'convnext_large_fold2_224_bs_32_ema',
    'convnext_xlarge_fold0_128_bs_32_ema',
    'convnext_xlarge_fold1_128_bs_32_ema',
    'convnext_xlarge_fold2_128_bs_32_ema',
    'convnext_xlarge_fold3_128_bs_32_ema',
    'convnext_xlarge_fold4_128_bs_32_ema'
    ]
weights_id = [
    '1_JlsQ3CHHCiO5LkCIgaPG-osyZ5CpRTf',
    '1Dwe7l8ArkkU4aOk5dx_KsxdoX6pIdqbM',
    '1wxNMCNIxBt8SewQESD-nzkdGxzumR0Zt',
    '1qAPsdf92LPQyvCke15PXDZ22pnAjqxie',
    '1fdq2ZQCnWqZ0IUOCpFg3PlrXNm2lu7Ht',
    '1ATvs-3Zxu5oUgwSQGCezOk9K74VVLsbY',
    '1_K_YPfDqQlg5xR7VRfIXPF6TuwtB6zzP',
    '1Df5IoAcCLuPJc7FSxMCZvy_JFLcsRbPM'
]

for name, w_id in zip(model_names, weights_id):
    w_path = os.path.join(model_folder, name)
    os.makedirs(w_path, exist_ok=True)
    weights_path = os.path.join(model_folder, name, 'checkpoint-best.pth')
    if not os.path.isfile(weights_path):
        url = f'https://drive.google.com/uc?id={w_id}'
        output = weights_path
        gdown.download(url, output, quiet=False)
