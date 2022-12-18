import os
import cv2
from tqdm import tqdm
import argparse
import pandas as pd
import numpy as np


STRIDE = 1.0
MAX_IMAGE_SIZE = 1920
N_FRAMES = 3
START_FRAME_SEC = 0.2
LABELS = ['liveness', 'spoof']

def get_frames_from_video(video_file):
    """
    video_file - path to file
    stride - i.e 1.0 - extract frame every second, 0.5 - extract every 0.5 seconds
    return: list of images, list of frame times in seconds
    """
#     root = input_path
#     video = cv2.VideoCapture(os.path.join(root, video_file))
    video = cv2.VideoCapture(video_file)
    frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    duration = frames/ fps
#     print(frames)
    i = duration * START_FRAME_SEC
    stride = (duration - i) / N_FRAMES
    images = []
    frame_times = []
    video.set(cv2.CAP_PROP_POS_MSEC, i * 1000)
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            images.append(frame)
            frame_times.append(i)
            i += stride
            video.set(1, round(i * fps))
        else:
            video.release()
            break
    return images, frame_times


def resize_if_necessary(image, max_size=MAX_IMAGE_SIZE):
    """
    if any spatial shape of image is greater 
    than max_size, resize image such that max. spatial shape = max_size,
    otherwise return original image
    """
    if max_size is None:
        return image
    height, width = image.shape[:2]
    if max([height, width]) > max_size:
        ratio = float(max_size / max([height, width]))
        image = cv2.resize(image, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
    return image

# sample_video= '9.avi'
# images, frame_times = get_frames_from_video(sample_video)
# images = [resize_if_necessary(image, MAX_IMAGE_SIZE) for image in images]

def resize_if_necessary(image, max_size=MAX_IMAGE_SIZE):
    """
    if any spatial shape of image is greater 
    than max_size, resize image such that max. spatial shape = max_size,
    otherwise return original image
    """
    if max_size is None:
        return image
    height, width = image.shape[:2]
    if max([height, width]) > max_size:
        ratio = float(max_size / max([height, width]))
        image = cv2.resize(image, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
    return image

import warnings
warnings.filterwarnings('ignore')

# Extract frames from test dataset
def extract_test_data(root, save_path):
    for video_id in tqdm(os.listdir(root)):
        id = video_id.split('.')[0]
        sample_video = os.path.join(root, video_id)
        images, frame_times = get_frames_from_video(sample_video)
        os.makedirs(os.path.join(save_path, id), exist_ok=True)
        path2save = os.path.join(save_path, id)
        for i, (image, frame_time) in enumerate(zip(images, frame_times)):
            ps = os.path.join(path2save, f'{id}_{i}.jpg')
            # image = resize_if_necessary(image)
            cv2.imwrite(ps, image)

# Extract liveness from train dataset
def extract_liveness_data(root, save_path):
    os.makedirs(f'{save_path}/liveness/images', exist_ok=True)
    save_img_path = f'{save_path}/liveness/images'
    for i, video_id in tqdm(enumerate(liveness_ids)):
        id = video_id.split('.')[0]
        save_single_folder = f'{save_img_path}/{id}'
        os.makedirs(save_single_folder, exist_ok=True)
        sample_video = os.path.join(root, video_id)
        images, frame_times = get_frames_from_video(sample_video)
        for k, (image, frame_time) in enumerate(zip(images, frame_times)):
            ps = os.path.join(save_single_folder, f'{id}_{k}.jpg')
            image = resize_if_necessary(image)
            cv2.imwrite(ps, image)

# Extract spoof from dataset
def extract_spoof_data(root, save_path):
    os.makedirs(f'{save_path}/spoof/images', exist_ok=True)
    save_img_path = f'{save_path}/spoof/images'
    for i, video_id in tqdm(enumerate(spoof_ids)):
        id = video_id.split('.')[0]
        save_single_folder = f'{save_img_path}/{id}'
        os.makedirs(save_single_folder, exist_ok=True)
        sample_video = os.path.join(root, video_id)
        images, frame_times = get_frames_from_video(sample_video)
        for k, (image, frame_time) in enumerate(zip(images, frame_times)):
            ps = os.path.join(save_single_folder, f'{id}_{k}.jpg')
            image = resize_if_necessary(image)
            cv2.imwrite(ps, image)

def preprocess(video_path):
    images, frame_times = get_frames_from_video(video_path)
    return images[:2]
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dir', type=str, help='Folder contains original train dataset')
    parser.add_argument('--test_dir', type=str, help='Folder contains original test dataset')
    parser.add_argument('--train_save_dir', type=str, help='Folder save seq frames train dataset')
    parser.add_argument('--test_save_dir', type=str, help='Folder save seq frames test dataset')
    parser.add_argument('--label_dir', type=str, help='Label dir', default='./data/train/label.csv')
    args = parser.parse_args()


    label_df = pd.read_csv(args.label_dir)

    liveness_ids = label_df[label_df['liveness_score'] == 1]['fname'].values.tolist()
    spoof_ids = label_df[label_df['liveness_score'] == 0]['fname'].values.tolist()

    TRAIN_DIR = args.train_dir
    TEST_DIR = args.test_dir

    TEST_SAVE_PATH = args.test_save_dir
    TRAIN_SAVE_PATH = args.train_save_dir

    # os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.makedirs(TRAIN_SAVE_PATH, exist_ok=True)
    extract_liveness_data(TRAIN_DIR, TRAIN_SAVE_PATH)
    extract_spoof_data(TRAIN_DIR, TRAIN_SAVE_PATH)
    # extract_test_data(root=TEST_DIR, save_path=TEST_SAVE_PATH)
