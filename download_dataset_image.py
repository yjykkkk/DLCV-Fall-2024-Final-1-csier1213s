import datasets
import os 
import json
from PIL import Image
from tqdm import tqdm
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='download dataset')
    parser.add_argument('--split', type=str, default="test")
    args = parser.parse_args()
    split = args.split

    data_dir = './dataset'    
    dataset = datasets.load_dataset("ntudlcv/dlcv_2024_final1", split = split, streaming = True)
    
    # process save paths
    split_dir = os.path.join(data_dir, split)
    image_dir = os.path.join(split_dir, 'images')
    # json_path = os.path.join(split_dir, f'{split}_mod.json')
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    
    # output_data = []

    # get all the data from the split
    for item in dataset:
        id = item['id']
        image = item['image']
        image_save_name = f'{id}.png'
        image.save(os.path.join(image_dir,image_save_name))
        print(id, end='\r', flush = True)
