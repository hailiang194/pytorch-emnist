import os
from pre_process_image import pre_process
import numpy as np
import cv2

TRAIN_PATH = './data/train/'
SAVING_PATH = './train.csv'

if __name__ == '__main__':
    train_data_folders = sorted(os.listdir(TRAIN_PATH))[1:]
    
    with open(SAVING_PATH, 'w') as saving_file:
        print('Resetting file')

    for image_label in train_data_folders:
        folder_path = TRAIN_PATH + image_label + '/'
        images = list(os.listdir(folder_path))
        
        label = int(image_label[-3:])
        print('Generating for ' + str(label))
        for image in images:
            print('    {}'.format(image))
            path = folder_path + image
            img = cv2.imread(path)
            flatten = pre_process(img)
            with open(SAVING_PATH, 'a') as saving_file:
                saving_file.write('{},'.format(label))
                saving_file.write(",".join(map(str, flatten)) + '\n')
