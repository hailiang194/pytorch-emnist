import os
from pre_process_image import pre_process
import numpy as np
import cv2

TEST_PATH = './data/test/'
SAVING_PATH = './test.csv'

if __name__ == '__main__':
    test_images = sorted(os.listdir(TEST_PATH))
    
    with open(SAVING_PATH, 'w') as saving_file:
        print('Resetting file')
    
    labels = None
    with open('./test-label.txt', 'r') as label_file:
        labels = ([int(label[:-1]) for label in label_file.readlines()])

    for index, image in enumerate(test_images):
        print('{}'.format(image))
        path = TEST_PATH + image
        img = cv2.imread(path)
        flatten = pre_process(img)
        with open(SAVING_PATH, 'a') as saving_file:
            saving_file.write('{},'.format(labels[index]))
            saving_file.write(','.join(map(str, flatten)) + '\n')

