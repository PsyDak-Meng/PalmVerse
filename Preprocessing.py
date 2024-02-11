from hand_tracker import HandTracker
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import pickle as pkl
import json
import cv2



train_path = r"dataset\ASL\asl_alphabet_train"
test_path = "dataset/ASL/asl_alphabet_test"

h_tracker = HandTracker()

IMG_SIZE:int = 500




# Create look up dict
# function to get unique values
def unique(list1) -> list:
    # initialize a null list
    unique_list = []
 
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

def get_landmarks(IMG_SIZE):
    labels = [None]
    X_train = np.zeros((1,84))
    # Loop directory
    for label in tqdm(os.listdir(train_path)):
        cls_path = os.path.join(train_path,label)
        # if "D" in cls_path:
        #     print("terminated for test case")
        #     break
        for img_name in tqdm(os.listdir(cls_path)):
            f_name = os.path.join(cls_path,img_name)

            # Turn to landmark
            x = h_tracker.landmarks_from_grayscale(f_name,IMG_SIZE)
            x = x.reshape(1,x.shape[0])
            X_train = np.vstack((X_train,x))
            labels.append(cls_path)
        print(f"Done processing {label}.\n \
              X_trian size {X_train.shape}")

    # Remove dummy first row
    labels = labels[1:]
    X_train = X_train[1:,:]
    
    unique_labels = unique(labels)
    labels_dict = {i:unique_labels[i] for i in range(len(unique_labels))}
    labels_dict = {unique_labels[i]:i for i in range(len(unique_labels))}

    # labels to numbers
    labels = [labels_dict[labels[i]] for i in range(len(labels))]

    labels = np.array(labels)
    with open('X_train.pkl','wb') as f:
        pkl.dump(X_train,f)
        print(X_train.shape)
    with open('y_train.pkl','wb') as f:
        pkl.dump(labels,f)
        print(labels.shape)
    with open('labels.json', 'w') as fp:
        json.dump(labels_dict, fp)


def get_pixels(IMG_SIZE):
    labels = [None]
    X_train = np.zeros((1,84))
     # Loop directory
    for label in tqdm(os.listdir(train_path)):
        cls_path = os.path.join(train_path,label)
        for img_name in tqdm(os.listdir(cls_path)):
            f_name = os.path.join(cls_path,img_name)
            # Turn to landmark
            cap = cv2.VideoCapture(f_name)
            success, image = cap.read()
            image = image.reshape((1,))
            X_train = np.vstack((X_train,image))
            labels.append(cls_path)
        print(f"Done processing {label}.")

    # Remove dummy first row

    X_train = X_train[1:,:]
    

    with open('X_train_pixel.pkl','wb') as f:
        pkl.dump(X_train,f)
        print(X_train.shape)



if __name__=="__main__":
    get_landmarks(IMG_SIZE)
    
    
    




