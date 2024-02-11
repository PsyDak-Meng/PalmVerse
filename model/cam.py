from CNN_1D_ATTN import create_NN_model
import tensorflow as tf
import os
import cv2 
from numpy import flip, argmax
from hand_tracker import HandTracker
import json

def process_label_dict(label_path):
    with open(label_path,'r') as f:
        labels_dict = json.load(f)
    labels = []
    cls_2_labels = {}
    for label in labels_dict.keys():
        labels.append(label.split("\\")[-1])
    for i in range(len(labels)):
        cls_2_labels[i] = labels[i]
    return cls_2_labels
    

class CamHandTracker(HandTracker):
    def __init__(self,label_path):
        super().__init__()
        self.cls_2_labels = process_label_dict(label_path)
        pass 

    def cam_track(self,waitkey,model):
        # define a video capture object 
        vid = cv2.VideoCapture(0) 

        while(True): 
            # Capture the video frame by frame 
            ret, frame = vid.read() 
            frame = flip(frame,axis=1)
            # Display the resulting frame 
            frame, hand_landmark = self.track(frame=frame)
            prediction = model.predict(hand_landmark,verbose=0)
            label = argmax(prediction)
            prediction = self.cls_2_labels[label]
            print("The hand sign is: ",prediction)
            cv2.imshow('frame', frame) 
            cv2.waitKey(waitkey)

            # if cv2.waitKey() == ord('q'):
            #     cv2.destroyAllWindows()

def main(model):
    cam_tracker = CamHandTracker("labels.json")
    cam_tracker.cam_track(waitkey=3,model=model)

if __name__ == "__main__":
    model = create_NN_model()
    checkpoint_path = "checkpoints/cp.ckpt"

    # Loads the weights
    model.build(input_shape = (1,84))
    if "NN.keras.h5" not in os.listdir(os.getcwd()):
        model.save('NN.keras',save_format='h5')
    model.load_weights(checkpoint_path)
    #print(model.summary())

    main(model=model)
