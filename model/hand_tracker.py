import cv2
import mediapipe as mp
import imutils
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from pandas import DataFrame

from model.CONSTANTS import WINDOW_SIZE

class HandTracker():
    def __init__(self, img_path=None,):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils
        self.img_path = img_path
        self.cap = cv2.VideoCapture(self.img_path)
        self.results = np.zeros((42,2)) # 21 points per hand [x,y]


    # Processing the input image
    def process_image(self,img) -> type:
        # Converting the input to grayscale
        if len(img.shape) == 3: 
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Reverse grayscale back to rgb
        elif len(img.shape) == 2:
            img = img.astype(np.float32)
            gray_image = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)  
            #print(gray_image.max(),gray_image.min(),gray_image)
            gray_image = gray_image*255.0
            gray_image = np.clip(gray_image, 0, 255).astype(np.uint8)
            gray_image = imutils.resize(gray_image, width=WINDOW_SIZE, height=WINDOW_SIZE)

        else:
            print("Wrong image input!")
            raise SystemError

        results = self.hands.process(gray_image)

        # Returning the detected hands to calling function
        return results

    # Drawing landmark connections
    def draw_hand_connections(self, img, results) -> NDArray:
        if results.multi_hand_landmarks:
            hand_count = 0
            for handLms in results.multi_hand_landmarks:
                #print(f"this is {hand_count+1} hand!")
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    # Finding the coordinates of each landmark
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    # Printing each landmark ID and coordinates
                    # on the terminal
                    #print(id, cx, cy)
                    #print(21*hand_count+id)
                    self.results[(21*hand_count+id),0] = cx
                    self.results[(21*hand_count+id),1] = cy

                    # Creating a circle around each landmark
                    cv2.circle(img, (cx, cy), 10, (0, 255, 0),
                            cv2.FILLED)
                    # Drawing the landmark connections
                    self.mpDraw.draw_landmarks(img, handLms,self.mpHands.HAND_CONNECTIONS)

                hand_count += 1
                # avoid index error
                if hand_count == 2:
                    hand_count = 0
            
        return img
    

    def track(self, frame=None): 
        """ frame control:
        Ndarray: use frame for cam.py input, 
        NoneType: local testing. """

        if frame is None:
            print("Local testing...")
            while True:
                # Taking the input
                success, image = self.cap.read()
                image = imutils.resize(image, width=WINDOW_SIZE, height=WINDOW_SIZE)
                results = self.process_image(image)
                self.draw_hand_connections(image,results)

                # Displaying the output
                cv2.imshow("Hand tracker", image)

                # Program terminates when q key is pressed
                if cv2.waitKey() == ord('q'):
                    self.cap.release()
                    cv2.destroyAllWindows()
                return image, None
        else:
            while True:
                image = frame
                image = imutils.resize(image, width=WINDOW_SIZE, height=WINDOW_SIZE)
                results = self.process_image(image)
                 # Return handlandmarks
                hand_landmarks = np.zeros((1,84))
                if results.multi_hand_landmarks:
                    hand_count = 0
                    for handLms in results.multi_hand_landmarks:
                        for id, lm in enumerate(handLms.landmark):
                            h, w, c = image.shape
                            # Finding the coordinates of each landmark
                            cx, cy = int(lm.x * w), int(lm.y * h)

                            #print(id, cx, cy)
                            hand_landmarks[0,20*hand_count+id] = cx
                            hand_landmarks[0,20*hand_count+id] = cy

                        hand_count += 1

                image = self.draw_hand_connections(image,results)
                
                return image, hand_landmarks
        

    # Training data processing 
    def landmarks_from_grayscale(self,img:str,img_size) -> NDArray: # 28*28 pixels grayscale images
        cap = cv2.VideoCapture(img)
        success, image = cap.read()

        hand_landmarks = np.zeros((1,84))
        image = imutils.resize(image, width=img_size, height=img_size)
        results = self.process_image(image)

        if results.multi_hand_landmarks:
            hand_count = 0
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = image.shape
                    # Finding the coordinates of each landmark
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    #print(id, cx, cy)
                    hand_landmarks[0,20*hand_count+id] = cx
                    hand_landmarks[0,20*hand_count+id] = cy

                hand_count += 1

        #print(f"shape of traing array is {hand_landmarks.shape}")
        return hand_landmarks


    
def main():
    img = 'test_2.jpg'
    h_tracker = HandTracker(img_path=img)
    h_tracker.track()



if __name__ == "__main__":
    main()


"""
Credits & Reference: https://www.makeuseof.com/python-hand-tracking-opencv/
"""