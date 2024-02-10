import cv2
import mediapipe as mp
import imutils
import numpy as np
from numpy.typing import NDArray

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
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(gray_image)

        # Returning the detected hands to calling function
        return results

    # Drawing landmark connections
    def draw_hand_connections(self, img, results) -> NDArray:
        if results.multi_hand_landmarks:
            hand_count = 0
            for handLms in results.multi_hand_landmarks:
                print(f"this is {hand_count+1} hand!")
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape

                    # Finding the coordinates of each landmark
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    # Printing each landmark ID and coordinates
                    # on the terminal
                    print(id, cx, cy)
                    self.results[(20*hand_count+id),0] = cx
                    self.results[(20*hand_count+id),0] = cy

                    # Creating a circle around each landmark
                    cv2.circle(img, (cx, cy), 10, (0, 255, 0),
                            cv2.FILLED)
                    # Drawing the landmark connections
                    self.mpDraw.draw_landmarks(img, handLms,self.mpHands.HAND_CONNECTIONS)

                hand_count += 1
            
        return img
    

    def track(self, frame=None): 
        """ Use frame for cam.py input, leave as None for local testing. """

        if frame is None:
            while True:
                # Taking the input
                success, image = self.cap.read()
                image = imutils.resize(image, width=500, height=500)
                results = self.process_image(image)
                self.draw_hand_connections(image,results)

                # Displaying the output
                cv2.imshow("Hand tracker", image)

                # Program terminates when q key is pressed
                if cv2.waitKey() == ord('q'):
                    self.cap.release()
                    cv2.destroyAllWindows()
        else:
            while True:
                image = frame
                image = imutils.resize(image, width=500, height=500)
                results = self.process_image(image)
                image = self.draw_hand_connections(image,results)

                return image


    
def main():
    img = 'test_2.jpg'
    h_tracker = HandTracker(img_path=img)
    h_tracker.track()



if __name__ == "__main__":
    main()


"""
Credits & Reference: https://www.makeuseof.com/python-hand-tracking-opencv/
"""