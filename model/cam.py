# import the opencv library 
import cv2 
from numpy import flip
from hand_tracker import HandTracker

class CamHandTracker(HandTracker):
    def __init__(self):
        super().__init__()
        pass 

    def cam_track(self,waitkey):
        # define a video capture object 
        vid = cv2.VideoCapture(0) 

        while(True): 
            # Capture the video frame by frame 
            ret, frame = vid.read() 
            frame = flip(frame,axis=1)
        
            # Display the resulting frame 
            frame = self.track(frame=frame)
            cv2.imshow('frame', frame) 
            cv2.waitKey(waitkey)

            # if cv2.waitKey() == ord('q'):
            #     cv2.destroyAllWindows()

def main():
    cam_tracker = CamHandTracker()
    cam_tracker.cam_track(waitkey=3)

if __name__ == "__main__":
    main()
