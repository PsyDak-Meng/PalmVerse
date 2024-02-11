from flask import Flask, render_template, Response
import cv2
from CNN_1D_ATTN import create_NN_model
from cam import CamHandTracker
import os
from numpy import argmax
import timeit
import tensorflow as tf
from CONSTANTS import FRAME_RATE, FPS, PIXEL_WINDOW
from imutils.video import videostream

app = Flask(__name__)

checkpoint_path = "checkpoints/cp.ckpt"

# Model Configuration
model = create_NN_model()
# Loads the weights
model.build(input_shape=(1, 84))
if "CNN_1D_ATTN.keras.h5" not in os.listdir(os.getcwd()):
    model.save('CNN_1D_ATTN', save_format='h5')
model.load_weights(checkpoint_path)

cam_tracker = CamHandTracker("labels.json")



# cv2 Webcam configuration
vid = cv2.VideoCapture(0)
if not vid.isOpened():
    raise RuntimeError('Could not start camera.')
# vid.set(5, FPS)
# vid.set(7, FRAME_RATE)
# vid.set(cv2.CAP_PROP_BUFFERSIZE, 2)


def generate_frames():
    global vid
    print("Open live cam...")
    while True:
        starttime = timeit.default_timer()
        success, frame = vid.read()
        frame = cv2.flip(frame, 1)
        frame, hand_landmark = cam_tracker.track(frame=frame)
        
        mid_time = timeit.default_timer()
        print("before model is :", timeit.default_timer() - starttime)
        prediction = model.predict((hand_landmark,hand_landmark), verbose=0)
        print("model inference is :", timeit.default_timer() - mid_time)
        
        label = argmax(prediction)
        prediction = cam_tracker.cls_2_labels[label]
        #print("The hand sign is :", prediction)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        print("one frame is :", timeit.default_timer() - starttime)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)
