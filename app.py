from flask import Flask, render_template, Response
import cv2
from NN import create_NN_model
from cam import CamHandTracker
import os

app = Flask(__name__)
model = create_NN_model()
checkpoint_path = "checkpoints/cp.ckpt"

# Loads the weights
model.build(input_shape=(1, 84))
if "NN.keras.h5" not in os.listdir(os.getcwd()):
    model.save('NN.keras', save_format='h5')
model.load_weights(checkpoint_path)

cam_tracker = CamHandTracker("labels.json")

def generate_frames():
    vid = cv2.VideoCapture(0)

    while True:
        success, frame = vid.read()
        frame = cv2.flip(frame, 1)
        frame, hand_landmark = cam_tracker.track(frame=frame)
        prediction = model.predict(hand_landmark, verbose=0)
        label = argmax(prediction)
        prediction = cam_tracker.cls_2_labels[label]

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
