from flask import Flask, render_template, Response
from numpy import argmax
import timeit, time
import tensorflow as tf
import cv2
import os
from imutils.video import videostream

from model.NN import create_NN_model
from model.cam import CamHandTracker
from model.CONSTANTS import LPF


input_text=""
def main():
    # cv2 Webcam configuration
    vid = cv2.VideoCapture(0)
    if not vid.isOpened():
        raise RuntimeError('Could not start camera.')

    checkpoint_path = "model/checkpoints/cp.ckpt"

    app = Flask(__name__, template_folder='./templates')

    # Model Configuration
    model = create_NN_model()
    # Loads the weights
    model.build(input_shape=(1, 84))
    if "NN.keras.h5" not in os.listdir(os.getcwd()):
        model.save('NN', save_format='h5')
    model.load_weights(checkpoint_path)

    cam_tracker = CamHandTracker("dataset/labels.json")


    def generate_frames():
        global input_text
        print("Open live cam...")
        count = 0
        while True:
            success, frame = vid.read()
            frame = cv2.flip(frame, 1)
            frame, hand_landmark = cam_tracker.track(frame=frame)

            if count%LPF == 0:
                prediction = model.predict((hand_landmark,hand_landmark), verbose=0)
                label = argmax(prediction)
                prediction = cam_tracker.cls_2_labels[label]

                if prediction=="space":
                    input_text += " "
                elif prediction=="del":
                    input_text = input_text[:-2]
                elif prediction=='nothing':
                    pass
                else:
                    input_text += prediction
                # print(input_text)

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            end_time = timeit.default_timer()
            count += 1
            # print("FPS:", 1/(timeit.default_timer() - end_time))

            # yield (b'--frame\r\n'
            #     b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
 

    @app.route('/')
    def index():
        return render_template("mainpage.html")

    @app.route('/video_feed')
    def video_feed():
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    @app.route('/output_text')
    def output_text():
        global input_text
        return input_text
    
    @app.route('/delete_text', methods=['POST'])
    def delete_text():
        global input_text  # Access the global text variable
        input_text = ""  # Clear the text variable
        return 'Text deleted', 200
        

    app.run(host="0.0.0.0", debug=True, port=5000)


if __name__ == "__main__":

    main()
    
