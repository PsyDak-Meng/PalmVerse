import cv2
import PySimpleGUI as sg
from NN import create_NN_model
from hand_tracker import HandTracker
from numpy import flip, argmax
import json
import os

def process_label_dict(label_path):
    with open(label_path, 'r') as f:
        labels_dict = json.load(f)
    labels = [label.split("\\")[-1] for label in labels_dict.keys()]
    cls_2_labels = {i: labels[i] for i in range(len(labels))}
    return cls_2_labels

class CamHandTracker(HandTracker):
    def __init__(self, label_path):
        super().__init__()
        self.cls_2_labels = process_label_dict(label_path)

    def cam_track(self, waitkey, model, window):
        vid = cv2.VideoCapture(0)
        while True:
            ret, frame = vid.read()
            frame = flip(frame, axis=1)
            frame, hand_landmark = self.track(frame=frame)
            prediction = model.predict(hand_landmark, verbose=0)
            label = argmax(prediction)
            prediction = self.cls_2_labels[label]
            window['-IMAGE-'].update(data=cv2.imencode('.PNG', frame)[1].tobytes())
            window['-TEXT-'].update("The hand sign is: " + prediction)
            event, values = window.read(timeout=waitkey)
            if event == sg.WINDOW_CLOSED or event == 'Exit':
                break
        vid.release()

def main():
    model = create_NN_model()
    checkpoint_path = "checkpoints/cp.ckpt"
    
    model.build(input_shape=(1, 84))
    if "NN.keras.h5" not in os.listdir(os.getcwd()):
        model.save('NN.keras', save_format='h5')
    model.load_weights(checkpoint_path)

    cam_tracker = CamHandTracker("labels.json")

    sg.theme('DarkGrey4')  # You can experiment with other built-in themes

    layout = [
        [sg.Image(filename='', key='-IMAGE-', size=(800, 600))],
        [sg.Text("The hand sign is: ", key='-TEXT-', font=('Helvetica', 16), text_color='white')],
        [sg.Text("Enter text:"), sg.Multiline(default_text='', size=(30, 4), key='-TEXTAREA-')],
        [sg.Button('Speak', font=('Helvetica', 12), button_color=('white', 'firebrick')), sg.Button('Exit', font=('Helvetica', 12), button_color=('white', 'firebrick'))]
    ]

    window = sg.Window('Sign Language to Text', layout, finalize=True, size=(800, 600), resizable=True, element_padding=(0, 0), margins=(0, 0))

    cam_tracker.cam_track(waitkey=30, model=model, window=window)

    window.close()

if __name__ == '__main__':
    main()
