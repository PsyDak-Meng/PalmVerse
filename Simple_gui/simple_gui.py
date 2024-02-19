import cv2
import PySimpleGUI as sg
from hand_tracker import HandTracker
from numpy import flip, argmax
import json
import os
import pyttsx3
import threading
import openai 
import google.generativeai as genai
#from google.colab import userdata

from CONSTANTS import CONFIDENCE
from NN import create_NN_model


# Or use `os.getenv('GOOGLE_API_KEY')` to fetch an environment variable.
#GOOGLE_API_KEY=userdata.get('AIzaSyAlHvgQc9gUO8YpeLy27yjyEf64iuEGaS0')
genai.configure(api_key="AIzaSyAlHvgQc9gUO8YpeLy27yjyEf64iuEGaS0")


openai.api_key = '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'
#client = openai.OpenAI()

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
        self.current_letter = ''
        self.engine = pyttsx3.init()
        self.thread_lock = threading.Lock()
        self.confidence = '-'
        self.output_text = ''


    def speak_text(self, text):
        with self.thread_lock:
            self.engine.say(text)
            self.engine.runAndWait()


    def cam_track(self, waitkey, model, window):
        vid = cv2.VideoCapture(0)
        output_text = ""  # Initialize the output text
        count = 0

        @staticmethod
        def clear_text():
            self.output_text = ''
            window['-OUTPUT-'].update(self.output_text)
            self.current_letter = ''
            current_letter = 'nothing'

        @staticmethod
        def gpt_fix_words(text):
            # all client options can be configured just like the `OpenAI` instantiation counterpart
            #openai.base_url = "https://..."
            openai.default_headers = {"x-foo": "true"}

            print('sending a message...')
            completion = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": f"User : Finish this word for me: f{text}",
                    },
                ],
            )
            reply = completion.choices[0].message.content
            window['-OUTPUT-'].update(reply)
            self.current_letter = ''

        # @staticmethod
        # def gemini_fix_words(text):
        #     model = genai.GenerativeModel('gemini-pro')
        #     reply = ''
        #     response = model.generate_content(f"You are a personal language assistant helping me to change my phrase into the most proper phrases. Here is my phrase:{text}, make it into a proper sentence or phrase. I only need the response.")
        #     for chunk in response:
        #         reply += chunk.text + '\n'
        #     print(reply)
        #     window['-OUTPUT-'].update(reply)
        #     self.current_letter = 'nothing'
            
            

        while True:
            count += 1
            ret, frame = vid.read()
            frame = flip(frame, axis=1)
            frame, hand_landmark = self.track(frame=frame)

            # Inhibit frame sensitivity
            # if count%3 == 0:
            prediction = model.predict(hand_landmark, verbose=0)
            label = argmax(prediction)
            if prediction.max() > CONFIDENCE:
                current_letter = self.cls_2_labels[label]
                self.confidence = str(prediction.max())
            else:
                current_letter = 'nothing'
                self.confidence = '-'
            # else: 
            #     continue
            # # reset
            # if count == 1000:
            #     count = 0

            window['-IMAGE-'].update(data=cv2.imencode('.PNG', frame)[1].tobytes())
            window['-TEXT-'].update(f"The hand sign is: {current_letter}\nConfidence: {self.confidence}")

            if current_letter != self.current_letter:
                # Append to the output text only when the letter changes
                if current_letter == "nothing":
                    current_letter = ''
                elif current_letter == "del":
                    current_letter = ''
                    self.output_text = self.output_text[:-2]
                    # gemini_fix_words(output_text)
                elif current_letter == 'space':
                    current_letter = ' '
                    # gemini_fix_words(output_text)

                self.output_text += current_letter
                
                window['-OUTPUT-'].update(self.output_text)
            if current_letter != 'nothing':
                self.current_letter = current_letter

            event, values = window.read(timeout=waitkey)
            if event == '-SPEAK-':
                threading.Thread(target=self.speak_text, args=(self.output_text,), daemon=True).start()
            # elif event == '-AI-':
            #     threading.Thread(target=gemini_fix_words,args=(self.output_text,), daemon=True).start()
            elif event == '-CLEAR-':
                threading.Thread(target=clear_text, daemon=True).start()
            elif event == sg.WINDOW_CLOSED or event == 'Exit':
                break

        vid.release()

def main():
    model = create_NN_model()
    checkpoint_path = "checkpoints/cp.ckpt"
    
    model.build(input_shape=(1, 84))
    if "NN.keras.h5" not in os.listdir(os.getcwd()):
        model.save('NN.keras', save_format='h5')
    model.load_weights(checkpoint_path)

    cam_tracker = CamHandTracker("dataset/labels.json")

    sg.theme('GreenMono')  # You can change the theme as needed

    layout = [
        [sg.Image(filename='', key='-IMAGE-', size=(800, 600)),
         sg.Text("The hand sign is: ", key='-TEXT-', font=('Helvetica', 16), text_color='white')],
        [sg.Multiline("", key='-OUTPUT-', size=(100, 5), font=('Helvetica', 12))],
        [sg.Button('Speak', size=(10, 1), font=('Helvetica', 12), key='-SPEAK-'),
        #  sg.Button('AI', size=(10, 1), font=('Helvetica', 12),key='-AI-'),
         sg.Button('CLEAR', size=(10, 1), font=('Helvetica', 12), key='-CLEAR-'),
         sg.Button('Exit', size=(10, 1), font=('Helvetica', 12),key='-EXIT-')]
    ]

    window = sg.Window('Sign Language to Text', layout, finalize=True, size=(800, 600), resizable=True)

    cam_tracker.cam_track(waitkey=30, model=model, window=window)

    window.close()

if __name__ == '__main__':
    main()
