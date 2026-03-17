import speech_recognition as sr

class SpeechToText:
    @staticmethod
    def print_mic_device_index():
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            print(name, ", device_index=", index)

    @staticmethod
    def speech_to_text(device_index=0, language = "en-US"):
        r = sr.Recognizer()
        text = None
        with sr.Microphone(device_index = device_index) as source : 
            print("Start talking")
            while not text : 
                audio = r.listen(source)
                try : 
                    text = r.recognize_google(audio, language = language)
                    print("You said : ", text)
                except:
                    print("Please try again")
        return text

if __name__ == "__main__":
    SpeechToText.speech_to_text(device_index=1)