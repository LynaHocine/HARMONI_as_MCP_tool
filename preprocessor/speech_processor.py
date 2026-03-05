import speech_recognition as sr

class SpeechToText:
    def print_mic_device_index():
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            print(name, ", device_index=", index)

    def speech_to_text(device_index, language = "en-US"):
        r = sr.Recognizer()
        with sr.Microphone(device_index = device_index) as source : 
            print("Start talking")
            audio = r.listen(source)
            try : 
                text = r.recognize_google(audio, language = language)
                print("You said : ", text)
            except:
                print("Please try again")

if __name__ == "__main__":
    SpeechToText.speech_to_text(device_index=1)