import speech_recognition as sr
def voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print('Say Something ')
        audio = recognizer.listen(source)

    try:
        print(f'{recognizer.recognize_google(audio)}')
    except :
        pass