import speech_recognition as sr
def voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print('Say Something ')
        audio = recognizer.listen(source)

    try:
        speech = recognizer.recognize_google(audio)    
        # print(f'{speech}')
    except :
        pass
    return speech
# print(voice_input())
# print(voice_input())