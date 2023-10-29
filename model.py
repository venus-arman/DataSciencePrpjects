import os
import speech_recognition as sr
import pickle


class Model_Voice_Text():
    
    """
    This class takes the voices, convert them to text
    """
    #open and read the file after the appending:
    
    SR_obj = sr.Recognizer()
   
    def voice_to_text(self, voicefolder):
        SR_obj = sr.Recognizer()
        voicefolder = "C:/Users/syous/OneDrive/Documents/Freelance projects/DataSciencePrpjects/Voice records/"
        text_list = []

        for subdir, dirs, files in os.walk(voicefolder):
            for file in files:
                print(os.path.join(subdir, file))
                info = sr.AudioFile(os.path.join(subdir, file))
                print(info)
       
                with info as source:
                    SR_obj.adjust_for_ambient_noise(source)
                    audio_data = SR_obj.record(source,duration=100)
                    result = SR_obj.recognize_google(audio_data)
                    text_list.append(result)
        return(text_list)
    
model = Model_Voice_Text()
# path = "C:/Users/syous/OneDrive/Documents/Freelance projects/DataSciencePrpjects/Voice records/"
# model.voice_to_text(path)

pickle.dump(model, open("voice_txt.pkl", "wb"))




