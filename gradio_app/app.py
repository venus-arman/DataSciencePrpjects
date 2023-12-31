# import os
# import speech_recognition as sr
# import pickle
# import nltk
# from nltk.corpus import wordnet
import pandas as pd
import difflib
import gradio as gr
from transformers import pipeline
import librosa

# import numpy as np

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base")


# nltk.download('wordnet')



class Model_Voice_Text():
    
    """
    This class takes the voices, convert them to text
    """
    #open and read the file after the appending:

    def __init__(self) -> None:
        # self.SR_obj = sr.Recognizer()
        self.KEYWORDS = ['suicide', 'urgent', 'poor', 'in-need', 'old', 'pregnant', 'refugee', 'new immigrant', 'patient', 'ill', 'sick', 'anxiety', 'anxious']
        # self.fuzzer = fuzz.Fuzz()
    
    # Define a function to find the number of times the word similar to the word stored in variable target_var, in a text stored in a variable named text_res
    def find_similar_word_count(self, text, target_var):
        """Finds the number of times the word similar to the word stored in variable target_var, in a text stored in a variable named text_res using difflib.

        Args:
            text: The text to search.
            target_var: The word to find the similar word to.

        Returns:
            The number of times the word similar to target_var appears in the text.
        """

        # Create a list of all words in the text
        words = text.split()

        # Find all words similar to target_var
        similar_words = difflib.get_close_matches(target_var, words, cutoff=0.75)

        # Return the number of similar words
        return len(similar_words)


    def matching_text(self, text):
        df = pd.DataFrame()
        ph_num = '000'
        sin = '0000'
        ret = []
        # words = nltk.word_tokenize(text)
        for target_var in self.KEYWORDS:
            count = self.find_similar_word_count(text, target_var)
            
            # matches = process.extract(text, word)
            if count>0:
                ret.append(target_var)
                ret.append(count)
        if ret == []:
            ret.append("nothing found")
        

        
        # initialize data of lists. 
        data = {'Keywords': [ret], 
                'Phone Number': ph_num,
                'SIN': sin,
                'text': text} 
        df = pd.DataFrame(data)
        
        # ret.append(text)
        return df
    
    def transcribe(self, audio_f):
        # sr, y = audio
        # y = y.astype(np.float32)
        # y /= np.max(np.abs(y))
        # print(type(audio))
        text = ""

        # First load the file
        audio, sr = librosa.load(audio_f)

        # Get number of samples for 20 seconds; replace 20 by any number
        buffer = 20 * sr

        samples_total = len(audio)
        samples_wrote = 0
        counter = 1

        while samples_wrote < samples_total:

            #check if the buffer is not exceeding total samples 
            if buffer > (samples_total - samples_wrote):
                buffer = samples_total - samples_wrote

            block = audio[samples_wrote : (samples_wrote + buffer)]
            # out_filename = "split_" + str(counter) + "_" + audio_f

            # Write 2 second segment
            # sf.write(out_filename, block, sr)

            # Transcribing the audio to text
            text += transcriber(block)["text"]
            counter += 1
            samples_wrote += buffer
            # print(counter)
            # print(text)

        return text
    
    def voice_to_text_s(self, audio):
        # SR_obj = self.SR_obj
        # info = sr.AudioFile(audio)
        tran_text = self.transcribe(audio)
        # print(tran_text)
        match_results = self.matching_text(tran_text.lower())
        return match_results

        # print(info)

        # with info as source:
        #     SR_obj.adjust_for_ambient_noise(source)
        #     audio_data = SR_obj.record(source,duration=100)
        #     result = SR_obj.recognize_google(audio_data)
        #     match_results = self.matching_text(result)
        #     return match_results

   
    # def voice_to_text(self, voicefolder):
    #     SR_obj = self.SR_obj
    #     text_list = []
    #     res_list = []

    #     for subdir, dirs, files in os.walk(voicefolder):
    #         for file in files:
    #             print(os.path.join(subdir, file))
    #             info = sr.AudioFile(os.path.join(subdir, file))
    #             print(info)
       
    #             with info as source:
    #                 SR_obj.adjust_for_ambient_noise(source)
    #                 audio_data = SR_obj.record(source,duration=100)
    #                 result = SR_obj.recognize_google(audio_data)
    #                 text_list.append(result)
    #                 match_results = self.matching_text(result)
    #                 res_list.append([file, match_results, result])

    #     return(text_list, res_list)


model = Model_Voice_Text()

# path = "/home/si-lab/Desktop/Projects/DataSciencePrpjects/Voice_records"
# text, results = model.voice_to_text(path)

# f = open("demofile2.txt", "a")
# f.write(text)
# f.close()
# df = pd.DataFrame(results)
# df.to_csv("list.csv", index=False)

demo = gr.Blocks()


micro_ph = gr.Interface(fn=model.voice_to_text_s,
             inputs=gr.Audio(source="microphone", type="filepath"),
             outputs=gr.Dataframe(label="Output Box", interactive=True))

file_ph = gr.Interface(fn=model.voice_to_text_s,
             inputs=gr.Audio(source="upload", type="filepath"),
             outputs=gr.Dataframe(label="Output Box", interactive=True))


with demo:
    gr.TabbedInterface(
        [micro_ph, file_ph],
        ["Transcribe Microphone", "Transcribe Audio File"],
    )

demo.launch(debug=True)

# pickle.dump(model, open("voice_txt.pkl", "wb"))




