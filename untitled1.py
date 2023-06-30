import streamlit as st
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
import numpy as np
from googletrans import Translator
# Google Translate setup for compare with predicted sentence
translator = Translator()

# Load the saved models
encoder_model = load_model('C:/Users/Umang/Downloads/encoder_model.h5')
decoder_model = load_model('C:/Users/Umang/Downloads/decoder_model.h5')

max_encoder_seq_length=6
max_decoder_seq_length=13

# Function to decode a new sentence
def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = ger_token.word_index['<start>']
    stop_condition=False
    decoded_sentence=''
    while not stop_condition:
        output_tokens,h,c=decoder_model.predict([target_seq]+states_value,verbose=0)
        sampled_token_index=np.argmax(output_tokens[0,-1,:])
        sampled_word=ger_token.index_word[sampled_token_index]
        if sampled_word != '<end>':
            decoded_sentence += ' '+sampled_word

        if (sampled_word == '<end>' or len(decoded_sentence.split()) > max_decoder_seq_length):
            stop_condition=True

        target_seq=np.zeros((1,1))
        target_seq[0,0]=sampled_token_index

        states_value=[h,c]
    return decoded_sentence

# Load and preprocess the data
data = pd.read_csv('C:/Users/Umang/Downloads/GER_ENN/GERMAN_ENGLISH_TRANSLATION.csv')
data = data.drop_duplicates(subset=['ENGLISH'])
data = data.head(20000)

english_sentences = data['ENGLISH'].str.lower().str.replace('[^\w\s]', '').tolist()
german_sentences = data['GERMAN'].str.lower().str.replace('[^\w\s]', '').apply(lambda x: '<start> ' + x + ' <end>').tolist()

# Preprocess the data and prepare the tokenizer
eng_token = Tokenizer(filters='')
eng_token.fit_on_texts(english_sentences)

ger_token = Tokenizer(filters='')
ger_token.fit_on_texts(german_sentences)


# Streamlit app
st.set_page_config(layout="wide")


# Streamlit app
st.title("Translation App: English-German")

# Text input
input_text = st.text_input("Enter an English sentence")

# Predict button
predict_button = st.button("Translate")

if predict_button and input_text:
    # Tokenize and pad the input sequence
    input_seq = eng_token.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_encoder_seq_length, padding='post')

    # Decode the input sequence
    translated_sentence = decode_sequence(input_seq)


    # Display the translated sentence
    st.write("Predicted German Sentence :", translated_sentence)
    # Translate the actual sentence using Google Translate
    translation = translator.translate(input_text, dest='german')
    st.write("Actual German Sentence:", translation.text)
