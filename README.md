# Translingua
This project is based on machine translation where I used encoder-decoder LSTM model to translate the english sentence into german. 
process which i applied are given below:
1- data cleaning.
2- preprocessing: convert the input sentence into lower and remove unnecessary syntax and generate tokenization using tensorflow "Tokenizer()" like "Tokenize().word_index" and "Tokenizer.index_word".
3- generate Word embedding for encoder_input_data and decoder_input_data
4- choose encoder-decoder LSTM RNN Architecture
5- create the model
6- train the model
7- Create the encoder and decoder models for inference 
8- evaluate the quality of generated translations by actual translations.(BLEU SCORE)
