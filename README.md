# Translingua
This project is based on machine translation where I used encoder-decoder LSTM model to translate the english sentence into german. 
process which i applied are given below:
1- data cleaning.
2- preprocessing: convert the input sentence into lower and remove unnecessary syntax and generate tokenization using tensorflow "Tokenizer()" like "Tokenize().word_index" and "Tokenizer.index_word".
3- choose encoder-decoder LSTM RNN Architecture
4- create the model
5- train the model
6- Create the encoder and decoder models for inference 
7- evaluate the quality of generated translations by actual translations.(BLEU SCORE)
