'''CREDITS
Based on Keras' Seq2Seq tutorial
    https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html

Adapted from OOP version of above tutorial written by cohort peer working on similar project
    GitHub: MattD82
    Matt's project: https://github.com/MattD82/Seinfeld-Neural-seq2seq-Chatbot
'''

import pandas as pd
import numpy as np
import re

from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.callbacks import ModelCheckpoint

class Seq2Seq_Train_Annotation_Generator(object):
    '''
    Executes training for character-based seq2seq model that takes in a .csv file of Lyric-Annotation pairs,
    and trains a seq2seq (endcoder-decoder) model on said pairs. Best to run on AWS instances.

    Data info and model weights are saved as output files.

    Genius.com Lingo:
        "refs" = referents (lyric segment being explained)
        "tates" = annotations
    '''
    def __init__(self, model_name='base',
                        data_file_path='../data/genius_data.csv',
                        num_epochs=100, batch_size=64, latent_dim=256,
                        optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'],
                        input_text_col='ref_text', target_text_col='tate_text',
                        start_char='\v', end_char='\b'):

        self.model_name = model_name
        self.data_file_path = data_file_path
        self.weights_file_path = f'../models/{self.model_name}_weights.h5'

        self.num_epochs = num_epochs # 20, 40, 80, 100, 120
        self.batch_size = batch_size #64
        self.latent_dim = latent_dim # 256, 512

        self.optimizer = optimizer # rmsprop, adam
        self.loss = loss # 'categorical_crossentropy'
        self.metrics = metrics #['accuracy']

        self.input_text_col = input_text_col #'ref_text'
        self.target_text_col = target_text_col #'tate_text'
        self.start_token = start_token # '\v'
        self.end_token = end_token # '\b'

    def _get_text_arr(self):
        '''Reads in text data
        INPUT: filepath to .csv file of lyric-annotation pairs
        OUTPUT: lyric text array, annoation text array (2 text arrays)
        '''
        df = pd.read_csv(self.data_file_path)
        self.num_samples = df.shape[0]
        self.input_text_arr = df[self.input_text_col].values
        self.target_text_arr = df[self.target_text_col].values

    def _clean_text(self, txt_arr):
        '''Strips bookend whitespace, lowercases, and removes characters that are not
        letters, numbers, typical punctuation, spaces, or newline characters.
        TYPICAL PUNCTUATION: ?.!,';:#$@&%*-+=
        INPUT: text array
        OUTPUT: cleaned text array
        '''
        clean_txt_lines = list(txt_arr)
        for idx, line in enumerate(clean_txt_lines):
            line = line.lower().strip()
            line = re.sub(r"[^a-zA-Z?.!, ';:#$@&%*-+=\n\d+]", "", line, re.M)
            # collapses multiple spaces into just one space
            line = re.sub(r'(  +)', " ", line, re.M)
            clean_txt_lines[idx] = line
            # line = self.start_token + line + self.end_token
        return clean_txt_lines

    def prep_text(self):
        '''Pulls in each text array and puts them through cleaner function
        INPUT: None
        OUTPUT: Cleaned lyric & annotation text arrays
        '''
        self._get_text_arr()
        self.clean_input_text_arr = self._clean_text(self.input_text_arr)
        self.clean_target_text_arr = self._clean_text(self.target_text_arr)

    def parse_txt(self, num_duplicate_pairs=1):
        '''Parses through text to get necessary info on tokens for later vectorization

            **MattD82 found that duplicating examples improved results in his model,
            so I included that as an argument here, although I'm not using it for my project (yet!)**

        INPUT: cleaned input and target text arrays (input = lyrics, target = annotations)
        OUTPUT: text data stats, token lookup dictionaries
        '''
        # Initializing here, for reuse purposes - always reset each time we execute this function
        self.input_texts = []
        self.target_texts = []
        self.input_characters = set()
        self.target_characters = set()

        for txt_idx in range(self.num_samples):
            input_text = self.clean_input_text_arr[txt_idx]
            target_text = self.clean_target_text_arr[txt_idx]
            target_text = self.start_char + target_text + self.end_char # adding start/end tokens to target text

            for _ in range(num_duplicate_pairs):
                self.input_texts.append(input_text)
                self.target_texts.append(target_text)

            # add unique chars to input and output sets
            for char in input_text:
                if char not in self.input_characters:
                    self.input_characters.add(char)
            for char in target_text:
                if char not in self.target_characters:
                    self.target_characters.add(char)

        if num_duplicate_pairs > 1:
            self.num_samples = len(self.input_texts)

        self.input_characters = sorted(list(self.input_characters))
        self.target_characters = sorted(list(self.target_characters))

        # calculate stats for this dataset of inputs and targets
        self.num_encoder_tokens = len(self.input_characters)
        self.num_decoder_tokens = len(self.target_characters)
        self.max_encoder_seq_length = max([len(txt) for txt in self.input_texts])
        self.max_decoder_seq_length = max([len(txt) for txt in self.target_texts])
        self.avg_encoder_seq_length = np.mean([len(txt) for txt in self.input_texts])
        self.avg_decoder_seq_length = np.mean([len(txt) for txt in self.target_texts])

        # printing stats for now
        print(f"Unique encoder tokens: {self.num_encoder_tokens}")
        print(f"Unique decoder tokens: {self.num_decoder_tokens}")
        print(f"Max encoder seq length: {self.max_encoder_seq_length}")
        print(f"Max decoder seq length: {self.max_decoder_seq_length}")
        print(f"Avg encoder seq length: {self.avg_encoder_seq_length}")
        print(f"Avg decoder seq length: {self.avg_decoder_seq_length}")

        # create dict of stats and save to .npy file
        text_stats = {}
        text_stats['num_encoder_tokens'] = self.num_encoder_tokens
        text_stats['num_decoder_tokens'] = self.num_decoder_tokens
        text_stats['max_encoder_seq_length'] = self.max_encoder_seq_length
        text_stats['max_decoder_seq_length'] = self.max_decoder_seq_length
        text_stats['avg_encoder_seq_length'] = self.avg_encoder_seq_length
        text_stats['avg_decoder_seq_length'] = self.avg_decoder_seq_length
        np.save(f'./{self.current_model}_text_stats.npy', text_stats)

        # create and save dicts of chars to indices and reverse for encoding and decoding one-hot values
        self.input_token_index = dict([(char, i) for i, char in enumerate(self.input_characters)])
        self.reverse_input_char_index = dict((i, char) for char, i in self.input_token_index.items())
        np.save(f'./{self.current_model}_input_token_index.npy', self.input_token_index)
        np.save(f'./{self.current_model}_reverse_input_char_index.npy', self.reverse_input_char_index)

        self.target_token_index = dict([(char, i) for i, char in enumerate(self.target_characters)])
        self.reverse_target_char_index  = dict((i, char) for char, i in self.target_token_index.items())
        np.save(f'./{self.model_name}_target_token_index.npy', self.target_token_index)
        np.save(f'./{self.model_name}_reverse_target_char_index.npy', self.reverse_target_char_index)

        self._save_actual_pairs_used()
        self._vectorize_text()

    def _save_actual_pairs_used(self):
        '''Saves the text pairs that are actually used in training, for later reference
        INPUT: model name, input and target token index lookup dictionaries, input and target texts
        OUTPUT: writes input and target token lookup dictionaries and actual input/target texts to text file and saves
        '''
        outF = open(f"./{self.model_name}_pairs_used.txt", "w")
        outF.write(str(self.input_token_index))
        outF.write("\n")
        outF.write(str(self.target_token_index))
        outF.write("\n")
        for lyric, annotation in zip(self.input_texts, self.target_texts):
            outF.write(lyric)
            outF.write(annotation)
        outF.close()

    def _vectorize_text(self):
        '''Vectorizes input and target texts
        INPUT: text arrays
        OUTPUT: 2 input and 1 target vectors for training
        '''
        self.encoder_input_data = np.zeros((self.num_samples,
                                           self.max_encoder_seq_length,
                                           self.num_encoder_tokens),
                                           dtype='float32')
        self.decoder_input_data = np.zeros((self.num_samples,
                                           self.max_decoder_seq_length,
                                           self.num_decoder_tokens),
                                           dtype='float32')
        self.decoder_target_data = np.zeros((self.num_samples,
                                            self.max_decoder_seq_length,
                                            self.num_decoder_tokens),
                                            dtype='float32')

        # loop to convert each sequence of chars to one-hot encoded 3D vectors
        for i, (input_text, target_text) in enumerate(zip(self.input_texts, self.target_texts)):
            for t, char in enumerate(input_text):
                self.encoder_input_data[i, t, self.input_token_index[char]] = 1.
            for t, char in enumerate(target_text):
                self.decoder_input_data[i, t, self.target_token_index[char]] = 1.
                # decoder_target_data is ahead of decoder_input_data by one timestep
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    self.decoder_target_data[i, t - 1, self.target_token_index[char]] = 1.

    def train_model(self):
        '''Trains model based on specifications
        INPUT: vectorized texts (encoder input, decoder input, decoder target)
        OUTPUT: trained model - saves best weights and final weights (to be loaded into inference model)
        '''
        # Define encoder model input and LSTM layers and states
        encoder_inputs = Input(shape=(None, self.num_encoder_tokens), name='encoder_inputs')

        encoder = LSTM(self.latent_dim, return_state=True, name='encoder_LSTM')
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)

        # discard 'encoder_outputs' and only keep the h anc c states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using 'encoder_states' as initial state.
        # We set up our decoder to return full output sequences (Jerry lines)
        # and to return internal states as well.
        # We don't use the return states in the training model, but we will use them in inference.

        decoder_inputs = Input(shape=(None, self.num_decoder_tokens), name='decoder_inputs')
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax', name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        print(self.model.summary())

        checkpoint = ModelCheckpoint(filepath=self.weights_file_path, save_best_only=True,
                                     save_weights_only=True, verbose=1)

        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

        self.history = self.model.fit([self.encoder_input_data, self.decoder_input_data],
                        self.decoder_target_data, batch_size=self.batch_size,
                        epochs=self.num_epochs, validation_split=0.1, callbacks=[checkpoint])

        with open(f'./trainHistoryDict_{self.model_name}.pkl', 'wb') as file_pi:
            pickle.dump(self.history.history, file_pi)

        self.model.save_weights(f'./{self.model_name}_final_weights.h5')

if __name__ == "__main__":
    ag = Seq2Seq_Train_Annotation_Generator(model_name='base',
                                        num_epochs=100,
                                        latent_dim=256,
                                        optimizer='adam')
    ag.prep_text()
    ag.parse_txt(num_duplicate_pairs=1)
    ag.train_model()
