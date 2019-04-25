import numpy as np
import re

from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.callbacks import ModelCheckpoint

class AnnotationGenerator(object):
    '''
    Character-based seq2seq model that uses pretrained weights to allow a user to pass in lyrics to be annotated/explained
    '''

    def __init__(self, use_weights=True, trained_model='base', models_filepath='./',
                data_filepath='./', final_weights_fp='_final_weights',
                data_name='baseline_data', latent_dim=256, temp=1, start_char='\v',
                end_char='\b'):
        self.trained_model = trained_model
        self.data_filepath = data_filepath
        self.models_filepath = models_filepath
        self.data_name = data_name
        self.final_weights_fp = final_weights_fp

        self.latent_dim = latent_dim # 256
        self.temp = temp # for sampling w/ diversity

        self.start_char = start_char
        self.end_char = end_char

        # load in model feature values
        self.text_stats = np.load(f'{self.data_filepath}{self.data_name}_text_stats.npy').item()
        self.num_encoder_tokens = self.text_stats['num_encoder_tokens']
        self.num_decoder_tokens = self.text_stats['num_decoder_tokens']
        self.max_encoder_seq_length = self.text_stats['max_encoder_seq_length']
        self.max_decoder_seq_length = self.text_stats['max_decoder_seq_length']

        # load in char to idx dictionary values
        self.input_token_index = np.load(f'{self.data_filepath}{self.data_name}_input_token_index.npy').item()
        self.reverse_input_char_index = np.load(f'{self.data_filepath}{self.data_name}_reverse_input_char_index.npy').item()
        self.target_token_index = np.load(f'{self.data_filepath}{self.data_name}_target_token_index.npy').item()
        self.reverse_target_char_index = np.load(f'{self.data_filepath}{self.data_name}_reverse_target_char_index.npy').item()

        # Define encoder model input and LSTM layers and states exactly as defined in training model
        if use_weights:
            encoder_inputs = Input(shape=(None, self.num_encoder_tokens), name='encoder_inputs')
            encoder = LSTM(self.latent_dim, return_state=True, name='encoder_LSTM')
            encoder_outputs, state_h, state_c = encoder(encoder_inputs)
            encoder_states = [state_h, state_c]

            decoder_inputs = Input(shape=(None, self.num_decoder_tokens), name='decoder_inputs')
            decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')
            decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
            decoder_dense = Dense(self.num_decoder_tokens, activation='softmax', name='decoder_dense')
            decoder_outputs = decoder_dense(decoder_outputs)

            self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
            self.model.load_weights(f'{self.models_filepath}{self.trained_model}{self.final_weights_fp}.h5')
        else:
            self.model = load_model(f'{self.models_filepath}{self.trained_model}.h5')

        # create encoder and decoder models for prediction
        self.encoder_model = Model(encoder_inputs, encoder_states)
        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    def _clean_text(self, txt):
        '''
        For now, data is small enough for regex to still be reasonable... Let's go with this for now
        '''
        txt = txt.lower().strip()
        # clean_txt_lines[idx] = ''.join([i for i in line if i in chars_to_keep_lst])
        txt = re.sub(r"[^a-zA-Z?.!, ';:#$@&%*-+=\n\d+]", "", txt, re.M)
        # collapses multiple spaces into just one space
        txt = re.sub(r'(  +)', " ", txt, re.M)
        # line = self.start_token + line + self.end_token
        return txt

    def _encode_input_sentence(self, sentence):
        sentence = self._clean_text(sentence)
        if len(sentence) > self.max_encoder_seq_length:
            sentence = sentence[:self.max_encoder_seq_length]
        encoder_input_sent = np.zeros((1,
                                       self.max_encoder_seq_length,
                                       self.num_encoder_tokens),
                                       dtype='float32')
        for t, char in enumerate(sentence):
            encoder_input_sent[0, t, self.input_token_index[char]] = 1
        return encoder_input_sent

    def _sample_with_diversity(self, preds, temperature=None):
        temperature = temperature if temperature else self.temp
        # same as Erin's function
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def reply(self, sentence, diversity=False, temp=None):
        self.encoder_input_sent = self._encode_input_sentence(sentence)
        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(self.encoder_input_sent)
        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, self.target_token_index[self.start_char]] = 1.
        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)
            if diversity:
                sampled_token_index = self._sample_with_diversity(output_tokens[0, -1, :], temperature=temp)
            else:
                sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char
            # Exit condition: either hit max length or find stop character.
            if (sampled_char == self.end_char or len(decoded_sentence) > self.max_decoder_seq_length):
                stop_condition = True
            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1
            # Update states
            states_value = [h, c]
        decoded_sentence = decoded_sentence.replace('nigga', 'n****')
        return decoded_sentence

    def test_run(self, chat=False):
        # input_sentence_1 = "bloodsuckin' succubuses, what the fuck is up with this?"
        input_sentence_2 = "she callin', she textin', she’s fallin', but let me explain"
        # input_sentence_3 = "i'm tryna make the goosebumps on your inner thigh show"

        # Testing temperatures
        # print(f"\nInput Sentence #1: {input_sentence_1}")
        # print(f"Reply #1 (No Diversity): {self.reply(input_sentence_1)}")
        # print(f"\nInput Sentence #1: {input_sentence_1}")
        # print(f"Reply #2 (Temp=0.4): {self.reply(input_sentence_1, diversity=True, temp=0.55)}")
        # print(f"\nInput Sentence #1: {input_sentence_1}")
        # print(f"Reply #3 (Temp=0.55): {self.reply(input_sentence_1, diversity=True, temp=0.60)}")
        # print(f"\nInput Sentence #1: {input_sentence_1}")
        # print(f"Reply #4 (Temp=0.65): {self.reply(input_sentence_1, diversity=True, temp=0.65)}")
        # print(f"\nInput Sentence #1: {input_sentence_1}")
        # print(f"Reply #5 (Temp=0.75): {self.reply(input_sentence_1, diversity=True, temp=0.70)}")

        print(f"\n\nInput Sentence #2: {input_sentence_2}")
        print(f"Reply #6: {self.reply(input_sentence_2)}")
        # print(f"\nInput Sentence #3: {input_sentence_3}")
        # print(f"Reply #7: {self.reply(input_sentence_3)}")

        if chat:
            self._chat_over_command_line()

def main():
    filepath = '../../cap3_models/models/'
    name = 'baseline_rms_800ep_512ld'
    weights = '-weights_final'
    model = AnnotationGenerator(use_weights=True, models_filepath=filepath, data_filepath=filepath,
                                trained_model=name, data_name=name, final_weights_fp=weights,
                                latent_dim=512)

    model.test_run(chat=True)

if __name__ == "__main__":
    main()
