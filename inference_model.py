import numpy as np
import pickle


with open('Model Misc/tokenizer_src.pickle','rb') as file:
    tokenizer_src = pickle.load(file)
with open('Model Misc/tokenizer_dest.pickle','rb') as file:
    tokenizer_dest=  pickle.load(file)


tokenizer_src.index_to_word
mark_start='ssss'
mark_end = ' eeee'

def translate(encoder,decoder,input_text, true_output_text=None):
    """Translate a single text-string."""

    # Convert the input-text to integer-tokens.
    # Note the sequence of tokens has to be reversed.
    # Padding is probably not necessary.
    input_tokens = tokenizer_src.text_to_tokens(text=input_text,
                                                reverse=True,
                                                padding=True)
    print(input_tokens)
    
    # Get the output of the encoder's GRU which will be
    # used as the initial state in the decoder's GRU.
    # This could also have been the encoder's final state
    # but that is really only necessary if the encoder
    # and decoder use the LSTM instead of GRU because
    # the LSTM has two internal states.
    initial_state = encoder.predict(input_tokens)
    # print(f'inital_state : {initial_state}')
    token_start = tokenizer_dest.word_index[mark_start.strip()]
    token_end = tokenizer_dest.word_index[mark_end.strip()]

    # Max number of tokens / words in the output sequence.
    max_tokens = tokenizer_dest.max_tokens

    # Pre-allocate the 2-dim array used as input to the decoder.
    # This holds just a single sequence of integer-tokens,
    # but the decoder-model expects a batch of sequences.
    shape = (1, max_tokens)
    decoder_input_data = np.zeros(shape=shape, dtype=np.int)

    # The first input-token is the special start-token for 'ssss '.
    token_int = token_start

    # Initialize an empty output-text.
    output_text = ''

    # Initialize the number of tokens we have processed.
    count_tokens = 0

    # While we haven't sampled the special end-token for ' eeee'
    # and we haven't processed the max number of tokens.
    while token_int != token_end and count_tokens < max_tokens:
        # Update the input-sequence to the decoder
        # with the last token that was sampled.
        # In the first iteration this will set the
        # first element to the start-token.
        decoder_input_data[0, count_tokens] = token_int

        # Wrap the input-data in a dict for clarity and safety,
        # so we are sure we input the data in the right order.
        x_data = \
        {
            'decoder_initial_state': initial_state,
            'decoder_input': decoder_input_data
        }

        # Note that we input the entire sequence of tokens
        # to the decoder. This wastes a lot of computation
        # because we are only interested in the last input
        # and output. We could modify the code to return
        # the GRU-states when calling predict() and then
        # feeding these GRU-states as well the next time
        # we call predict(), but it would make the code
        # much more complicated.

        # Input this data to the decoder and get the predicted output.
        decoder_output = decoder.predict(x_data)
#         print(decoder_output)
        # Get the last predicted token as a one-hot encoded array.
        token_onehot = decoder_output[0, count_tokens, :]
        #print(token_onehot)
        
        # Convert to an integer-token.
        token_int = np.argmax(token_onehot)

        # Lookup the word corresponding to this integer-token.
        sampled_word = tokenizer_dest.token_to_word(token_int)

        # Append the word to the output-text.
        output_text += " " + sampled_word

        # Increment the token-counter.
        count_tokens += 1

    # Sequence of tokens output by the decoder.
    #output_tokens = decoder_input_data[0]
    
    # Print the input-text.
    print("Input text:")
    print(input_text)
    print()

    # Print the translated output-text.
    print("Translated text:")
    print(output_text)
    print()

    # Optionally print the true translated text.
    if true_output_text is not None:
        print("True output text:")
        print(true_output_text)
        print()