# %% importing the libraries
from model import load_model
from voice_recognition import voice_input
from text_preprocessing import TokenizerWrap
from inference_model import translate
# %% loading the models

encoder = load_model('Model Weights/GRU/encoder_model.json', 'Model Weights/GRU/encoder_model_weights.h5')
decoder = load_model('Model Weights/GRU/decoder_model.json', 'Model Weights/GRU/decoder_model_weights.h5')


#%%
query  = 'Can an individual open a Current Account '
translate(encoder,decoder,query)

#%%
query  = 'How can I apply for the Sweep-in facility '
translate(encoder,decoder,query)

#%%
query = 'Can I take any loan'
translate(encoder,decoder,query)

#%%
query = 'What is a retirement plan'
translate(encoder,decoder,query)

#%%
query = 'Can I Login to Check My Account'
translate(encoder,decoder,query)

#%%
query='Can I sell my car before I repay my loan'
translate(encoder,decoder,query)
#%%
query='How do I obtain a NOC from the bank if I have lost my Registration Certificate book for a vehicle under finance with HDFC Bank'
translate(encoder,decoder,query)

#%%
query='How should the gold loan be repaid'
translate(encoder,decoder,query)
#%%
query = 'How can I repay my Personal Loan'
translate(encoder,decoder,query)


#%%
query= ' What is EMI '
translate(encoder,decoder,query)

#%%
