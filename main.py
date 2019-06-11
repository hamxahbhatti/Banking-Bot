# %% importing the libraries
from model import load_model
from voice_recognition import voice_input
from text_preprocessing import TokenizerWrap
from inference_model import translate
from flask import Flask,render_template,jsonify,request
from voice_output import speak
# %% loading the models
# app = Flask(__name__)
# encoder =  None
# decoder = None

# @app.route('/')
# def home():
#     return render_template('index.html')
    
# def load_models():
#     global encoder
#     global decoder
#     encoder = load_model('Model Weights/GRU/encoder_model.json', 'Model Weights/GRU/encoder_model_weights.h5')
#     decoder = load_model('Model Weights/GRU/decoder_model.json', 'Model Weights/GRU/decoder_model_weights.h5')
  

# def predict(query):
#     # encoder = load_model('Model Weights/GRU/encoder_model.json', 'Model Weights/GRU/encoder_model_weights.h5')
#     # decoder = load_model('Model Weights/GRU/decoder_model.json', 'Model Weights/GRU/decoder_model_weights.h5')
#     prediction=translate(encoder,decoder,query)
#     return prediction

# #%%
# @app.route('/receive_query',methods=['POST'])
# def receive_query():
#     request_data = request.form
#     print(request_data)
#     query = request_data['question']
#     answer = predict(query)
#     return jsonify({'Answer':answer})

# @app.route('/voice_query',methods=["POST"])
# def voice_query():
#     pass

# if __name__ == "__main__":
#     load_models()
#     app.run(port=5000)

#
#%% 
encoder = load_model('Model Weights/GRU/encoder_model.json', 'Model Weights/GRU/encoder_model_weights.h5')
decoder = load_model('Model Weights/GRU/decoder_model.json', 'Model Weights/GRU/decoder_model_weights.h5')
#%%
# query = input("Enter the query : ")
# translate(encoder,decoder,query)
# speak(Answer)
# #%%
# query  = 'How can I apply for the Sweep-in facility '
# translate(encoder,decoder,query)

#%%
# query = 'Can I take any loan'
# query = input( " Enter a query : ")
# translate(encoder,decoder,query)

#%%
# query = 'What is a retirement plan'
# translate(encoder,decoder,query)

# #%%
# query = 'Can I Login to Check My Account'
# translate(encoder,decoder,query)

# #%%
# query='Can I sell my car before I repay my loan'
# translate(encoder,decoder,query)
# #%%
# query='How do I obtain a NOC from the bank if I have lost my Registration Certificate book for a vehicle under finance with HDFC Bank'
# translate(encoder,decoder,query)

# #%%
query='How should the gold loan be repaid'
translate(encoder,decoder,query)
# #%%
# query = 'How can I repay my Personal Loan'
# translate(encoder,decoder,query)


# #%%
# query= ' What is EMI '
# translate(encoder,decoder,query)



    

#%%	
