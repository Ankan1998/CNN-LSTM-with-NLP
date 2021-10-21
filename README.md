# <p align="center"> Hate Speech Detector with Experimental CNN+LSTM</p>
This repo is created to experiment with power of CNN on NLP tasks.
The model architecture has a embedding layer(token and positional) then it is passed to 
CNN (1D) to get contextual meaning taking few words at one go and then again moving into next part.
then these context vector is combined with another vector which is formed by context vector + embedding vector. Then 
these are concatenated and passed through LSTM and last layer output of hidden state is passed through sigmoid
to get the classification.<br>
This repo is experimental so model may be updated with some new addition
### TODO
* [x] Data Preparation
  * [x] Text Preprocessing
  * [x] Data pipeline
* [x] Model Design
  * [x] Encoder Head
  * [x] Classifier Head
* [x] Training
* [x] Saving & Loading
  * [x] Saving
  * [x] Loading
* [x] Inference
