## Haiku Generating with Long Short Term Memory Networks

A little project I did while studying about LSTM networks. The repository contains an LSTM network implemented using keras which aims to generate new haikus (traditional 3 lined Japanese poems). The training data set is stored in haikuzou.txt which is a corpus with over 8000 haikus obtained from this repository: https://github.com/sballas8/PoetRNN/blob/master/data/haikus.csv. 

## A bit about LSTM networks 
LSTM networks are a special kind of recurrent neural network (RNN) which is capable of retaining long-term memories. They attempt to solve the vanishing gradient problem which is a problem common in RNNs (and also other deep learning algorithms) which results in RNNs being only capable of retaining short-term memories. All RNNs have the form of a chain of repeating modules where each module has a single neural network layer. A repeating module in a standard RNN is shown in the image below.  

<p align="center">
  <img src="https://user-images.githubusercontent.com/35329219/57978201-a754d000-7a4b-11e9-99df-3b9d203c548b.JPG">
</p>

Whilst LSTMs also form a chain of repeating modules, unlike RNNs, each module contains four network layers interacting in a special way:

<p align="center">
  <img src="https://user-images.githubusercontent.com/35329219/57978207-c05d8100-7a4b-11e9-9261-19c4f60ae6e7.JPG">
</p>

Each LSTM unit contains 4 important components - the cell state, forget gate, input gate and output gate. 
- Cell State: serves as the long-term memory of the network. Unlike hidden states which stores information of the overall state from what the network has so far, cell states only store SELECTIVE memory of the past. 
- Forget Gate: takes the previous hidden state and decides what must be removed from the previous timestep. This gate aims to remove any irrelevant information. 
- Input Gate: this gate processes the new input from the current timestep and decides which parts of this information is actually worth saving.
- Output Gate: this gate basically decides what the next hidden state should be. It's main function is to learn to only focus the model's long-term memory into information that will be immediately useful. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/35329219/57978215-e5ea8a80-7a4b-11e9-841d-a73b84ab33e6.JPG">
</p>
