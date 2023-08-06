# FeedForwardNNLibrary
## Description
The folder "FeedForwardNNLibrary" is the actual library, and the other three (not Loaded) folders are examples of implementation (which reference the library as a project reference). The Iris Flower (Loaded) folder is an example of importing a previously exported model.
## Features
### Constructor & Methods
- Creation of Network (parameters: number of inputs, learning rate, momentum scalar, batch size)
- Add layer (parameters: number of neurons in layer, activation function)
- Train (parameters: list of training samples, number of epochs)
- Forward Propagate - for seeing results for feeding a sample the network hasn't trained on (parameters: inputs)
### Activation Functions
- Tanh
- ReLu
- Softmax
- None
### Export/Import Model
- Export model to a .xml file path
- Import model from a .xml file path
