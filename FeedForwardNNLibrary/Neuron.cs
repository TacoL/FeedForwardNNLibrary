using System;
using System.Collections.Generic;

namespace FeedForwardNNLibrary
{
    internal class Neuron
    {
        internal double neuronValue, activationValue, bias, neuronGradient, biasGradient, previousBG;
        internal double[] weights, weightGradients, previousWG;
        internal ActivationFunctions _activationFunction;
        private double _learningRate, _momentumScalar;
        private int _batchSize;

        internal Neuron(int numInputs, ActivationFunctions activation, double learningRate, double momentumScalar, int batchSize)
        {
            _activationFunction = activation;
            _learningRate = learningRate;
            _momentumScalar = momentumScalar;
            _batchSize = batchSize;

            neuronValue = 0;
            activationValue = 0;
            neuronGradient = 0;

            weights = new double[numInputs];
            weightGradients = new double[numInputs];
            previousWG = new double[numInputs];

            double y = 1 / Math.Sqrt(numInputs);
            for (int weightIdx = 0; weightIdx < weights.Length; weightIdx++)
                weights[weightIdx] = Network.r.NextDouble() * y * (Network.r.NextDouble() > .5 ? 1 : -1);

            bias = 0;
            biasGradient = 0;
            previousBG = 0;
        }

        internal Neuron(Neuron original)
        {
            _activationFunction = original._activationFunction;
            _learningRate = original._learningRate;
            _momentumScalar = original._momentumScalar;
            _batchSize = original._batchSize;

            neuronValue = 0;
            activationValue = 0;
            neuronGradient = 0;

            weights = new double[original.weights.Length];
            weightGradients = new double[original.weightGradients.Length];
            previousWG = new double[original.previousWG.Length];

            for (int weightIdx = 0; weightIdx < weights.Length; weightIdx++)
                weights[weightIdx] = original.weights[weightIdx];

            bias = original.bias;
            biasGradient = 0;
            previousBG = 0;
        }

        internal void calcActivationValue(List<Neuron> inputNeurons)
        {
            neuronValue = 0; //reset neuron value every time

            for (int i = 0; i < inputNeurons.Count; i++)
                neuronValue += inputNeurons[i].activationValue * weights[i];
            neuronValue += bias;

            activationValue = activationFunction(neuronValue);
        }

        internal void calcSoftmaxActivation(double[] currentLayerNeuronValues)
        {
            double expSum = 0;
            for (int i = 0; i < currentLayerNeuronValues.Length; i++)
                expSum += Math.Exp(currentLayerNeuronValues[i]);

            activationValue = Math.Exp(activationValue) / expSum;
        }

        internal double activationFunction(double x)
        {
            if (_activationFunction.activation == "Tanh")
                return Math.Tanh(x);
            else if (_activationFunction.activation == "ReLu")
                return x <= 0 ? 0 : x;
            else if (_activationFunction.activation == "None")
                return x;
            else if (_activationFunction.activation == "Softmax")
                return x;
            else
                return 0;
        }

        internal double derivativeActivation(double x)
        {
            if (_activationFunction.activation == "Tanh")
                return 1 - Math.Pow(Math.Tanh(x), 2);
            else if (_activationFunction.activation == "ReLu")
                return x <= 0 ? 0 : 1;
            else if (_activationFunction.activation == "None")
                return 0;
            else if (_activationFunction.activation == "Softmax")
                return activationValue * (1 - activationValue);
            else
                return 0;
        }

        internal void firstLayerSetup(double actV)
        {
            activationValue = actV;
        }

        internal void updateWeightsAndBias()
        {
            //update weights and biases
            for (int weightIdx = 0; weightIdx < weights.Length; weightIdx++)
            {
                weights[weightIdx] -= (weightGradients[weightIdx] / _batchSize * _learningRate) + (previousWG[weightIdx] * _momentumScalar);
                previousWG[weightIdx] = weightGradients[weightIdx] / _batchSize;
            }
                
            bias -= (biasGradient / _batchSize * _learningRate) + (previousBG * _momentumScalar);
            previousBG = biasGradient / _batchSize;

            //reset neuron and weight and bias GRADIENTS after each update (update every batch)
            neuronGradient = 0;
            for (int wgIdx = 0; wgIdx < weightGradients.Length; wgIdx++)
                weightGradients[wgIdx] = 0;
            biasGradient = 0;
        }
    }
}