using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FeedForwardNNLibrary
{
    internal class Neuron
    {
        internal double neuronValue, activationValue, bias, neuronGradient, biasGradient, previousBG;
        internal double[] weights, weightGradients, previousWG;
        private ActivationFunctions _activationFunction;
        private double _learningRate, _momentumScalar;
        private int _batchSize;

        internal Neuron(int numInputs, ActivationFunctions activation, double learningRate, double momentumScalar, int batchSize)
        {
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


            activationValue = output ? outputActivation(neuronValue) : activationFunction(neuronValue);
        }

        internal double activationFunction(double x)
        {
            return Math.Tanh(x);

            //return Math.Tanh(x/125);

            //return x <= 0 ? 0 : activationValue;
        }

        internal double derivativeActivation(double x)
        {
            return 1 - Math.Pow(Math.Tanh(x), 2);

            //double secant = 2.0 / (Math.Exp(x/125) + Math.Exp(-x/125));
            //return Math.Pow(secant, 2) / 125.0;

            //return x <= 0 ? 0 : 1;
        }

        internal double outputActivation(double x)
        {
            //softmax
            return x;
        }

        internal double derivativeOutputActivation(double x)
        {
            //for softmax, not used for an individual neuron
            return x;
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