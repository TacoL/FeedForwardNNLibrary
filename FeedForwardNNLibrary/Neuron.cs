using System;
using System.Collections.Generic;

namespace FeedForwardNNLibrary
{
    internal class Neuron
    {
        #region Properties
        internal double NeuronValue { get; set; }
        internal double NeuronGradient { get; set; }
        internal double[] Weights { get; set; }
        internal double[] WeightGradients { get; set; }
        internal double Bias { get; set; }
        internal double BiasGradient { get; set; }
        internal double ActivationValue { get; set; }
        internal ActivationFunctions ActivationFunction { get; set; }
        #endregion

        private double _previousBiasGradient;
        private double[] _previousWeightGradient;

        private readonly double _learningRate, _momentumScalar;
        private readonly int _batchSize;

        internal Neuron(int numInputs, ActivationFunctions activation, double learningRate, double momentumScalar, int batchSize)
        {
            ActivationFunction = activation;
            _learningRate = learningRate;
            _momentumScalar = momentumScalar;
            _batchSize = batchSize;

            NeuronValue = 0;
            ActivationValue = 0;
            NeuronGradient = 0;

            Weights = new double[numInputs];
            WeightGradients = new double[numInputs];
            _previousWeightGradient = new double[numInputs];

            double y = 1 / Math.Sqrt(numInputs);
            for (int weightIdx = 0; weightIdx < Weights.Length; weightIdx++)
                Weights[weightIdx] = Network.r.NextDouble() * y * (Network.r.NextDouble() > .5 ? 1 : -1);

            Bias = 0;
            BiasGradient = 0;
            _previousBiasGradient = 0;
        }

        internal Neuron(Neuron original)
        {
            ActivationFunction = original.ActivationFunction;
            _learningRate = original._learningRate;
            _momentumScalar = original._momentumScalar;
            _batchSize = original._batchSize;

            NeuronValue = 0;
            ActivationValue = 0;
            NeuronGradient = 0;

            Weights = new double[original.Weights.Length];
            WeightGradients = new double[original.WeightGradients.Length];
            _previousWeightGradient = new double[original._previousWeightGradient.Length];

            for (int weightIdx = 0; weightIdx < Weights.Length; weightIdx++)
                Weights[weightIdx] = original.Weights[weightIdx];

            Bias = original.Bias;
            BiasGradient = 0;
            _previousBiasGradient = 0;
        }

        internal void CalcActivationValue(List<Neuron> inputNeurons)
        {
            NeuronValue = 0; //reset neuron value every time

            for (int i = 0; i < inputNeurons.Count; i++)
                NeuronValue += inputNeurons[i].ActivationValue * Weights[i];
            NeuronValue += Bias;

            ActivationValue = CalcActivationFunction(NeuronValue);
        }

        internal void CalcSoftmaxActivation(double[] currentLayerNeuronValues)
        {
            double expSum = 0;
            for (int i = 0; i < currentLayerNeuronValues.Length; i++)
                expSum += Math.Exp(currentLayerNeuronValues[i]);

            ActivationValue = Math.Exp(ActivationValue) / expSum;
        }

        private double CalcActivationFunction(double x)
        {
            if (ActivationFunction.Activation == "Tanh")
                return Math.Tanh(x);
            else if (ActivationFunction.Activation == "ReLu")
                return x <= 0 ? 0 : x;
            else if (ActivationFunction.Activation == "None")
                return x;
            else if (ActivationFunction.Activation == "Softmax")
                return x;
            else
                return 0;
        }

        internal double DerivativeActivation(double x)
        {
            if (ActivationFunction.Activation == "Tanh")
                return 1 - Math.Pow(Math.Tanh(x), 2);
            else if (ActivationFunction.Activation == "ReLu")
                return x <= 0 ? 0 : 1;
            else if (ActivationFunction.Activation == "None")
                return 0;
            else if (ActivationFunction.Activation == "Softmax")
                return ActivationValue * (1 - ActivationValue);
            else
                return 0;
        }

        internal void UpdateWeightsAndBias()
        {
            //update weights and biases
            for (int weightIdx = 0; weightIdx < Weights.Length; weightIdx++)
            {
                Weights[weightIdx] -= (WeightGradients[weightIdx] / _batchSize * _learningRate) + (_previousWeightGradient[weightIdx] * _momentumScalar);
                _previousWeightGradient[weightIdx] = WeightGradients[weightIdx] / _batchSize;
            }
                
            Bias -= (BiasGradient / _batchSize * _learningRate) + (_previousBiasGradient * _momentumScalar);
            _previousBiasGradient = BiasGradient / _batchSize;

            //reset neuron and weight and bias GRADIENTS after each update (update every batch)
            NeuronGradient = 0;
            for (int wgIdx = 0; wgIdx < WeightGradients.Length; wgIdx++)
                WeightGradients[wgIdx] = 0;
            BiasGradient = 0;
        }
    }
}