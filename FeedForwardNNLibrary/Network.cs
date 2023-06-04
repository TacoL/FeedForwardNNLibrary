using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FeedForwardNNLibrary
{
    internal class Network
    {
        internal static Random r = new Random();
        public readonly int _numInputs, _numOutputs;

        private readonly double _learningRate;
        private readonly double _momentumScalar;
        private readonly int _batchSize;

        private List<List<Neuron>> layers = new List<List<Neuron>>();

        public Network(int numInputs, int numOutputs, double learningRate, double momentumScalar, int batchSize)
        {
            _numInputs = numInputs;
            _numOutputs = numOutputs;
            _learningRate = learningRate;
            _momentumScalar = momentumScalar;
            _batchSize = batchSize;

            // Add input layer
            List<Neuron> layer = new List<Neuron>();
            for (int neuronIdx = 0; neuronIdx < numInputs; neuronIdx++)
                layer.Add(new Neuron(0, ActivationFunctions.None, _learningRate, _momentumScalar, _batchSize));
            layers.Add(layer);
        }

        internal Network(Network original)
        {
            for (int layerIdx = 0; layerIdx < original.layers.Count; layerIdx++)
            {
                List<Neuron> layer = new List<Neuron>();
                for (int neuronIdx = 0; neuronIdx < original.layers[layerIdx].Count; neuronIdx++)
                {
                    layer.Add(new Neuron(original.layers[layerIdx][neuronIdx]));
                }
                layers.Add(layer);
            }
        }

        public void AddLayer(int numNeurons, ActivationFunctions activation)
        {
            List<Neuron> layer = new List<Neuron>();
            for (int neuronIdx = 0; neuronIdx < numNeurons; neuronIdx++)
                layer.Add(new Neuron(layers[layers.Count - 1].Count, activation, _learningRate, _momentumScalar, _batchSize));
            layers.Add(layer);
        }

        internal double[] forwardPropagateBeforeSoftmax(double[] inputs)
        {
            //First Layer Setup
            for (int i = 0; i < inputs.Length; i++)
                layers[0][i].firstLayerSetup(inputs[i]);
            //layers[0].Select((n, i) => (n, i)).ToList().ForEach(t => t.n.firstLayerSetup(inputs[t.i]));

            //Propagate Forward
            for (int layerIdx = 1; layerIdx < layers.Count; layerIdx++)
                layers[layerIdx].ForEach(neuron => neuron.calcActivationValue(layers[layerIdx - 1]));

            //Convert from Neuron to Double
            return convertLayerToDoubles(layers[layers.Count - 1]);
        }
        internal double[] forwardPropagate(double[] inputs)
        {
            double[] outputsBeforeSoftmax = forwardPropagateBeforeSoftmax(inputs);

            //Apply Softmax
            return applySoftmax(outputsBeforeSoftmax);
        }

        internal double backPropagate(double[] inputs, double[] targets)
        {
            double[] outputsBeforeSoftmax = forwardPropagateBeforeSoftmax(inputs);
            double[] outputs = applySoftmax(outputsBeforeSoftmax); //after Softmax
            double[] cost = new double[targets.Length];
            double[] costGradient = new double[targets.Length]; //technically the activation gradient for the outer layer

            int idx = 0;
            outputs.ToList().ForEach(o =>
            {
                cost[idx] = Math.Pow(o - targets[idx], 2);
                costGradient[idx] = 2 * (o - targets[idx]);
                idx++;
            });

            for (int layerIdx = layers.Count - 1; layerIdx > 0; layerIdx--) //don't count the first layer, since that's just input
            {
                List<Neuron> layer = layers[layerIdx];
                List<Neuron> previousLayer = layers[layerIdx - 1];
                for (int neuronIdx = 0; neuronIdx < layers[layerIdx].Count; neuronIdx++)
                {
                    Neuron neuron = layer[neuronIdx];
                    if (layerIdx == layers.Count - 1)
                        neuron.neuronGradient = costGradient[neuronIdx] * softmaxDerivative(outputsBeforeSoftmax)[neuronIdx]; //reset neuron gradient for each sample
                    else
                    {
                        //calculate activation gradient
                        double activationGradient = 0;
                        layers[layerIdx + 1].ForEach(frontNeuron => activationGradient += frontNeuron.neuronGradient * frontNeuron.weights[neuronIdx]);

                        neuron.neuronGradient = activationGradient * neuron.derivativeActivation(neuron.neuronValue);
                    }

                    neuron.biasGradient += neuron.neuronGradient;
                    for (int weightIdx = 0; weightIdx < neuron.weights.Length; weightIdx++)
                        neuron.weightGradients[weightIdx] += neuron.neuronGradient * previousLayer[weightIdx].activationValue;
                }
            }

            //calculate mse for this sample
            double mse = 0;
            cost.ToList().ForEach(c => mse += c);
            return mse / cost.Length;
        }

        internal double[] convertLayerToDoubles(List<Neuron> layer)
        {
            double[] newArray = new double[layer.Count];
            int neuronIdx = 0;
            layer.ForEach(neuron =>
            {
                newArray[neuronIdx] = neuron.activationValue;
                neuronIdx++;
            });

            return newArray;
        }

        internal double[] applySoftmax(double[] array)
        {
            double expSum = 0;
            for (int i = 0; i < array.Length; i++)
                expSum += Math.Exp(array[i]);

            double[] newArray = new double[array.Length];
            for (int i = 0; i < newArray.Length; i++)
                newArray[i] = Math.Exp(array[i]) / expSum;

            return newArray;
        }

        internal double[] softmaxDerivative(double[] array)
        {
            double[] softmaxArray = applySoftmax(array);
            double[] newArray = new double[array.Length];
            for (int i = 0; i < newArray.Length; i++)
                newArray[i] = softmaxArray[i] * (1 - softmaxArray[i]);
            return newArray;
        }

        internal void updateWeightsAndBiases()
        {
            layers.ForEach(layer => layer.ForEach(neuron => neuron.updateWeightsAndBias()));
        }
    }
}
