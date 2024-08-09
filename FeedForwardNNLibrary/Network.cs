using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Xml;

namespace FeedForwardNNLibrary
{
    public class Network
    {
        internal readonly static Random r = new Random();

        private readonly int _numInputs;
        private readonly double _learningRate;
        private readonly double _momentumScalar;
        private readonly int _batchSize;
        private readonly List<List<Neuron>> layers = new List<List<Neuron>>();

        public Network(int numInputs, double learningRate, double momentumScalar, int batchSize)
        {
            _numInputs = numInputs;
            _learningRate = learningRate;
            _momentumScalar = momentumScalar;
            _batchSize = batchSize;

            // Add input layer
            List<Neuron> layer = new List<Neuron>();
            for (int neuronIdx = 0; neuronIdx < numInputs; neuronIdx++)
                layer.Add(new Neuron(0, ActivationFunctions.None, _learningRate, _momentumScalar, _batchSize));
            layers.Add(layer);
        }

        private Network(Network original)
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

        public double[] ForwardPropagate(double[] inputs)
        {
            //First Layer Setup
            for (int i = 0; i < inputs.Length; i++)
                layers[0][i].ActivationValue = inputs[i];

            //Propagate Forward
            for (int layerIdx = 1; layerIdx < layers.Count; layerIdx++)
            {
                layers[layerIdx].ForEach(neuron => neuron.CalcActivationValue(layers[layerIdx - 1]));

                layers[layerIdx].ForEach(neuron =>
                {
                    if (neuron.ActivationFunction.Activation == "Softmax")
                        neuron.CalcSoftmaxActivation(ConvertLayerToDoubles(layers[layerIdx]));
                });
            }

            //Convert from Neuron to Double
            return ConvertLayerToDoubles(layers[layers.Count - 1]);
        }

        private double BackPropagate(double[] inputs, double[] targets)
        {
            double[] outputs = ForwardPropagate(inputs);
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
                        neuron.NeuronGradient = costGradient[neuronIdx] * neuron.DerivativeActivation(neuron.NeuronValue); //reset neuron gradient for each sample
                    else
                    {
                        //calculate activation gradient
                        double activationGradient = 0;
                        layers[layerIdx + 1].ForEach(frontNeuron => activationGradient += frontNeuron.NeuronGradient * frontNeuron.Weights[neuronIdx]);

                        neuron.NeuronGradient = activationGradient * neuron.DerivativeActivation(neuron.NeuronValue);
                    }

                    neuron.BiasGradient += neuron.NeuronGradient;
                    for (int weightIdx = 0; weightIdx < neuron.Weights.Length; weightIdx++)
                        neuron.WeightGradients[weightIdx] += neuron.NeuronGradient * previousLayer[weightIdx].ActivationValue;
                }
            }

            //calculate mse for this sample
            double mse = 0;
            cost.ToList().ForEach(c => mse += c);
            return mse / cost.Length;
        }

        private double[] ConvertLayerToDoubles(List<Neuron> layer)
        {
            double[] newArray = new double[layer.Count];
            int neuronIdx = 0;
            layer.ForEach(neuron =>
            {
                newArray[neuronIdx] = neuron.ActivationValue;
                neuronIdx++;
            });

            return newArray;
        }

        #region Training
        public async Task Train(List<TrainingSample> trainingSamples, int numEpochs)
        {
            if (trainingSamples.Count == 0) { throw new Exception("You must have training samples!"); }

            for (int epoch = 0; epoch < numEpochs; epoch++)
            {
                double mse = 0;
                int numBatches = trainingSamples.Count / _batchSize;

                //batching
                for (int batchIdx = 0; batchIdx < numBatches; batchIdx++) //for each batch
                {
                    double batchMse = await TrainBatch(batchIdx, trainingSamples);
                    mse += batchMse / _batchSize;

                    // Update weights and biases
                    layers.ForEach(layer => layer.ForEach(neuron => neuron.UpdateWeightsAndBias()));

                    //Console.WriteLine($"Epoch {epoch + 1} / {numEpochs}      Batch #{batchIdx + 1} / {numBatches}      BMSE = {batchMse / Network.batchSize}");
                }

                Console.WriteLine("Epoch: {0}         MSE: {1}", epoch + 1, mse / numBatches);
            }
        }

        private async Task<double> TrainBatch(int batchIdx, List<TrainingSample> trainingSamples)
        {
            double batchMse = 0;
            List<Task> tasks = new List<Task>();

            for (int sampleIdx = 0; sampleIdx < _batchSize; sampleIdx++)
            {
                int thisSampleIdx = sampleIdx; // makes it work with Tasks
                tasks.Add(Task.Run(() => batchMse += TrainSample(batchIdx, thisSampleIdx, trainingSamples)));
            }

            await Task.WhenAll(tasks);
            return batchMse;
        }

        private double TrainSample(int batchIdx, int sampleIdx, List<TrainingSample> trainingSamples)
        {
            int sampleIdxToTest = batchIdx * _batchSize + sampleIdx;
            if (trainingSamples[sampleIdxToTest].targets.Length != layers[layers.Count - 1].Count)
            {
                throw new Exception("Your final layer must match number of targets!");
            }

            Network sampleNN = new Network(this);
            double sampleMse = sampleNN.BackPropagate(trainingSamples[sampleIdxToTest].inputs, trainingSamples[sampleIdxToTest].targets);

            AddToGradients(sampleNN);
            return sampleMse;
        }

        private void AddToGradients(Network sampleNN)
        {
            for (int layerIdx = 0; layerIdx < this.layers.Count; layerIdx++)
            {
                for (int neuronIdx = 0; neuronIdx < this.layers[layerIdx].Count; neuronIdx++)
                {
                    Neuron mainNeuron = this.layers[layerIdx][neuronIdx];
                    Neuron sampleNeuron = sampleNN.layers[layerIdx][neuronIdx];

                    for (int i = 0; i < mainNeuron.WeightGradients.Length; i++)
                    {
                        mainNeuron.WeightGradients[i] += sampleNeuron.WeightGradients[i];
                    }

                    mainNeuron.BiasGradient += sampleNeuron.BiasGradient;
                }
            }
        }
        #endregion

        #region Export
        /// <summary>
        /// Exports the neural network as an xml file
        /// </summary>
        /// <param name="filePath">File path to export to</param>
        public void ExportModel(string filePath)
        {
            XmlWriterSettings settings = new XmlWriterSettings();
            settings.Indent = true;
            settings.IndentChars = "\t";

            using XmlWriter writer = XmlWriter.Create(filePath, settings);
            writer.WriteStartElement("network");

            writer.WriteElementString("numInputs", _numInputs.ToString());
            writer.WriteElementString("learningRate", _learningRate.ToString());
            writer.WriteElementString("momentumScalar", _momentumScalar.ToString());
            writer.WriteElementString("batchSize", _batchSize.ToString());

            writer.WriteStartElement("layers");
            for (int layerIdx = 1; layerIdx < layers.Count; layerIdx++)
            {
                writer.WriteStartElement("layer" + layerIdx.ToString());
                writer.WriteElementString("NumNeurons", layers[layerIdx].Count.ToString());
                writer.WriteElementString("ActivationFunction", layers[layerIdx][0].ActivationFunction.Activation);
                writer.WriteStartElement("neurons");
                for (int neuronIdx = 0; neuronIdx < layers[layerIdx].Count; neuronIdx++)
                {
                    writer.WriteStartElement("neuron" + neuronIdx.ToString());

                    Neuron neuron = layers[layerIdx][neuronIdx];
                    writer.WriteStartElement("weights");
                    for (int weightIdx = 0; weightIdx < neuron.Weights.Length; weightIdx++)
                    {
                        writer.WriteElementString("weight" + weightIdx.ToString(), neuron.Weights[weightIdx].ToString());
                    }
                    writer.WriteEndElement(); // end weights
                    writer.WriteElementString("bias", neuron.Bias.ToString());
                    writer.WriteEndElement(); // end neuron[idx]
                }
                writer.WriteEndElement(); // end neurons
                writer.WriteEndElement(); // end layer[idx]
            }

            writer.WriteEndElement(); // end layers
            writer.WriteEndElement(); // end network
            writer.Close();
        }

        /// <summary>
        /// Imports a model from a file path.
        /// </summary>
        /// <param name="filePath"></param>
        /// <returns></returns>
        public static Network ImportModel(string filePath)
        {
            using XmlReader reader = XmlReader.Create(filePath);

            int numInputs = 0;
            double learningRate = 0;
            double momentumScalar = 0;
            int numNeurons = 0;
            Network network = new Network(0, 0, 0, 0);
            Neuron currentNeuron = new Neuron(0, ActivationFunctions.None, 0, 0, 0);

            while (reader.Read())
            {
                if (reader.NodeType == XmlNodeType.Element)
                {
                    switch (reader.Name)
                    {
                        case "numInputs": numInputs = reader.ReadElementContentAsInt(); break;
                        case "learningRate": learningRate = reader.ReadElementContentAsDouble(); break;
                        case "momentumScalar": momentumScalar = reader.ReadElementContentAsDouble(); break;
                        case "batchSize":
                            int batchSize = reader.ReadElementContentAsInt();
                            network = new Network(numInputs, learningRate, momentumScalar, batchSize);
                            break;
                        case "NumNeurons": numNeurons = reader.ReadElementContentAsInt(); break;
                        case "ActivationFunction": network.AddLayer(numNeurons, ActivationFunctions.ConvertFromString(reader.ReadElementContentAsString())); break;
                        case "bias": currentNeuron.Bias = reader.ReadElementContentAsDouble(); break;
                        default: break;
                    }

                    if (reader.Name.Contains("neuron") && reader.Name != "neurons")
                    {
                        int neuronIdx = int.Parse(reader.Name.Substring(6, 1));
                        currentNeuron = network.layers[network.layers.Count - 1][neuronIdx];
                    }
                    else if (reader.Name.Contains("weight") && reader.Name != "weights")
                    {
                        int weightIdx = int.Parse(reader.Name.Substring(6, 1));
                        currentNeuron.Weights[weightIdx] = reader.ReadElementContentAsDouble();
                    }
                }
            }

            if (network == null) { throw new Exception("Network doesn't exist"); }
            return network;
        }
        #endregion
    }
}
