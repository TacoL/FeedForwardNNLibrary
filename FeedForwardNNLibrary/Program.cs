using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace FeedForwardNNLibrary
{
    static class Program
    {
        /// <summary>
        /// The main entry point for the application.
        /// </summary>
        [STAThread]
        public static async Task Main()
        {
            //Application.EnableVisualStyles();
            //Application.SetCompatibleTextRenderingDefault(false);
            
            //set up the network
            Network.learningRate = 0.2;
            Network.momentumScalar = 0.02;
            Network.batchSize = 32;
            Network mainNN = new Network(new int[] { 784, 100, 10 });
            int numEpochs = 15;

            //set up training samples
            //assuming a (row x column) image

            List<TrainingSample> trainingSamples = new List<TrainingSample>();

            StreamReader sr = new StreamReader(File.OpenRead("mnist_train.csv"));
            String line = sr.ReadLine(); //skips first line
            int setupIdx = 0;
            List<Task> samplesToAdd = new List<Task>();
            while ((line = sr.ReadLine()) != null)
            {
                String lineDuplicate = line;
                Task t = new Task(() => createSample(lineDuplicate, trainingSamples));
                samplesToAdd.Add(t);
                Console.WriteLine($"Sample: {setupIdx}");
                setupIdx++;
            }

            samplesToAdd.ForEach(task => task.Start());
            Task.WaitAll(samplesToAdd.ToArray());

            sr.Close();
            Console.WriteLine("Ready to train");

            //train network
            for (int epoch = 0; epoch < numEpochs; epoch++)
            {
                double mse = 0;
                int numBatches = trainingSamples.Count / Network.batchSize;

                //batching
                for (int batchIdx = 0; batchIdx < numBatches; batchIdx++) //for each batch
                {
                    double batchMse = trainBatch(batchIdx, mainNN, trainingSamples);

                    mse += batchMse / Network.batchSize;
                    mainNN.updateWeightsAndBiases();
                    //Console.WriteLine($"Epoch {epoch + 1} / {numEpochs}      Batch #{batchIdx + 1} / {numBatches}      BMSE = {batchMse / Network.batchSize}");
                }

                Console.WriteLine("Epoch: {0}         MSE: {1}", epoch + 1, mse / numBatches);
            }

            //results
            testNetwork(mainNN, "mnist_train.csv");
            testNetwork(mainNN, "mnist_test.csv");

            Application.Run(new Form1(mainNN));
        }

        

        public static void createSample(String line, List<TrainingSample> trainingSamples)
        {
            if (line == null)
            {
                Console.WriteLine("line is null");
                return;
            }

            String[] dividedString = line.Split(',');

            //standardize inputs
            double[] standardizedPixelValues = new double[784];
            for (int i = 0; i < standardizedPixelValues.Length; i++)
                standardizedPixelValues[i] = double.Parse(dividedString[i + 1]) / 255.0;

            //classify output
            double[] targets = new double[10];
            int trgIndex = int.Parse(dividedString[0]);
            targets[trgIndex] = 1;

            lock (trainingSamples)
            {
                trainingSamples.Add(new TrainingSample(standardizedPixelValues, targets));
            }
        }
    }
}
