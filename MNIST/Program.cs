global using FeedForwardNNLibrary;
global using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace MNIST
{
    static class Program
    {
        /// <summary>
        /// The main entry point for the application.
        /// </summary>
        [STAThread]
        public static void Main()
        {
            //Application.EnableVisualStyles();
            //Application.SetCompatibleTextRenderingDefault(false);

            // set up the network
            Network mainNN = new Network(784, 0.2, 0.02, 32);
            mainNN.AddLayer(100, ActivationFunctions.Tanh);
            mainNN.AddLayer(10, ActivationFunctions.Softmax);
            int numEpochs = 15;

            // set up training samples
            // assuming a (row x column) image

            List<TrainingSample> trainingSamples = new List<TrainingSample>();

            StreamReader sr = new StreamReader(File.OpenRead("mnist_train.csv"));
            String line = sr.ReadLine(); // skips first line
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

            // train network
            mainNN.train(trainingSamples, numEpochs);

            // results
            testNetwork(mainNN, "mnist_train.csv");
            testNetwork(mainNN, "mnist_test.csv");

            Application.Run(new Form1(mainNN));
        }

        public static void testNetwork(Network mainNN, string fileName)
        {
            StreamReader srTest = new StreamReader(File.OpenRead(fileName));
            String line = srTest.ReadLine(); //skips first line

            int successes = 0;
            int total = 0;
            while ((line = srTest.ReadLine()) != null)
            {
                String lineDuplicate = line;
                String[] dividedString = lineDuplicate.Split(',');

                //standardize inputs
                double[] standardizedPixelValues = new double[784];
                for (int i = 0; i < standardizedPixelValues.Length; i++)
                    standardizedPixelValues[i] = double.Parse(dividedString[i + 1]) / 255.0;

                //print output
                double[] output = mainNN.forwardPropagate(standardizedPixelValues);
                int label = int.Parse(dividedString[0]);
                int val = output.ToList().IndexOf(output.Max());
                if (val == label)
                    successes++;
                total++;
            }

            srTest.Close();
            Console.WriteLine($"{successes}, {total}");
            Console.WriteLine("Success Rate: " + ((double)successes / (double)total * 100d) + "%");
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