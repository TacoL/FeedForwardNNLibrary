using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;
using FeedForwardNNLibrary;

namespace Iris_Flower
{
    internal static class Program
    {
        /// <summary>
        /// The main entry point for the application.
        /// </summary>
        [STAThread]
        static void Main()
        {
            // Create Training Samples
            List<TrainingSample> samples = new List<TrainingSample>();

            int sampleIdx = 0;
            StreamReader sr = new StreamReader(File.OpenRead(@"IRIS.csv"));
            String line = sr.ReadLine(); //skips first line
            while ((line = sr.ReadLine()) != null)
            {
                TrainingSample sample = new TrainingSample();
                double[] inputs = new double[4];
                double[] targets = new double[3];

                String[] dividedString = line.Split(',');
                for (int i = 0; i < dividedString.Length; i++)
                {
                    if (i < 4) //inputs
                    {
                        double value = Double.Parse(dividedString[i]);
                        inputs[i] = value;
                    }
                    else //output
                    {
                        switch (dividedString[i])
                        {
                            case "Iris-setosa":
                                targets[0] = 1;
                                targets[1] = 0;
                                targets[2] = 0;
                                break;
                            case "Iris-versicolor":
                                targets[0] = 0;
                                targets[1] = 1;
                                targets[2] = 0;
                                break;
                            case "Iris-virginica":
                                targets[0] = 0;
                                targets[1] = 0;
                                targets[2] = 1;
                                break;
                        }
                    }
                }

                sample.inputs = inputs; sample.targets = targets;
                samples.Add(sample);
                sampleIdx++;
            }

            // Standardize inputs
            for (int i = 0; i < 4; i++)
            {
                double maxValue = double.MinValue;
                samples.ForEach(sample =>
                {
                    if (sample.inputs[i] > maxValue) { maxValue = sample.inputs[i]; }
                });

                samples.ForEach(sample => sample.inputs[i] /= maxValue);
            }

            Network nn = new Network(4, .22, .12, samples.Count);
            nn.AddLayer(8, ActivationFunctions.Tanh);
            nn.AddLayer(3, ActivationFunctions.Softmax);
            nn.train(samples, 5000);

            // Evaluate
            int numMatched = 0;

            for (int i = 0; i < samples.Count; i++)
            {
                double[] outputs = nn.forwardPropagate(samples[i].inputs);

                int rowWithMaxValue = Array.IndexOf(outputs, outputs.Max());

                String[] names = { "Iris-setosa", "Iris-versicolor", "Iris-virginica" };
                if (samples[i].targets[rowWithMaxValue] == 1)
                {
                    Console.WriteLine("Match: " + names[rowWithMaxValue]);
                    numMatched++;
                }
                else
                {
                    Console.WriteLine("Error: Predicted = " + names[rowWithMaxValue] + ", Actual = ");
                }
            }

            Console.WriteLine(numMatched + "/" + samples.Count);

            nn.ExportModel(@"IrisFlowerModel.xml");

            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new Form1());
        }
    }
}
