using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;
using FeedForwardNNLibrary;

namespace XOR
{
    internal static class Program
    {
        /// <summary>
        /// The main entry point for the application.
        /// </summary>
        [STAThread]
        static void Main()
        {
            //set up the network
            Network nn = new Network(2, 0.1, 0.001, 4);
            nn.AddLayer(2, ActivationFunctions.Tanh);
            nn.AddLayer(1, ActivationFunctions.Tanh);

            //set up training samples
            List<TrainingSample> trainingSamples = new List<TrainingSample>();
            trainingSamples.Add(new TrainingSample(new double[] { 0, 0 }, new double[] { 0 }));
            trainingSamples.Add(new TrainingSample(new double[] { 0, 1 }, new double[] { 1 }));
            trainingSamples.Add(new TrainingSample(new double[] { 1, 0 }, new double[] { 1 }));
            trainingSamples.Add(new TrainingSample(new double[] { 1, 1 }, new double[] { 0 }));

            //train network
            nn.train(trainingSamples, 100000);

            //results
            Console.WriteLine(nn.forwardPropagate(new double[] { 0, 0 })[0]);
            Console.WriteLine(nn.forwardPropagate(new double[] { 0, 1 })[0]);
            Console.WriteLine(nn.forwardPropagate(new double[] { 1, 0 })[0]);
            Console.WriteLine(nn.forwardPropagate(new double[] { 1, 1 })[0]);

            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new Form1());
        }
    }
}
