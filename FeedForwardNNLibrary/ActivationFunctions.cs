using System;

namespace FeedForwardNNLibrary
{
    public class ActivationFunctions
    {
        internal string Activation;

        // Constructor
        private ActivationFunctions(string activation) { this.Activation = activation; }
        
        // Activation Functions
        public static ActivationFunctions Tanh = new ActivationFunctions("Tanh");
        public static ActivationFunctions ReLu = new ActivationFunctions("ReLu");
        public static ActivationFunctions Softmax = new ActivationFunctions("Softmax");
        public static ActivationFunctions None = new ActivationFunctions("None");

        public static ActivationFunctions ConvertFromString(string str)
        {
            if (str == null) throw new ArgumentNullException("str");

            switch(str)
            {
                case "Tanh": return Tanh;
                case "ReLu": return ReLu;
                case "Softmax": return Softmax;
                case "None": return None;
                default: throw new Exception("Not a valid activation function");
            }
        }
    }
}
