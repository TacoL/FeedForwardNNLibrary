using System;

namespace FeedForwardNNLibrary
{
    public class ActivationFunctions
    {
        private ActivationFunctions(string activation) { this.activation = activation; }
        internal string activation;

        public static ActivationFunctions Tanh { get; private set; } = new ActivationFunctions("Tanh");
        public static ActivationFunctions ReLu { get; private set; } = new ActivationFunctions("ReLu");
        public static ActivationFunctions Softmax { get; private set; } = new ActivationFunctions("Softmax");
        public static ActivationFunctions None { get; private set; } = new ActivationFunctions("None");

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
