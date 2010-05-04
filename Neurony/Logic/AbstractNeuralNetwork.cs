using System;
namespace Neurony.Logic
{
    abstract class AbstractNeuralNetwork
    {
        public abstract NeuralLayer[] Layers { get; }
        public abstract string Type { get; }
        public abstract double[] Output(double[] data);

        public override string ToString()
        {
            string result = "<network type=\""+Type+"\">" + Environment.NewLine;
            foreach (NeuralLayer layer in Layers)
            {
                result += layer.ToString();
            }
            result += "</network>";
            return result;
        }
    }
}
