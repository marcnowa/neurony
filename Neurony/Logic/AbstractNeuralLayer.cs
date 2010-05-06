using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Neurony.Logic
{
    abstract class AbstractNeuralLayer
    {
        public abstract Neuron[] Neurons { get; }
        public abstract string Type { get; }
        public abstract double[] Output(double[] data);

        public override string ToString()
        {
            string result = "\t<layer type=\"" + Type + "\">" + Environment.NewLine;
            foreach (Neuron neuron in Neurons)
            {
                result += neuron.ToString();
            }
            result += "\t</layer>" + Environment.NewLine;
            return result;
        }
    }
}
