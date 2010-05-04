using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Neurony.Logic
{
    class NeuralLayer
    {
        public Neuron[] Neurons;
        public NeuralLayer(Neuron[] neurons)
        {
            Neurons = neurons;
        }
        public double[] Compute(double[] input)
        {
            double[] result = new double[Neurons.Length];
            for (int i = 0; i < Neurons.Length; i++)
                result[i] = Neurons[i].Compute(input);

            return result;
        }

        public override string ToString()
        {
            string result = "\t<layer transition_function=\"" + Neurons[0].TransitionFunction + "\">" + Environment.NewLine;
            foreach (Neuron neuron in Neurons)
            {
                result += neuron.ToString();
            }
            result += "\t</layer>" + Environment.NewLine;
            return result;
        }
    }
}
