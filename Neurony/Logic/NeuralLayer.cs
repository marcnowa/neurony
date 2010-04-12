using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Neurony.Logic
{
    class NeuralLayer
    {
        private Neuron[] m_Neurons;
        public NeuralLayer(Neuron[] neurons)
        {
            m_Neurons = neurons;
        }
        public double[] Compute(double[] input)
        {
            double[] result = new double[m_Neurons.Length];
            for (int i = 0; i < m_Neurons.Length; i++)
                result[i] = m_Neurons[i].Compute(input);

            return result;
        }
    }
}
