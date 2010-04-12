using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Neurony.Logic
{
    class NeuralNetwork
    {
        private NeuralLayer[] m_Layers;
        public NeuralNetwork(NeuralLayer[] layers)
        {
            m_Layers = layers;
        }
        public double[] Compute(double[] data)
        {
            foreach (NeuralLayer layer in m_Layers)
            {
                data = layer.Compute(data);
            }
            return data;
        }
    }
}
