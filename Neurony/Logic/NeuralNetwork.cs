using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Neurony.Logic
{
    class NeuralNetwork
    {
        private AbstractNeuralLayer[] layers;
        public AbstractNeuralLayer[] Layers
        {
            get { return layers; }
        }

        public NeuralNetwork(AbstractNeuralLayer[] layers)
        {
            this.layers = layers;
        }

        public double[] Output(double[] data)
        {
            foreach (AbstractNeuralLayer layer in Layers)
            {
                data = layer.Output(data);
            }
            return data;
        }

        public override string ToString()
        {
            string result = "<network>" + Environment.NewLine;
            foreach (AbstractNeuralLayer layer in Layers)
            {
                result += layer.ToString();
            }
            result += "</network>";
            return result;
        }
    }
}
