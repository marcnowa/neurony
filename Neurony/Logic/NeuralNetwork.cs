using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Neurony.Logic
{
    class NeuralNetwork : Neurony.Logic.AbstractNeuralNetwork
    {
        private NeuralLayer[] layers;
        public override NeuralLayer[] Layers
        {
            get { return layers; }
        }
        public override string Type
        {
            get { return "normal"; }
        }

        public NeuralNetwork(NeuralLayer[] layers)
        {
            this.layers = layers;
        }

        public override double[] Output(double[] data)
        {
            foreach (NeuralLayer layer in Layers)
            {
                data = layer.Compute(data);
            }
            return data;
        }
    }
}
