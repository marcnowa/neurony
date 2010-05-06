using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Neurony.Logic
{
    class NeuralLayer : AbstractNeuralLayer
    {
        private Neuron[] neurons;

        public override Neuron[] Neurons
        {
            get
            {
                return neurons;
            }
        }
        public NeuralLayer(Neuron[] neurons)
        {
            this.neurons = neurons;
        }
        public override double[] Output(double[] input)
        {
            double[] result = new double[Neurons.Length];
            for (int i = 0; i < Neurons.Length; i++)
                result[i] = Neurons[i].Compute(input);

            return result;
        }

        public override string Type
        {
            get { return "normal"; }
        }
    }
}
