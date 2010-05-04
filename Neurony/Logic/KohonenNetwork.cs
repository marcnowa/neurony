using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Neurony.Logic
{
    class KohonenNetwork : AbstractNeuralNetwork
    {
        private Neuron[] neurons;
        private int phases = 8;

        public override string Type
        {
            get { return "kohonen"; }
        }

        public override NeuralLayer[] Layers
        {
            get
            {
                return new NeuralLayer[1] { new NeuralLayer(this.neurons) };
            }
        }

        public override double[] Output(double[] input)
        {
            double minDistance = double.PositiveInfinity;
            int minNeuronId = -1;
            for (int i = 0; i < neurons.Length; i++)
            {
                double distance = GetDistance(input, neurons[i].Weights);
                if (distance < minDistance)
                {
                    minDistance = distance;
                    minNeuronId = i;
                }
            }
            double[] result = new double[neurons.Length];
            result[minNeuronId] = 1;
            return result;
        }

        public KohonenNetwork(Neuron[] neurons)
        {
            this.neurons = neurons;
        }

        public KohonenNetwork(int inputSize, int neuronsSize, bool randomWeights)
        {
            Init(inputSize, neuronsSize, randomWeights);
        }
        public KohonenNetwork(int inputSize, int neuronsSize)
        {
            Init(inputSize, neuronsSize, false);
        }

        private void Init(int inputSize, int neuronsSize, bool randomWeights)
        {
            neurons = new Neuron[neuronsSize];

            for (int j = 0; j < neuronsSize; j++)
            {
                double[] weights = new double[inputSize];
                if (randomWeights)
                    randomFeel(weights);
                neurons[j] = new Neuron(weights, 0);
            }
        }

        private void randomFeel(double[] weights)
        {
            Random r = new Random();
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = r.NextDouble();
            }
        }

        private void UpdateWeights(double[] data, int phase, bool useNeighbourhood)
        {
            double distanceFactor = 1;
            foreach (Neuron neuron in neurons.OrderBy(n => GetDistance(n.Weights, data)))
            {
                for (int j = 0; j < data.Length; j++)
                {
                    double difference = neuron.Weights[j] - data[j];

                    neuron.Weights[j] -= difference / (phase * distanceFactor);
                }
                if (useNeighbourhood)
                    distanceFactor *= Math.Pow(5, phase);
            }
        }
        
        public double GetDistance(double[] data1, double[] data2)
        {
            double result = 0;
            for (int i = 0; i < data1.Length; i++)
            {
                result += Math.Pow(data1[i] - data2[i], 2);
            }
            result = Math.Sqrt(result);
            return result;
        }

        public void Learn(double[][] data, int length, bool useNeighbourhood)
        {
            for (int phase = 1; phase < phases + 1; phase++)
            {
                for (int i = 0; i < length / phases; i++)
                {
                    for (int j = 0; j < data.Length; j++)
                    {
                        UpdateWeights(data[j], phase, useNeighbourhood);
                    }
                }
            }
        }
    }
}
