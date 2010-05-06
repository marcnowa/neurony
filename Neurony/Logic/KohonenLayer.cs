using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Neurony.Logic
{
    class KohonenLayer : AbstractNeuralLayer
    {
        private Neuron[] neurons;
        private int phases = 4;


        public KohonenLayer(Neuron[] neurons)
        {
            this.neurons = neurons;
        }
        public KohonenLayer(int inputSize, int neuronsSize, bool randomWeights, double[] randomWeightsLimits, int neighbourhoodDimension)
        {
            Init(inputSize, neuronsSize, randomWeights, randomWeightsLimits, neighbourhoodDimension);
        }
        public KohonenLayer(int inputSize, int neuronsSize)
        {
            Init(inputSize, neuronsSize, false, null, 2);
        }

        private void Init(int inputSize, int neuronsSize, bool randomWeights, double[] randomWeightsLimits, int neighbourhoodDimension)
        {
            neurons = new Neuron[neuronsSize];

            for (int j = 0; j < neuronsSize; j++)
            {
                double[] weights = new double[inputSize];
                if (randomWeights)
                    RandomFeel(weights, randomWeightsLimits);
                neurons[j] = new Neuron(weights, 0);
            }

            int length = (int)Math.Ceiling(Math.Pow(neuronsSize, 1.0 / neighbourhoodDimension));
            for (int i = 0; i < neuronsSize; i++)
            {
                double[] pos = new double[neighbourhoodDimension];
                int p = i;
                for (int j = 0; j < neighbourhoodDimension - 1; j++)
                {
                    pos[j] = p % length;
                    p /= length;
                }
                pos[neighbourhoodDimension - 1] = p;
                neurons[i].Position = pos;
            }
        }


        public override Neuron[] Neurons
        {
            get { return neurons; }
        }

        public override string Type
        {
            get { return "kohonen"; }
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

        private void RandomFeel(double[] weights, double[] limits)
        {
            Random r = new Random();
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = r.NextDouble()*(limits[1]-limits[0])+limits[0];
            }
        }

        private void UpdateWeights(double[] data, int phase, bool useNeighbourhood)
        {
            double distanceFactor = 1;

            Neuron closest = neurons.OrderBy(n => GetDistance(n.Weights, data)).First();

            foreach (Neuron neuron in neurons.OrderBy(n => GetDistance(n.Position, closest.Position)))
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
