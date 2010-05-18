﻿using System;
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

        public NeuralLayer (int inputSize, int neuronsSize, bool randomWeights, double[] randomWeightsLimits)
        {
            neurons = new Neuron[neuronsSize];

            for (int j = 0; j < neuronsSize; j++)
            {
                double[] weights = new double[inputSize];
                if (randomWeights)
                    KohonenLayer.RandomFill(weights, randomWeightsLimits);
                neurons[j] = new Neuron(weights, 0, TransitionFunction.Sigmoid);
            }
        }
		public void Learn(double[][] kohOutput, double[][] expectedVal, int length)
		{
			// kohOutput- zestaw odpowiedzi warstwy kohonena
			int phases = 4; // uczenie z nauczycielem

			if (neurons.Length != expectedVal[0].Length) //ilosc neuronow
			{
				throw new System.ArgumentException();
			}

			int iloscNeuronowKohonena = kohOutput[0].Length; //dlugosc kazdej odpowiedzi warstwy kohonena jest stala
			double ni = 0.1; //stala nauczania


			for (int phase = 1; phase < phases + 1; phase++)
			{
				for (int k = 0; k < length / phases; k++)
				{
					for (int x = 0; x < kohOutput.Length; x++) // dla kazdego obrazka 
					{
						double[] odp = kohOutput[x]; //odpowiedz sieci kohonena na obrazek
						double[] y = this.Output(odp); // wynik zwracany przez siec z dotychczasowymi wagami

						for (int j = 0; j < iloscNeuronowKohonena; j++) // (ilosc neuronow kohonena) (ilosc wag w v2)
						{
							for (int i = 0; i < neurons.Length; i++) // dla kazdego neuronu z v2
							{
								neurons[i].Weights[j] = neurons[i].Weights[j] - odp[j] * ni * (y[i] - expectedVal[x][i]) * TransitionFunctionDerivative(y[i]);
							}
						}
					}
					
				}
				ni /= 10;
			}
		}

		private double TransitionFunctionDerivative(double p)
		{
			switch (Neurons[0].TransitionFunction)
			{
				case TransitionFunction.Linear:
					return 1;
				case TransitionFunction.Sigmoid:
					return (1 - p) * p;
				default:
					throw new ArgumentException();
			}
			
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
