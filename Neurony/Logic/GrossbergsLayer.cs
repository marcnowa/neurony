using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Neurony.Logic
{
	class GrossbergsLayer : NeuralLayer
    {
        private Neuron[] neurons;


        public GrossbergsLayer(int inputSize, int neuronsSize, bool randomWeights, double[] randomWeightsLimits)
			: base(inputSize, neuronsSize, randomWeights, randomWeightsLimits)	
        {
			
			this.neurons = new Neuron[neuronsSize];

            for (int j = 0; j < neuronsSize; j++)
            {
                double[] weights = new double[inputSize];
                if (randomWeights)
                    KohonenLayer.RandomFeel(weights, randomWeightsLimits);
                neurons[j] = new Neuron(weights, 0, TransitionFunction.Linear);
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

			for (int x = 0; x < kohOutput.Length; x++) // dla kazdego obrazka 
			{
				for (int k = 0; k < length / 5; k++)
				{

					double[] odp = kohOutput[x]; //odpowiedz sieci kohonena na obrazek

					for (int j = 0; j < iloscNeuronowKohonena; j++) // (ilosc neuronow kohonena) (ilosc wag w v2)
					{
						double[] y = Output(odp); // wynik zwracany przez siec z dotychczasowymi wagami

						for (int i = 0; i < neurons.Length; i++) // dla kazdego neuronu z v2
						{
							neurons[i].Weights[j] = neurons[i].Weights[j] + odp[j] * ni * (y[i] - expectedVal[x][i]);
						}

					}
					ni /= 3;
				}
			}
		}

		public override string Type
		{
			get { return "grossberg"; }
		}
    }
}
