﻿using System;
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
		public void Learn(double[][] kohonenInput, bool useNeighbourhood, 
			double[][] expectedVal, double ni, double divisor,
			int phases, int phaseLength)
		{
			if (layers.Length != 2)
			{
				throw new System.ArgumentException();
			}

			KohonenLayer kohonenLayer = (KohonenLayer) layers[0];
			NeuralLayer secondLayer = (NeuralLayer) layers[1];

			kohonenLayer.Learn(kohonenInput,  phaseLength, useNeighbourhood, phases);
			Console.WriteLine("Warstwa Kohonena zostala nauczona rozpoznawania wzorcow");

			
			List<double[]> kohonenOutput = new List<double[]>();
			foreach (double[] inputData in kohonenInput)
			{
				double[] kohOut = kohonenLayer.Output(inputData);
				kohonenOutput.Add(kohOut); // 0,0,1,0,0,0,0
				Console.Write(" Kohonen output -> ");
				foreach (var d in kohOut)
				{
					Console.Write(d + " ");
				}
				Console.WriteLine();
			}


			secondLayer.Learn(kohonenOutput.ToArray(), expectedVal, phases, phaseLength, ni, divisor);
			Console.WriteLine("\n Warstwa Druga zostala nauczona rozpoznawania wzorcow");
			foreach (double[] kohOut in kohonenOutput)
			{
				Console.Write(" Grossberg Output------> ");
				foreach (var outp in secondLayer.Output(kohOut))
				{
					Console.WriteLine(outp + " ");
					Console.WriteLine();
				}
			}
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
