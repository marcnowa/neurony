using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Neurony.Logic
{
    class Neuron
    {
        public double Bias;
        public double[] Weights;
        public double OutputSignal;
        public double[] InputSignal;

        public double Error;

        public double[] Position;

        public TransitionFunction TransitionFunction;

        public Neuron(double[] weights, double bias)
        {
            Init(weights, bias, TransitionFunction.Linear);
        }
        public Neuron(double[] weights, double bias, TransitionFunction function)
        {
            Init(weights, bias, function);
        }

        private void Init(double[] weights, double bias, TransitionFunction function)
        {
            Bias = bias;
            Weights = weights;
            TransitionFunction = function;
        }

        private double ApplyTransitionFunction(double input)
        {
            switch (TransitionFunction)
            {
                case TransitionFunction.Linear:
                    return input;
                case TransitionFunction.Sigmoid:
                    return (1.0 / (1.0 + Math.Pow(Math.E, -1.0 * input)));
                case TransitionFunction.Threshold:
                    if (input > 0) return 1;
                    return 0;
            }
            return 0;
        }

        public double Compute(double[] input)
        {
            InputSignal = input;

            if (input.Length != Weights.Length)
                throw new System.ArgumentException();

            double result = Bias;

            for (int i = 0; i < input.Length; i++)
                result += input[i] * Weights[i];

            OutputSignal = ApplyTransitionFunction(result);
            return OutputSignal;
        }

        public override string ToString()
        {
            string result = "\t\t<neuron>" + Environment.NewLine;
            foreach (double weight in Weights)
            {
                result += "\t\t\t<connection weight=\"" + weight + "\"/>" + Environment.NewLine;
            }
            result += "\t\t\t<bias weight=\"" + Bias + "\"/>" + Environment.NewLine;
            result += "\t\t</neuron>" + Environment.NewLine;
            return result;
        }

        internal void RandomFill(double[] limits)
        {
            Random r = new Random();
            for (int i = 0; i < Weights.Length; i++)
            {
                Weights[i] = r.NextDouble() * (limits[1] - limits[0]) + limits[0];
            }

            Bias = r.NextDouble() * (limits[1] - limits[0]) + limits[0];
        }

        internal void UpdateWeights(double learningRate, double transDerOut)
        {

            for (int j = 0; j < Weights.Length; j++)
            {
                this.Weights[j] += learningRate * this.Error * transDerOut * this.InputSignal[j]; ;
            }
            this.Bias += learningRate * this.Error * transDerOut;
        }
    }
}
