using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Neurony.Logic
{
    class Neuron
    {
        private double m_Bias;
        public double[] Weights;
        private double m_Result;

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
            m_Bias = bias;
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
            if (input.Length != Weights.Length)
                throw new System.ArgumentException();

            double result = m_Bias;

            for (int i = 0; i < input.Length; i++)
                result += input[i] * Weights[i];

            m_Result = ApplyTransitionFunction(result);
            return m_Result;
        }

        public override string ToString()
        {
            string result = "\t\t<neuron>" + Environment.NewLine;
            foreach (double weight in Weights)
            {
                result += "\t\t\t<connection weight=\"" + weight + "\"/>" + Environment.NewLine;
            }
            result += "\t\t\t<bias weight=\"" + m_Bias + "\"/>" + Environment.NewLine;
            result += "\t\t</neuron>" + Environment.NewLine;
            return result;
        }
    }
}
