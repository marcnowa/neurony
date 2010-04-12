using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Neurony.Logic
{
    class Neuron
    {
        private double m_Bias;
        private double[] m_Weights;
        private double m_Result;

        private TransitionFunction m_TransitionFunction;

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
            m_Weights = weights;
            m_TransitionFunction = function;
        }

        private double ApplyTransitionFunction(double input)
        {
            switch (m_TransitionFunction)
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
            if (input.Length != m_Weights.Length)
                throw new System.ArgumentException();

            double result = m_Bias;

            for (int i = 0; i < input.Length; i++)
                result += input[i] * m_Weights[i];

            m_Result = ApplyTransitionFunction(result);
            return m_Result;
        }
    }
}
