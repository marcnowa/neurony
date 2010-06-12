using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using Neurony.Logic;

namespace Neurony
{
    class Program
    {
        static void Main(string[] args)
        {
            bool learn = false;
            bool cp_learn = false;
            String learnFilename = "";
            bool test = false;
            String testFilename = "";
            bool useNeighbourhood = true;
            bool randomWeights = false;
            double[] randomWeightsLimits = null;
            bool saveOutputNet = false;
            String outputFile = "";
            bool input = false;
            String inputFilename = "";

            bool bplearn = false;

            bool usebias = true;

            bool wh = false;

            int dimension = 2;

            // Learn params - default values
            // 1. Kohonen params 
            int phases = 4;
            // 2. Counter Propagation params
            double ni = 0.1;
            int lengthOfPhase = 1000;
            double divisor = 10;

            int[] neuronCounts = new int[3] { 3, 2, 1 };

            for (int i = 0; i < args.Length; i++)
            {

                if (args[i] == "-i")
                {
                    input = true;
                    inputFilename = args[i + 1];
                    i++;
                }

                if (args[i] == "-nb")
                {
                    usebias = false;
                }

                if (args[i] == "-bpl")
                {
                    bplearn = true;
                    learnFilename = args[i + 1];
                    i++;
                }

                if (args[i] == "-l" || args[i] == "--learn")
                {
                    learn = true;
                    learnFilename = args[i + 1];
                    i += 2;
                }

                if (args[i] == "-phases")
                {
                    phases = int.Parse(args[i + 1]);
                    i += 2;
                }

                if (args[i] == "-cpl" || args[i] == "--counter-propagation-learn")
                {
                    cp_learn = true;
                    learnFilename = args[i + 1];
                    i++;
                }

                if (args[i] == "-ni")
                {
                    ni = double.Parse(args[i + 1]);
                    i++;
                }
                if (args[i] == "-length")
                {
                    lengthOfPhase = int.Parse(args[i + 1]);
                    i++;
                }
                if (args[i] == "-divisor")
                {
                    divisor = double.Parse(args[i + 1]);
                    i++;
                }

                if (args[i] == "-wh")
                {
                    wh = true;
                }

                if (args[i] == "-nn" || args[i] == "--no-neighbourhood")
                {
                    useNeighbourhood = false;
                }
                if (args[i] == "-rw" || args[i] == "--random-weights")
                {
                    randomWeights = true;
                    string limits = args[i + 1];
                    i++;
                    string[] lim = limits.Split(new char[1] { ';' }, 2);
                    randomWeightsLimits = new double[2];
                    randomWeightsLimits[0] = double.Parse(lim[0]);
                    randomWeightsLimits[1] = double.Parse(lim[1]);
                }
                if (args[i] == "-o")
                {
                    saveOutputNet = true;
                    outputFile = args[i + 1];
                    i++;
                }
                if (args[i] == "-t")
                {
                    test = true;
                    testFilename = args[i + 1];
                    i++;
                }
                if (args[i] == "-d")
                {
                    dimension = int.Parse(args[i + 1]);
                    i++;
                }

                if (args[i] == "-h")
                {
                    Console.WriteLine(File.ReadAllText("Help.txt"));
                    return;
                }

                if (args[i] == "-neurons")
                {
                    neuronCounts = ParseNeuronCounts(args[i + 1]);
                    i++;
                }
            }

            /*Console.WriteLine("wyjściowo: phases = " + phases
                + " / ni = " + ni
                + " / lengthOfPhase= " + lengthOfPhase
                + " / divisor= " + divisor);*/

            NeuralNetwork net = null;

            if (input)
            {
                net = XML.XMLNetworkCreator.Create(inputFilename);
            }

            if (learn)
            {
                double[][] data = ReadData(learnFilename);

                int neuralCount = data.Length;
                int inputSize = data[0].Length;
                KohonenLayer kohonenLayer = new KohonenLayer(inputSize, neuralCount, randomWeights, randomWeightsLimits, dimension);
                net = new NeuralNetwork(new AbstractNeuralLayer[1] { kohonenLayer });
                kohonenLayer.Learn(data, 10000, useNeighbourhood, phases);
            }
            
            if (cp_learn)
            {
                //dane wejściowe dla warstwy Kohonena + wyjściowe dla warstwy 2giej (output programu)
                List<double[][]> data = ReadInputAndAnswersData(learnFilename);

                double[][] kohonenInput = data[0];
                double[][] expectedOutputForSecondLayer = data[1];

                int inputSize = kohonenInput[0].Length;
                int neuralCount = kohonenInput.Length; // == expectedOutputForSecondLayer.Length
                KohonenLayer kohonenLayer = new KohonenLayer(inputSize, neuralCount, randomWeights, randomWeightsLimits, dimension);

                inputSize = neuralCount;
                neuralCount = expectedOutputForSecondLayer[0].Length;

                TransitionFunction tf = TransitionFunction.Sigmoid;
                if (wh)
                {
                    tf = TransitionFunction.Linear;
                }

                NeuralLayer secondLayer = new NeuralLayer(inputSize, neuralCount, randomWeights, randomWeightsLimits, tf, true);

                net = new NeuralNetwork(new AbstractNeuralLayer[2] { kohonenLayer, secondLayer });

                net.CounterPropagationLearn(kohonenInput, useNeighbourhood,
                    expectedOutputForSecondLayer, ni, divisor,
                    phases, lengthOfPhase);
            }

            if (bplearn)
            {
                List<double[][]> data = ReadInputAndAnswersData(learnFilename);
                double[][] inputData = data[0];
                double[][] expectedOutput = data[1];

                int inputSize = inputData[0].Length;

                net = new NeuralNetwork(inputSize, neuronCounts, randomWeightsLimits, TransitionFunction.Sigmoid, usebias);

                net.BackPropagationLearn(inputData, expectedOutput, lengthOfPhase, ni, usebias);
            }

            if (saveOutputNet)
            {
                SaveOutputNet(net, outputFile);
            }

            if (test)
            {
                double[][] data = ReadData(testFilename);
                double[][] result = Test(net, data);
                PrintResult(result);
            }
            Console.ReadKey();
        }

        private static int[] ParseNeuronCounts(string p)
        {
            List<int> result = new List<int>();
            foreach (string n in p.Split(';'))
            {
                result.Add(int.Parse(n));
            }
            return result.ToArray();
        }

        private static void PrintResult(double[][] result)
        {
            foreach (var vector in result)
            {
                foreach (var item in vector)
                {
                    Console.Write(item + " ");
                }
                Console.WriteLine();
            }
        }

        private static double[][] Test(NeuralNetwork net, double[][] data)
        {
            double[][] result = new double[data.Length][];
            for (int i = 0; i < data.Length; i++)
            {
                result[i] = net.Output(data[i]);
            }
            return result;
        }

        private static void SaveOutputNet(NeuralNetwork net, string outputFile)
        {
            StreamWriter sw = new StreamWriter(outputFile);
            sw.Write(net);
            sw.Close();
        }

        private static double[][] ReadData(String learnFilename)
        {
            StreamReader sr = File.OpenText(learnFilename);

            String line;

            List<double[]> result = new List<double[]>();

            while ((line = sr.ReadLine()) != null)
            {
                List<double> vector = new List<double>();
                foreach (String s in line.Split(';'))
                {
                    vector.Add(double.Parse(s));
                }
                result.Add(vector.ToArray());
            }
            sr.Close();

            return result.ToArray();
        }

        private static List<double[][]> ReadInputAndAnswersData(String learnFilename)
        {
            StreamReader sr = File.OpenText(learnFilename);

            String line;

            List<double[]> r1 = new List<double[]>();
            List<double[]> r2 = new List<double[]>();
            while ((line = sr.ReadLine()) != null)
            {
                string[] tab = line.Split(' ');
                string kohData = tab[0];
                string secLayerData = tab[1];

                List<double> v1 = new List<double>();
                foreach (String s in kohData.Split(';'))
                {
                    v1.Add(double.Parse(s));
                }
                r1.Add(v1.ToArray());

                List<double> v2 = new List<double>();
                foreach (String s in secLayerData.Split(';'))
                {
                    v2.Add(double.Parse(s));
                }
                r2.Add(v2.ToArray());
            }
            sr.Close();

            List<double[][]> result = new List<double[][]>();
            result.Add(r1.ToArray());
            result.Add(r2.ToArray());

            return result;
        }
    }
}
