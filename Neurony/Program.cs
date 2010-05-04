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
            /*KohonenNetwork knet = new KohonenNetwork(3, 10, false, 3);
            foreach (Neuron item in knet.Layers[0].Neurons)
            {
                for (int i = 0; i < item.Position.Length; i++)
                {
                    Console.Write(item.Position[i] + " ");
                }
                Console.WriteLine();
            }
            Console.ReadLine();*/

            bool learn = false;
            String learnFilename = "";
            bool test = false;
            String testFilename = "";
            bool useNeighbourhood = true;
            bool randomWeights = false;
            bool saveOutputNet = false;
            String outputFile = "";
            bool input = false;
            String inputFilename = "";

            int dimension = 2;

            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == "-i")
                {
                    input = true;
                    inputFilename = args[i + 1];
                    i++;
                }
                if (args[i] == "-l")
                {
                    learn = true;
                    learnFilename = args[i + 1];
                    i++;
                }
                if (args[i] == "-nn" || args[i]=="--no-neighbourhood")
                {
                    useNeighbourhood = false;
                }
                if (args[i] == "-rw" || args[i] == "--random-weights")
                {
                    randomWeights = true;
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
            }

            AbstractNeuralNetwork net = null;

            if (input)
            {
                net = (KohonenNetwork)XML.XMLNetworkCreator.Create(inputFilename);
            }

            if (learn)
            {
                double[][] data = ReadData(learnFilename);

                net = new KohonenNetwork(data[0].Length, data.Length, randomWeights, dimension);
                (net as KohonenNetwork).Learn(data, 10000, useNeighbourhood);
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

        private static double[][] Test(AbstractNeuralNetwork net, double[][] data)
        {
            double[][] result = new double[data.Length][];
            for (int i = 0; i < data.Length; i++)
            {
                result[i] = net.Output(data[i]);
            }
            return result;
        }

        private static void SaveOutputNet(AbstractNeuralNetwork net, string outputFile)
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
    }
}
