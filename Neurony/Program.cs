using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Neurony.Logic;

namespace Neurony
{
    class Program
    {
        static void Main(string[] args)
        {
            NeuralNetwork net = XML.XMLNetworkCreator.Create(args[0]);

            /*double[] result = net.Compute(new double[2] { 0, 0 });
            Console.WriteLine(result[0]);

            result = net.Compute(new double[2] { 0, 1.0 });
            Console.WriteLine(result[0]);

            result = net.Compute(new double[2] { 1.0, 0 });
            Console.WriteLine(result[0]);

            result = net.Compute(new double[2] { 1.0, 1.0 });
            Console.WriteLine(result[0]);*/

            string s = "";
            while (s != null)
            {
                s = Console.ReadLine();

                try
                {
                    List<double> input = new List<double>();
                    foreach (string item in s.Split(' '))
                    {
                        double d;
                        Double.TryParse(item, out d);
                        input.Add(d);
                    }

                    double[] res = net.Compute(input.ToArray());

                    foreach (double item in res)
                    {
                        Console.Write(item);
                        Console.Write(" ");
                    }
                    Console.WriteLine();
                }
                catch (System.ArgumentException)
                {
                    Console.WriteLine("Niepoprawny wektor danych wejściowych.");
                }
            }

            Console.ReadLine();
        }
    }
}
