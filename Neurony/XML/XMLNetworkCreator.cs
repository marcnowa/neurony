using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Xml;
using System.Xml.XPath;
using Neurony.Logic;

namespace Neurony.XML
{
    class XMLNetworkCreator
    {
        static public AbstractNeuralNetwork Create(string path)
        {
            using (XmlReader reader = XmlReader.Create(path))
            {
                XPathDocument doc = new XPathDocument(reader);
                XPathNavigator rootNav = doc.CreateNavigator();
                rootNav.MoveToChild("network", "");
                string type = rootNav.GetAttribute("type", "");

                List<NeuralLayer> layers = new List<NeuralLayer>();
                foreach (XPathNavigator layerNav in rootNav.SelectChildren("layer", ""))
                {
                    TransitionFunction transitionFunction = TransitionFunction.Linear;

                    string tf = layerNav.GetAttribute("transition_function", "");
                    if (tf.Equals("sigmoid"))
                    {
                        transitionFunction = TransitionFunction.Sigmoid;
                    }
                    if (tf.Equals("threshold"))
                    {
                        transitionFunction = TransitionFunction.Threshold;
                    }

                    List<Neuron> neurons = new List<Neuron>();
                    foreach (XPathNavigator neuronNav in layerNav.SelectChildren("neuron", ""))
                    {
                        List<double> fractions = new List<double>();
                        foreach (XPathNavigator connectionNav in neuronNav.SelectChildren("connection", ""))
                        {
                            string weight = connectionNav.GetAttribute("weight", "");
                            double w;
                            Double.TryParse(weight, out w);
                            fractions.Add(w);
                        }
                        neuronNav.MoveToChild("bias", "");
                        string biasS = neuronNav.GetAttribute("weight", "");
                        double bias;
                        Double.TryParse(biasS, out bias);

                        neurons.Add(new Neuron(fractions.ToArray(), bias, transitionFunction));
                    }
                    layers.Add(new NeuralLayer(neurons.ToArray()));
                }
                if (type == "kohonen")
                    return new KohonenNetwork(layers[0].Neurons);
                else
                    return new NeuralNetwork(layers.ToArray());
            }
        }
    }
}
