using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

/*
 *  Loop:
 *  -A <- best attribute
 *  -Assign A as decision attribute for Node
 *  -For each value of A
 *      Create a descendent of node
 * 
 *  -sort training examples to leaves
 *  -if examples perfectly classified STOP
 *      else iterate over leaves 
 * 
 * 
 *  S = set of training examples
 *  A = particular attribute?
 *  Gain(S,A) = Entropy (S) - average entropy over each set of examples you have over a particular value   
 * 
 * */


namespace AiAssignment4
{
    struct Data
    {
        public bool decision;
        public string[] attributeValues;
    }

    struct Feature
    {
        public string featureName;
        public string[] attributes;
        public int index;
    }


    class Node
    {
        public Feature feature;
        public Node[] branches;
        public Node previousNode;
        public string parentAttribute;

        public bool decision;
        public bool end;

        public Node(Feature feature)
        {
            this.feature = feature;
            branches = new Node[this.feature.attributes.Length];
        }

        public Node(bool decision)
        {
            this.decision = decision;
            end = true;
        }

    }

    class Program
    {

        static Feature[] features;
        static Data[] dataSet;

        static void Main(string[] args)
        {

            try
            {
                List<string> trainingFileContents = new List<string>();
                List<string> testingFileContents = new List<string>();

                //ReadFile("mydata.txt", ref trainingFileContents);
                ReadFile("train-titanic-fatalities.data", ref trainingFileContents);

                features = new Feature[int.Parse(trainingFileContents[2])];
                List<string> examples = new List<string>();
                string[] classLabels = { trainingFileContents[0], trainingFileContents[1] };


                for (int i = 3, k = 0; i < features.Length + 3; i++, k++)   //input file has number of features at index 2, so we start populating features array at index of 3
                {
                    string[] featureSplit = trainingFileContents[i].Split(' ');
                    features[k].featureName = featureSplit[0];
                    features[k].attributes = new string[featureSplit.Length - 1];
                    features[k].index = k;

                    for (int j = 0; j < features[k].attributes.Length; j++) features[k].attributes[j] = featureSplit[j + 1];
                }

                int exampleCount = int.Parse(trainingFileContents[features.Length + 3]);
                for (int i = features.Length + 4; i < trainingFileContents.Count(); i++) examples.Add(trainingFileContents[i]);


                var sw = new Stopwatch();
                sw.Start();

                dataSet = new Data[examples.Count()];
                for (int i = 0; i < examples.Count(); i++)
                {
                    RegexOptions options = RegexOptions.None;
                    Regex regex = new Regex("[ ]{2,}", options);
                    string temp = examples[i].Replace("\t", " ");
                    temp = regex.Replace(temp, " ");

                    string[] dataSplit = temp.Split(' ');

                    if (dataSplit[1].Equals(classLabels[0]))
                    {
                        dataSet[i].decision = true;
                    }
                    else
                    {
                        dataSet[i].decision = false;
                    }

                    dataSet[i].attributeValues = new string[features.Length];

                    for (int j = 2, attribIndex = 0; j < dataSplit.Length; j++, attribIndex++) dataSet[i].attributeValues[attribIndex] = dataSplit[j];

                }

                int positiveCount = dataSet.Where(n => n.decision).Count();
                int negativeCount = dataSet.Where(n => !n.decision).Count();
                float entropy = Entropy(positiveCount, negativeCount, positiveCount + negativeCount);
                float entropyDecision = entropy;


                float maxGain = 0;
                Feature chosenFeature = new Feature();
                for (int i = 0; i < features.Length; i++)
                {
                    float Gain = 0;
                    for (int featureIndex = 0; featureIndex < features[i].attributes.Length; featureIndex++)
                    {
                        positiveCount = dataSet.Where(n => (n.decision) && (n.attributeValues[i].Equals(features[i].attributes[featureIndex]))).Count();
                        negativeCount = dataSet.Where(n => (!n.decision) && (n.attributeValues[i].Equals(features[i].attributes[featureIndex]))).Count();
                        int count = positiveCount + negativeCount;

                        float var_a = (-((float)negativeCount / (float)count) * (float)Math.Log((float)negativeCount / (float)count, 2));
                        float var_b = ((float)positiveCount / (float)count) * (float)Math.Log((float)positiveCount / (float)count, 2);
                        if (float.IsNaN(var_a)) var_a = 0;
                        if (float.IsNaN(var_b)) var_b = 0;
                        entropy = var_a - var_b;

                        Gain -= ((float)count / (float)dataSet.Count() * entropy);
                    }
                    Gain += entropyDecision;

                    if (Gain >= maxGain)
                    {
                        chosenFeature = features[i];
                        maxGain = Gain;
                    }
                }








                //  START of TREE creation


                Node root = new Node(chosenFeature), currentNode = root;

                List<Data> filteredDataSet = new List<Data>();
                List<Node> currentLayer = null, nextLayer = new List<Node>();
                

                int currentClassificationIndex = 0;

                bool isRunning = true;

                while (isRunning)
                {


                    int currentFeatureLength = (currentNode.feature.attributes == null) ? 0: currentNode.feature.attributes.Length;

                    for(int i = 0; i < currentFeatureLength; i++)
                    {

                        if(currentNode.previousNode == null)
                        {
                            filteredDataSet = dataSet.Where(n => n.attributeValues[currentNode.feature.index].Equals(currentNode.feature.attributes[i])).ToList();
                        }
                        else
                        {
                            filteredDataSet = dataSet.Where(n => n.attributeValues[currentNode.previousNode.feature.index].Equals(currentNode.parentAttribute)).ToList();
                            
                        }


                        positiveCount = filteredDataSet.Where( n=> (n.decision) && (n.attributeValues[currentNode.feature.index].Equals(currentNode.feature.attributes[i])) ).Count();
                        negativeCount = filteredDataSet.Where(n => (!n.decision) && (n.attributeValues[currentNode.feature.index].Equals(currentNode.feature.attributes[i]))).Count();
                        int count = positiveCount + negativeCount;

                        if(positiveCount == count)
                        {
                            Node n = new Node(true);
                            n.previousNode = currentNode;
                            n.parentAttribute = currentNode.feature.attributes[i];
                            currentNode.branches[i] = n;
                            continue;
                        }

                        if(negativeCount == count)
                        {
                            Node n = new Node(false);
                            n.previousNode = currentNode;
                            n.parentAttribute = currentNode.feature.attributes[i];
                            currentNode.branches[i] = n;
                            continue;
                        }


                        if (currentLayer!=null && currentLayer.Count() > 50)
                        {
                            float posRatio = (float)positiveCount / (float)count;
                            if(posRatio < 0.5)
                            {
                                Node n = new Node(false);
                                n.previousNode = currentNode;
                                n.parentAttribute = currentNode.feature.attributes[i];
                                currentNode.branches[i] = n;
                                continue;
                            }
                            else
                            {
                                Node n = new Node(true);
                                n.previousNode = currentNode;
                                n.parentAttribute = currentNode.feature.attributes[i];
                                currentNode.branches[i] = n;
                                continue;
                            }

                        }


                        //calculate gain
                        entropyDecision = Entropy(positiveCount, negativeCount, count);
                        maxGain = 0;

                        foreach(var feature in features)
                        {

                            if (feature.featureName.Equals(currentNode.feature.featureName)) continue;

                            float Gain = 0;

                            foreach(var attribute in feature.attributes)
                            {

                                positiveCount = filteredDataSet.Where( n=> (n.decision) &&
                                    (n.attributeValues[feature.index].Equals(attribute))).Count();
                                negativeCount = filteredDataSet.Where(n => (!n.decision) &&
                                    (n.attributeValues[feature.index].Equals(attribute))).Count();

                                count = positiveCount + negativeCount;
                                entropy = Entropy(positiveCount, negativeCount, count);
                                
                                if(filteredDataSet.Count()!=0)
                                    Gain -= ((float)count / (float)filteredDataSet.Count() * entropy);

                            }

                            Gain += entropyDecision;

                            if(Gain >= maxGain)
                            {
                                maxGain = Gain;
                                chosenFeature = feature;
                            }

                        }

                        currentNode.branches[i] = new Node(chosenFeature);
                        currentNode.branches[i].previousNode = currentNode;
                        currentNode.branches[i].parentAttribute = currentNode.feature.attributes[i];

                    }

                    if(currentNode.branches!=null)
                        nextLayer.AddRange(currentNode.branches);

                    if (currentLayer == null)
                    {
                        //currentLayer = new List<Node>(currentNode.branches);
                        currentLayer = new List<Node>(nextLayer);
                        nextLayer.Clear();
                    }

                    if(currentClassificationIndex < currentLayer.Count())
                    {
                        //nextLayer.AddRange(currentNode.branches);
                        currentNode = currentLayer[currentClassificationIndex];
                        currentClassificationIndex++;
                    }
                    else
                    {
                        //here check to see if every branch has reached end decision.
                        isRunning = false;
                        foreach(var node in currentLayer) if (!node.end) { isRunning = true; break; }
                        //foreach (var node in currentNode.branches) if (!node.end) { isRunning = true; break; }
                        if (!isRunning) break;
                        //if not, keep classifying.

                        //Console.WriteLine("classifying: " + currentLayer.Count());

                        currentClassificationIndex = 0;
                        currentLayer = new List<Node>(nextLayer);
                        nextLayer.Clear();
                    }

                }

                long elapsedMilliseconds = sw.ElapsedMilliseconds; sw.Stop();


                PrintTree(root);

                //Console.WriteLine("ijou desu yo. jikan wa: " + elapsedMilliseconds);
            }
            catch (Exception e) //exception handling out of scope
            {
                if(e.Message != null)
                Console.WriteLine("Invalid input file: " + e.Message);
            }
        }


        static void PrintTree(Node tree)
        {
            if (tree.feature.attributes == null) return;
            

            Console.WriteLine("\n" +tree.feature.featureName + ": ");

            for(int i = 0; i < tree.feature.attributes.Length; i++)
            {
                Console.Write(tree.feature.attributes[i]);
                    
                if(tree.branches[i].end) Console.Write("==>" + tree.branches[i].decision);

                //if (tree.end) Console.Write("==>" + tree.decision);

                Console.Write("\t");
            }
            
            if (tree.branches == null) return;

            foreach(var branch in tree.branches)
            {
                PrintTree(branch);
            }


        }



        static float Entropy(int positiveCount, int negativeCount, int count)
        {

            var a = (-((float)positiveCount / (float)count) * (float)Math.Log((float)positiveCount / (float)count, 2));
            var b = ((float)negativeCount / (float)count) * (float)Math.Log((float)negativeCount / (float)count, 2);

            if (float.IsNaN(a)) a = 0;
            if (float.IsNaN(b)) b = 0;

            return a - b;
        }


        static bool ReadFile(string fileName, ref List<string> output)
        {
            try
            {
                string dir = Directory.GetCurrentDirectory() + "\\_data\\" + fileName, line;
                StreamReader file = new System.IO.StreamReader(dir);
                List<string> lines = new List<string>();

                while ((line = file.ReadLine()) != null) if (line.Length != 0 && line[0] != '/') lines.Add(line);

                file.Close();
                output = lines;
            }
            catch (Exception e)
            {
                if (e.Message != null) Console.WriteLine("Error reading data file: " + e.Message);
                return false;
            }
            return true;
        }


    }

}
