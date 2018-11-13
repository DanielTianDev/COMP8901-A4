using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

/*
 * @Author: Daniel Tian (A00736794)
 * @Date November 12, 2018
 * 
 * AI Assignment 4 (ID3 Decision tree)
 * 
 * This code was written without recursion because I felt like challenging myself, 
 * but ended up creating really really bad code. Lesson learned for the future; apologies for this.
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
        static Data[] testingSet;
        static int treeMaxDepth = 5;

        static void Main(string[] args)
        {

            try
            {
                List<string> trainingFileContents = new List<string>();
                List<string> testingFileContents = new List<string>();

                if (args.Length > 0)
                { 
                    ReadFile(args[0], ref trainingFileContents);
                    ReadFile(args[1], ref testingFileContents);
                }
                else
                {
                    ReadFile("train-titanic-fatalities.data", ref trainingFileContents);
                    ReadFile("train-titanic-fatalities.data", ref testingFileContents); //default data file if none provided
                }


                features = new Feature[int.Parse(trainingFileContents[2])];
                string[] classLabels = { trainingFileContents[0], trainingFileContents[1] };

                var sw = new Stopwatch(); sw.Start();

                dataSet = GenerateDataSet(trainingFileContents, classLabels);
                testingSet = GenerateDataSet(testingFileContents, classLabels);

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

                        if(currentNode.previousNode == null) filteredDataSet = dataSet.Where(n => n.attributeValues[currentNode.feature.index].Equals(currentNode.feature.attributes[i])).ToList();
                        else filteredDataSet = dataSet.Where(n => n.attributeValues[currentNode.previousNode.feature.index].Equals(currentNode.parentAttribute)).ToList();

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


                        if (currentLayer!=null && currentLayer.Count() > treeMaxDepth)
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
                        currentLayer = new List<Node>(nextLayer);
                        nextLayer.Clear();
                    }

                    if(currentClassificationIndex < currentLayer.Count())
                    {
                        currentNode = currentLayer[currentClassificationIndex];
                        currentClassificationIndex++;
                    }
                    else
                    {
                        //here check to see if every branch has reached end decision.
                        isRunning = false;
                        foreach(var node in currentLayer) if (!node.end) { isRunning = true; break; }
                        if (!isRunning) break;
                        //if not, keep classifying.

                        currentClassificationIndex = 0;
                        currentLayer = new List<Node>(nextLayer);
                        nextLayer.Clear();
                    }

                }

                long elapsedMilliseconds = sw.ElapsedMilliseconds; sw.Stop();

                Console.WriteLine("================================Decision Tree================================: \n");

                PrintTree(root);

                Console.WriteLine("\n\n================================Testing================================: \n");

                int correct = 0, incorrect = 0;
                foreach (var set in testingSet)
                {
                    if(TraverseTree(root, set.attributeValues) == set.decision)
                    {
                        correct++;
                    }
                    else
                    {
                        incorrect++;
                    }
                }

                float incorrectFraction = ((float)incorrect /(float)testingSet.Count());


                Console.WriteLine("\nIncorrect: " + incorrect + " out of " + testingSet.Length);
                Console.WriteLine("Fraction incorrect: " + incorrectFraction);
                Console.WriteLine("\nProgram Finished, Time took (milliseconds): " + elapsedMilliseconds);
            }
            catch (Exception e) //exception handling out of scope
            {
                Console.WriteLine("Invalid input file: ");
            }
        }


        static void PrintTree(Node tree)
        {

            if (tree.end)
            {

                List<string> result = new List<string>();

                Node current = tree;
                while (current.parentAttribute != null)
                {
                    if (current.end)
                    {
                        result.Add(current.decision + "");
                        result.Add(current.parentAttribute);
                    }
                    else
                    {
                        result.Add(current.feature.featureName);
                        result.Add(current.parentAttribute);
                    }

                    current = current.previousNode;
                }

                result.Add(current.feature.featureName);

                for(int i = result.Count()-1; i >= 0; i--) if (i == 0) Console.Write(result[i]); else Console.Write(result[i] + "->");
                Console.WriteLine();

                return;
            }

            foreach(var branch in tree.branches) PrintTree(branch);

        }

        static bool TraverseTree(Node tree, string[] attributeValues)
        {
            Node currentNode = tree;

            while (!currentNode.end)
            {
                if (currentNode.branches != null)
                {
                    for (int i = 0; i < tree.branches.Length; i++)
                    {
                        if (attributeValues.Contains(currentNode.feature.attributes[i]))
                        {

                            currentNode = currentNode.branches[i];
                            break;
                        }
                    }
                }
            }
            return currentNode.decision;
        }

        static float Entropy(int positiveCount, int negativeCount, int count)
        {
            var a = (-((float)positiveCount / (float)count) * (float)Math.Log((float)positiveCount / (float)count, 2));
            var b = ((float)negativeCount / (float)count) * (float)Math.Log((float)negativeCount / (float)count, 2);
            if (float.IsNaN(a)) a = 0;
            if (float.IsNaN(b)) b = 0;
            return a - b;
        }

        static Data[] GenerateDataSet(List<string> filecontents, string[] classLabels)
        {
            List<string> examples = new List<string>();

            for (int i = 3, k = 0; i < features.Length + 3; i++, k++)   //input file has number of features at index 2, so we start populating features array at index of 3
            {
                string[] featureSplit = filecontents[i].Split(' ');
                features[k].featureName = featureSplit[0];
                features[k].attributes = new string[featureSplit.Length - 1];
                features[k].index = k;
                for (int j = 0; j < features[k].attributes.Length; j++) features[k].attributes[j] = featureSplit[j + 1];
            }

            int exampleCount = int.Parse(filecontents[features.Length + 3]);
            for (int i = features.Length + 4; i < filecontents.Count(); i++) examples.Add(filecontents[i]);

            Data[] set = new Data[examples.Count()];
            for (int i = 0; i < examples.Count(); i++)
            {
                RegexOptions options = RegexOptions.None;
                Regex regex = new Regex("[ ]{2,}", options);
                string temp = examples[i].Replace("\t", " ");
                temp = regex.Replace(temp, " ");
                string[] dataSplit = temp.Split(' ');

                if (dataSplit[1].Equals(classLabels[0]))
                {
                    set[i].decision = true;
                }
                else
                {
                    set[i].decision = false;
                }

                set[i].attributeValues = new string[features.Length];
                for (int j = 2, attribIndex = 0; j < dataSplit.Length; j++, attribIndex++) set[i].attributeValues[attribIndex] = dataSplit[j];
            }

            return set;
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
