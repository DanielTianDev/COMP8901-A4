using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace AiAssignment4
{
    class Program
    {
        static void Main(string[] args)
        {

            List<string> fileContents = new List<string>();

            ReadFile("test-titanic-fatalities.data", ref fileContents);

            foreach (string f in fileContents) Console.WriteLine(f);


        }

        static bool ReadFile(string fileName, ref List<string> output)
        {
            string line;
            string dir = Directory.GetCurrentDirectory() + "\\_data\\" + fileName;

            try
            {
                StreamReader file = new System.IO.StreamReader(dir); // Read the file line by line.  
                List<string> lines = new List<string>();

                while ((line = file.ReadLine()) != null) lines.Add(line);
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
