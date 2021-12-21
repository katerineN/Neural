﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;

namespace NeuralNetwork1
{

    public class StudentNetwork : BaseNetwork
    {
        public class Layer
        {
            public double[] input = null;
            public static double[] data = null;
            public Layer prev = null;

            public double[,] weights = null;

            //генератор для инициализации весов
            public static Random randGenerator = new Random();

            //Минимальное значение для инициализации весов
            private static double initMinWeight = -1;

            //Максимальное значение для инициализации весов
            private static double initMaxWeight = 1;

            public Layer(Layer pr, int l)
            {
                input = new double[l];
                //доп нейрон со значением 1 или -1
                input[l - 1] = 1;
                prev = pr;
            }

            public static double[] MultVecMatrix(double[] m1, double[,] m2)
            {  
                double[] res = new double[m2.GetLength(1)];
                for (int i = 0; i < m2.GetLength(1); i++)
                {
                    double sum = 0.0;
                    for (int j = 0; j < m1.Length - 1; j++)
                    {
                        sum += m1[j] * m2[j, i];
                    }

                    res[i] = sum;
                }

                return res;
            }

            public void SetData()
            {
                input = data;
            }

            public void SetData(double[] inp)
            {
                input = inp;
            }

            
            public double[] ProduceValue()
            {
                var temp = MultVecMatrix(input, weights);
                double[] matrix = new double[temp.Length + 1];
                for (int i = 0; i < temp.Length; i++)
                {
                    matrix[i] = Sigmoid(temp[i]);
                }
                matrix[matrix.Length - 1] = 1;
                return matrix;
            }

            public void ComputeValue()
            {
                if (prev == null)
                    input = data;
                else
                {
                    prev.ComputeValue();
                    var temp = prev.ProduceValue();
                    input = temp; 
                }

            }
            //Функция активации
            public double Sigmoid(double a)
            {
                return 1 / (1 + Math.Exp(-a));
            }

            public double[] FuncActivate()
            {
                double[] temp = MultVecMatrix(input, weights);
                double[] matrix = new double[temp.Length + 1];
                for (int i = 0; i < temp.Length; i++)
                {
                    matrix[i] = Sigmoid(temp[i]);
                }
                matrix[matrix.Length - 1] = 1;
                return matrix;
            }
            
        }
        
        public List<Layer> layers = new List<Layer>();
        public double Speed = 0.1;
        public Stopwatch Stopwatch = new Stopwatch();

        //Минимальное значение для инициализации весов
        private static double initMinWeight = -1;

        //Максимальное значение для инициализации весов
        private static double initMaxWeight = 1;
        /// <summary>
        /// Конструктор сети с указанием структуры (количество слоёв и нейронов в них)
        /// </summary>
        /// <param name="structure">Массив с указанием нейронов на каждом слое (включая сенсорный)</param>
        public StudentNetwork(int[] structure)
        {
            Layer prev = null;
            for (int i = 0; i < structure.Length; i++)
            {
                Layer temp = new Layer(prev, structure[i] + 1);
                layers.Add(temp);
                //зададим рандомно веса
                if (prev != null)
                {
                    //prev.FillWeights(temp.input.Length);
                    prev.weights = new double[prev.input.Length, temp.input.Length - 1];
                    for (int ii = 0; ii < prev.input.Length; ii++)
                    {
                        for (int j = 0; j < temp.input.Length - 1; j++)
                        {
                            prev.weights[ii, j] = initMinWeight + Layer.randGenerator.NextDouble() * (initMaxWeight - initMinWeight);
                            //prev.weights[ii, j] = Layer.randGenerator.NextDouble() - 0.5;
                        }
                    }
                }
                prev = temp;
            }
        }

        /// <summary>
        /// Обратное распространение ошибки
        /// </summary>
        /// <param name="sample"></param>
        public double[][] Backward(double[] SampleOutput)
        {
            int length = layers.Count;
            double[][] res = new double[layers.Count][];
            for (int i = 0; i < length; i++)
            {
                //без свободного
                res[i] = new double[layers[i].input.Length - 1];
            }

            for (int i = length - 1; i > 0; i--)
            {
                //если слой последний
                if (i == length - 1)
                {
                    for (int j = 0; j < layers[i].input.Length - 1; j++)
                    {
                        double yj = layers[i].input[j];
                        //высчитываем дельту
                        res[i][j] = yj * (1 - yj) * (SampleOutput[j] - yj);
                    }
                }

                else
                {
                    for (int j = 0; j < layers[i].input.Length - 1; j++)
                    {
                        double yj = layers[i].input[j];
                        double sum = 0.0;
                        for (int k = 0; k < res[i + 1].Length; k++)
                        {
                            sum += res[i + 1][k] * layers[i].weights[j, k];
                        }
                        res[i][j] = yj * (1 - yj) * sum;
                    }
                }
            }

            return res;
        }
        
        public void CreateNeuralNetwork()
        {
            for (int i = 0; i < layers.Count; i++)
            {
                if (i == 0)
                {
                    layers[i].SetData();
                }
                else
                {
                    layers[i].input = layers[i].prev.FuncActivate();
                }
            }
        }
        
        public void CreateNeuralNetwork(double[] inp)
        {
            for (int i = 0; i < layers.Count; i++)
            {
                if (i == 0)
                {
                    layers[i].SetData(inp);
                }
                else
                {
                    layers[i].input = layers[i].prev.FuncActivate();
                }
            }
        }

        public void Temp(List<Layer> l)
        {
            for (int i = 0; i < l.Count; i++)
            {
                if (i == 0)
                {
                    l[i].SetData();
                }
                else
                {
                    l[i].input = l[i].prev.ProduceValue();
                }
            }
        }
        public double EstimatedError(double[] output)
        {
            double res = 0.0;
            for (int i = 0; i < output.Length; ++i)
                res += Math.Pow(layers.Last().input[i] - output[i], 2);
            return res / 2;
        }

        public double HelpTrain(Sample sample)
        {
            Layer.data = sample.input;
            CreateNeuralNetwork();
            //layers.Last().ComputeValue();
            double error = EstimatedError(sample.Output);
            double[][] deltas = Backward(sample.Output);
            //корректируем веса
            for (int i = 1; i < layers.Count; i++)
            {
                for (int j = 0; j < layers[i - 1].input.Length; j++)
                {
                    for (int k = 0; k < layers[i].input.Length - 1; k++)
                    {
                        layers[i - 1].weights[j, k] += Speed * deltas[i][k] * layers[i - 1].input[j];
                    }
                }
            }

            return error;
        }
        
        public double HelpTrain(double[] input, double[] output)
        {
            CreateNeuralNetwork(input);
            double error = EstimatedError(output);
            double[][] deltas = Backward(output);
            //корректируем веса
            for (int i = 1; i < layers.Count; i++)
            {
                for (int j = 0; j < layers[i - 1].input.Length; j++)
                {
                    for (int k = 0; k < layers[i].input.Length - 1; k++)
                    {
                        layers[i - 1].weights[j, k] += Speed * deltas[i][k] * layers[i - 1].input[j];
                    }
                }
            }

            return error;
        }


        /// <summary>
        /// Обучение сети одному образу
        /// </summary>
        /// <param name="sample"></param>
        /// <returns>Количество итераций для достижения заданного уровня ошибки</returns>
        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            int cnt = 0;
            while (HelpTrain(sample) > acceptableError)
            {
                cnt++;
            }
            return cnt;
        }
        
        /// <summary>
        /// Обучаем нейросеть на готовом датасете
        /// </summary>
        /// <param name="samplesSet">Сам датасет</param>
        /// <param name="epochsCount">Количество эпох для тренировки</param>
        /// <param name="acceptableError"></param>
        /// <param name="parallel"></param>
        /// <returns></returns>
        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            /*for (int i = 0; i < samplesSet.Count; ++i)
            {
                samplesSet[i].input = samplesSet[i].input.Append(1).ToArray();
            }
            //ошибка
            double error = 0;
            //int epoch = 0;
            Stopwatch.Restart();
            /*while (epoch++ <epochsCount && (error = TrainEpoch(samplesSet, acceptableError,parallel))>acceptableError)
            {
                OnTrainProgress(((epoch * 1.0) / epochsCount), error, Stopwatch.Elapsed);
            }
            */
            /*while (epoch ++ < epochsCount)
            {
                int count = 0;
                error = 0;
                foreach (var sample in samplesSet.samples)
                {
                    count++;
                    error += HelpTrain(sample);
                    if (error > acceptableError)
                        if(count % 50 == 0)
                            OnTrainProgress((epoch * 1.0) / epochsCount, error, Stopwatch.Elapsed);
                    if(error <= acceptableError) 
                        break;
                }
                /*if (error > acceptableError)
                    OnTrainProgress((epoch * 1.0) / epochsCount, error, Stopwatch.Elapsed);
                else break;
            }
            
            OnTrainProgress(1.0, double.MaxValue, Stopwatch.Elapsed);
            Stopwatch.Stop();*/


            int totalSamplesCount = epochsCount * samplesSet.Count;
            int count = 0;
            double error = 0.0;
            double meanError;
            Stopwatch.Restart();
            for (int e = 0; e < epochsCount; e++)
            {
                for (int i = 0; i < samplesSet.samples.Count; i++)
                {
                    var s = samplesSet.samples[i];
                    error += HelpTrain(s);
                    count++;
                    if (i % 100 == 0)
                    {
                        OnTrainProgress(1.0 * count / totalSamplesCount, error / (e * samplesSet.Count + i + 1),
                            Stopwatch.Elapsed);
                    }
                }

                meanError = error / ((e + 1) * samplesSet.Count + 1);
                if (meanError <= acceptableError)
                {
                    OnTrainProgress(1.0, meanError, Stopwatch.Elapsed);
                    return meanError;
                }
            }
            meanError = error / (epochsCount * samplesSet.Count + 1);
            OnTrainProgress(1.0, meanError, Stopwatch.Elapsed);
            return meanError / (epochsCount * samplesSet.Count);
        }

        protected override double[] Compute(double[] input)
        {
            CreateNeuralNetwork(input);
            return layers.Last().input.Take(layers.Last().input.Length - 1).ToArray();
        }
    }
}