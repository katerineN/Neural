using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows.Forms;
using Accord.Diagnostics;
using System.Diagnostics;
using System.Threading.Tasks;

namespace NeuralNetwork1
{
    public class StudentNetwork : BaseNetwork
    {

        public class Neuron
        {
            //входной сигнал
            public double input = 0;

            //выходной сигнал
            public double output = 0;

            //ошибка
            public double error = 0;

            //сигнал поляризации 
            public static double biasSignal = 1.0;

            //генератор для инициализации весов
            private static Random randGenerator = new Random();

            //Минимальное значение для инициализации весов
            private static double initMinWeight = -1;

            //Максимальное значение для инициализации весов
            private static double initMaxWeight = 1;

            //Колво узлов на предыдущем слое
            public int inputLayerSize = 0;

            //Массив входных весов нейрона
            public double[] weights = null;

            //Вес на сигнале поляризации
            public double biasWeight = 0.001;

            //Ссылка на предыдущий слой нейронов
            public List<Neuron> inputLayer = null;

            public Neuron()
            {
            }

            public Neuron(List<Neuron> prevLayer)
            {
                //задаем ссылку на предыдущий слов
                inputLayer = prevLayer;

                //if (prevLayer == null) return;

                //указываем колво узлов на предыдущем слое
                inputLayerSize = prevLayer.Count;
                //создаем вектор весов для заполнения
                weights = new double[inputLayerSize];

                for (int i = 0; i < weights.Length; i++)
                {
                    //weights[i] = initMinWeight + randGenerator.NextDouble() * (initMaxWeight - initMinWeight);
                    weights[i] = -1 + randGenerator.NextDouble() * 2;
                }
            }

            //Функция активации
            public void funcActivate()
            {
                input = biasWeight * biasSignal;
                for (int i = 0; i < inputLayerSize; i++)
                {
                    input += inputLayer[i].output * weights[i];
                }

                output = 1 / (1 + Math.Exp(-input));

                //сбросили входной сигнал
                input = 0;
            }

            public void BackError(double speed)
            {
                //ошибка в текущем нейроне
                error *= output * (1 - output);
                biasWeight += speed * error * biasSignal;

                //переносим ошибку на предыдущий слой
                for (int i = 0; i < inputLayerSize; i++)
                {
                    inputLayer[i].error += error * weights[i];
                }

                //корректируем веса
                for (int i = 0; i < inputLayerSize; i++)
                {
                    weights[i] += speed * error * inputLayer[i].output;
                }

                error = 0;
            }
        }



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

            public static double[] Mult(double[] vec, double[,] matr)
            {
                double[] res = new double[matr.GetLength(1)];
                Parallel.For(0, matr.GetLength(1), j =>
                {
                    double a = 0.0;
                    for (int i = 0; i < vec.Length - 1; ++i)
                    {
                        a += vec[i] * matr[i, j];
                    }

                    res[j] = a;
                });
                return res;
            }

            public static double[] MultVecMatrix(double[] m1, double[,] m2)
            {  
                double[] res = new double[m2.GetLength(1)];
                for (int i = 0; i < m2.GetLength(0); i++)
                {
                    double sum = 0.0;
                    for (int j = 0; j < m1.Length; j++)
                    {
                        sum += m1[i] * m2[i, j];
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

            //Функция активации
            public double Sigmoid(double a)
            {
                return 1 / (1 + Math.Exp(-a));
            }

            public double[] FuncActivate()
            {
                double[] temp = Mult(input, weights);
                double[] matrix = new double[temp.Length + 1];
                for (int i = 0; i < temp.Length; i++)
                {
                    matrix[i] = Sigmoid(temp[i]);
                }
                matrix[matrix.Length - 1] = 1;
                return matrix;
            }



            public void FillWeights(int length)
            {
                weights = new double[input.Length + 1, length];
                for (int i = 0; i < input.Length; i++)
                {
                    for (int j = 0; j < length - 1; j++)
                    {
                        weights[i, j] = initMinWeight + randGenerator.NextDouble() * (initMaxWeight - initMinWeight);
                    }
                }
            }
        }

        // Список всех слоев нейронных сетей
        //public List<List<Neuron>> Layers = new List<List<Neuron>>();

        public List<Layer> layers = new List<Layer>();
        public double Speed = 0.1;
        public Stopwatch Stopwatch = new Stopwatch();

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
                            //prev.weights[ii, j] = initMinWeight + randGenerator.NextDouble() * (initMaxWeight - initMinWeight);
                            prev.weights[ii, j] = Layer.randGenerator.NextDouble() - 0.5;
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
        public double[][] Backward(double[] sampleOutput)
        {
            int length = layers.Count;
            double[][] res = new double[length][];
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
                        res[i][j] = yj * (1 - yj) * (sampleOutput[j] * yj);
                    }
                }

                else
                {
                    for (int j = 0; j < layers[i].input.Length - 1; j++)
                    {
                        double yj = layers[i].input[j];
                        double sum = 0;
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

        public void CreateNeuralNetwork(Sample sample)
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

        public double EstimatedError(double[] output)
        {
            double res = 0;
            for (int i = 0; i < output.Length; ++i)
                res += Math.Pow(layers.Last().input[i] - output[i], 2);
            return res;
        }

        public double HelpTrain(Sample sample)
        {
            Layer.data = sample.input;
            CreateNeuralNetwork(sample);
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

        protected override double[] Compute(double[] input)
        {
            CreateNeuralNetwork(input);
            
            return layers.Last().input.Take(layers.Last().input.Length - 1).ToArray();
        }
        
        private double TrainEpoch(double[][] inputs, double[][] outputs, bool parallel)
        {
            double err = 0.0;
            if (parallel)
                Parallel.ForEach(inputs.Zip(outputs, (i, o) => (i, o)), p => err += HelpTrain(p.i, p.o));
            else
                foreach (var pair in inputs.Zip(outputs, (i, o) => (i, o)))
                {
                    err += HelpTrain(pair.i, pair.o);
                }

            return err;
        }
        
        /// <summary>
        /// Обучаем нейросеть на готовом датасете
        /// </summary>
        /// <param name="samplesSet">Сам датасет</param>
        /// <param name="epochsCount">Количество эпох для тренировки</param>
        /// <param name="acceptableError"></param>
        /// <param name="parallel"></param>
        /// <returns></returns>
        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError,
            bool parallel)
        {
            //ошибка
            double error = 0;
            int epoch = 0;
            Stopwatch.Restart();
            while (epoch ++ < epochsCount)
            {
                int count = 0;
                error = 0;
                foreach (var sample in samplesSet.samples)
                {
                    count++;
                    error += HelpTrain(sample);
                    if (error > acceptableError)
                        if(count % 10 == 0)
                            OnTrainProgress((epoch * 1.0) / epochsCount, error, Stopwatch.Elapsed);
                    if(error <= acceptableError) 
                        break;
                    
                }
                if (error > acceptableError)
                    OnTrainProgress((epoch * 1.0) / epochsCount, error, Stopwatch.Elapsed);
                else break;
            }
            
            OnTrainProgress(1.0, double.MaxValue, Stopwatch.Elapsed);
            Stopwatch.Stop();
            return error;
            /*int epochs = 0;
            double err = double.MaxValue;

            double[][] inputs = new double[samplesSet.Count][];
            double[][] outputs = new double[samplesSet.Count][];

            //  Теперь массивы из samplesSet группируем в inputs и outputs
            for (int i = 0; i < samplesSet.Count; ++i)
            {
                inputs[i] = samplesSet[i].input.Append(1).ToArray();
                outputs[i] = samplesSet[i].Output;
            }

            Stopwatch.Restart();

            double error = 0.0;

            while (epochs++ < epochsCount && (error = TrainEpoch(inputs, outputs, parallel)) > acceptableError)
            {
#if DEBUG
                Console.WriteLine(epochs + " " + error);
#endif
                Console.WriteLine(epochs + " " + error);
                OnTrainProgress((epochs * 1.0) / epochsCount, error, Stopwatch.Elapsed);
            }

            OnTrainProgress(1.0, err, Stopwatch.Elapsed);

            Stopwatch.Stop();

            return error;*/
        }
    }
}