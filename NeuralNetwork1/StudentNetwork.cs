using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows.Forms;
using Accord.Diagnostics;
using System.Diagnostics;

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

            public Neuron(){ }
            
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
        
        // Список всех слоев нейронных сетей
        public List<List<Neuron>> Layers = new List<List<Neuron>>();

        /// <summary>
        /// Конструктор сети с указанием структуры (количество слоёв и нейронов в них)
        /// </summary>
        /// <param name="structure">Массив с указанием нейронов на каждом слое (включая сенсорный)</param>
        public StudentNetwork(int[] structure)
        {
            for (int i = 0; i < structure.Length; i++)
            {
                //Задаем слой сети
                List<Neuron> layer = new List<Neuron>();
                for (int ii = 0; ii < structure[i]; ii++)
                {
                    //заполняем первый слой
                    //так как слой ни на что не ссылается
                    if (i == 0)
                        layer.Add(new Neuron());
                    //иначе берем конструктор со ссылкой на предыдущий слой
                    else layer.Add(new Neuron(Layers[i - 1]));
                }
                Layers.Add(layer);
            }
        }

        /// <summary>
        /// Прямое распространение сигнала по сети по заданному образу
        /// </summary>
        /// <param name="sample"></param>
        public void Forward(Sample sample)
        {
            //по образу заполняем сенсоры
            for (int i = 0; i < sample.input.Length; i++)
            {
                Layers[0][i].output = sample.input[i];
            }
            
            //проходим по всем слоям (кроме 0) и выполняем нужные вычисления
            for (int i = 1; i < Layers.Count; i++)
            {
                for (int j = 0; j < Layers[i].Count; j++)
                {
                    Layers[i][j].funcActivate();
                }
            }

            //передаем в образ результат распознавания
            double[] temp = new double[Layers.Last().Count];
            for (int i = 0; i < Layers.Last().Count; i++)
            {
                temp[i] = Layers.Last()[i].output;
            }

            sample.ProcessPrediction(temp);
        }

        /// <summary>
        /// Обратное распространение ошибки
        /// </summary>
        /// <param name="sample"></param>
        public void Backward(Sample sample)
        {
            //берем данные с образа
            for (int i = 0; i < Layers[Layers.Count-1].Count; i++)
            {
                if ((int) sample.actualClass == i)
                {
                    Layers.Last()[i].error = (1 - Layers.Last()[i].output) * Layers.Last()[i].error;
                }
                else Layers.Last()[i].error = (0 - Layers.Last()[i].output) * Layers.Last()[i].error;
            }

            for (int i = Layers.Count-1; i > 0; i--)
            {
                foreach (var n in Layers[i])
                {
                    n.BackError(0.02);
                }
            }
        }

        /// <summary>
        /// Обучение сети одному образу
        /// </summary>
        /// <param name="sample"></param>
        /// <returns>Количество итераций для достижения заданного уровня ошибки</returns>
        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            int cnt = 0;
            
            while (cnt < 50)
            {
                Forward(sample);
                if (sample.EstimatedError() < 0.1 && sample.Correct())
                {
                    return cnt;
                }

                cnt++;
                Backward(sample);
            }

            System.Diagnostics.Debug.WriteLine("Выход по числу итераций");
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
            var t = System.DateTime.Now;
            //точность
            double accuracy = 0;
            //рассмотренные образы
            int samplesLooked = 0; 
            //всевозможные элементы для рассмотрения
            double allSamples = samplesSet.samples.Count * epochsCount; 

            while (epochsCount-- > 0)
            {
                //колво правильно обученного
                double rightCnt = 0;
                
                foreach (var sample in samplesSet.samples)
                {
                    //если мы правильно предсказали
                    if (Train(sample, 0.0005, true) < 50) 
                        rightCnt++;
                    samplesLooked++;
                    //обновляем прогресс
                    if (samplesLooked % 10 == 0) 
                        OnTrainProgress(samplesLooked / allSamples, accuracy, DateTime.Now - t);
                }
                
                //точность, которую мы передаем потом в форму
                accuracy = rightCnt / samplesSet.samples.Count; 
                if (accuracy >= 1 - acceptableError - 1e-10) {
                    OnTrainProgress(1, accuracy, DateTime.Now - t);
                    return accuracy;
                }
                else OnTrainProgress(samplesLooked / allSamples, accuracy, DateTime.Now - t);
            }

            OnTrainProgress(1, accuracy, DateTime.Now - t);
            return accuracy;
        }
        
        /// <summary>
        /// Запустить сеть на примере и вернуть распознанный класс
        /// </summary>
        public override FigureType Predict(Sample sample)
        {
            Forward(sample);
            return sample.recognizedClass;
        }

        protected override double[] Compute(double[] input)
        {
            //по образу заполняем сенсоры
            for (int i = 0; i < input.Length; i++)
            {
                Layers[0][i].output = input[i];
            }
            
            //проходим по всем слоям (кроме 0) и выполняем нужные вычисления
            for (int i = 1; i < Layers.Count; i++)
            {
                for (int j = 0; j < Layers[i].Count; j++)
                {
                    Layers[i][j].funcActivate();
                }
            }

            //передаем в образ результат распознавания
            double[] temp = new double[Layers.Last().Count];
            for (int i = 0; i < Layers[Layers.Count-1].Count; i++)
            {
                temp[i] = Layers[Layers.Count - 1][i].output;
            }

            return temp;
        }
    }
}