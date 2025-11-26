using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace ABCAlgorithm
{
    public class TaskItem
    {
        public int[] MachineTimes { get; set; }

        public TaskItem(int[] machineTimes)
        {
            MachineTimes = machineTimes;
        }

        public TaskItem DeepCopy()
        {
            return new TaskItem((int[])MachineTimes.Clone());
        }

        public override string ToString()
        {
            return string.Join(", ", MachineTimes);
        }
    }

    class Program
    {
        private static readonly object lockObject = new object();
        private static Random threadSafeRandom = new Random();

        static List<TaskItem> GeneratingTasks(int n, int m, long seed)
        {
            RandomNumberGenerator rng = new RandomNumberGenerator(seed);
            List<TaskItem> tasks = new List<TaskItem>();

            for (int i = 0; i < n; i++)
            {
                int[] machineTimes = new int[m];
                for (int j = 0; j < m; j++)
                {
                    machineTimes[j] = rng.NextInt(1, 29);
                }
                tasks.Add(new TaskItem(machineTimes));
            }

            return tasks;
        }

        static int CalculateCmax(List<TaskItem> tasks, int n, int m)
        {
            int[,] sValues = new int[n, m];
            int[,] cValues = new int[n, m];

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < m; j++)
                {
                    if (i == 0 && j == 0)
                    {
                        sValues[i, j] = 0;
                    }
                    else if (i == 0)
                    {
                        sValues[i, j] = cValues[i, j - 1];
                    }
                    else if (j == 0)
                    {
                        sValues[i, j] = cValues[i - 1, j];
                    }
                    else
                    {
                        sValues[i, j] = Math.Max(cValues[i - 1, j], cValues[i, j - 1]);
                    }
                    cValues[i, j] = sValues[i, j] + tasks[i].MachineTimes[j];
                }
            }

            return cValues[n - 1, m - 1];
        }

        static double CalculateFitness(List<TaskItem> order, int n, int m)
        {
            return 1.0 / (1.0 + CalculateCmax(order, n, m));
        }

        static List<TaskItem> GetNeighbor(List<TaskItem> solution, int n)
        {
            List<TaskItem> neighbor = solution.Select(t => t.DeepCopy()).ToList();

            int i, j;
            lock (lockObject)
            {
                i = threadSafeRandom.Next(n);
                j = threadSafeRandom.Next(n);
                while (i == j)
                {
                    j = threadSafeRandom.Next(n);
                }
            }

            TaskItem temp = neighbor[i];
            neighbor[i] = neighbor[j];
            neighbor[j] = temp;

            return neighbor;
        }

        static int RouletteSelection(double[] fitness, double totalFitness)
        {
            double pick;
            lock (lockObject)
            {
                pick = threadSafeRandom.NextDouble() * totalFitness;
            }

            double current = 0;
            for (int i = 0; i < fitness.Length; i++)
            {
                current += fitness[i];
                if (current > pick)
                {
                    return i;
                }
            }
            return fitness.Length - 1;
        }

        static List<TaskItem> ShuffleTasks(List<TaskItem> tasks)
        {
            List<TaskItem> shuffled = tasks.Select(t => t.DeepCopy()).ToList();
            lock (lockObject)
            {
                int n = shuffled.Count;
                for (int i = n - 1; i > 0; i--)
                {
                    int j = threadSafeRandom.Next(i + 1);
                    TaskItem temp = shuffled[i];
                    shuffled[i] = shuffled[j];
                    shuffled[j] = temp;
                }
            }
            return shuffled;
        }

        static (List<TaskItem>, int) ArtificialBeeColony(
            List<TaskItem> tasks,
            int n,
            int m,
            int colonySize = 30,
            int limit = 100,
            int maxIter = 500)
        {
            
            List<List<TaskItem>> foodSources = new List<List<TaskItem>>();
            double[] fitness = new double[colonySize];
            int[] trialCounters = new int[colonySize];

            object initLock = new object();
            Parallel.For(0, colonySize, i =>
            {
                List<TaskItem> solution = ShuffleTasks(tasks);
                double fit = CalculateFitness(solution, n, m);

                lock (initLock)
                {
                    foodSources.Add(solution);
                    fitness[i] = fit;
                    trialCounters[i] = 0;
                }
            });

            List<TaskItem> bestSolution = foodSources[0].Select(t => t.DeepCopy()).ToList();
            int bestCmax = CalculateCmax(bestSolution, n, m);

            for (int iteration = 0; iteration < maxIter; iteration++)
            {
                Parallel.For(0, colonySize, i =>
                {
                    List<TaskItem> neighbor = GetNeighbor(foodSources[i], n);
                    double neighborFitness = CalculateFitness(neighbor, n, m);

                    if (neighborFitness > fitness[i])
                    {
                        lock (lockObject)
                        {
                            foodSources[i] = neighbor;
                            fitness[i] = neighborFitness;
                            trialCounters[i] = 0;
                        }
                    }
                    else
                    {
                        lock (lockObject)
                        {
                            trialCounters[i]++;
                        }
                    }
                });

                double totalFitness = fitness.Sum();

                Parallel.For(0, colonySize, _ =>
                {
                    int selectedIndex = RouletteSelection(fitness, totalFitness);
                    List<TaskItem> neighbor = GetNeighbor(foodSources[selectedIndex], n);
                    double neighborFitness = CalculateFitness(neighbor, n, m);

                    if (neighborFitness > fitness[selectedIndex])
                    {
                        lock (lockObject)
                        {
                            foodSources[selectedIndex] = neighbor;
                            fitness[selectedIndex] = neighborFitness;
                            trialCounters[selectedIndex] = 0;
                        }
                    }
                    else
                    {
                        lock (lockObject)
                        {
                            trialCounters[selectedIndex]++;
                        }
                    }
                });

                Parallel.For(0, colonySize, i =>
                {
                    if (trialCounters[i] > limit)
                    {
                        List<TaskItem> newSolution = ShuffleTasks(tasks);
                        double newFitness = CalculateFitness(newSolution, n, m);

                        lock (lockObject)
                        {
                            foodSources[i] = newSolution;
                            fitness[i] = newFitness;
                            trialCounters[i] = 0;
                        }
                    }
                });

                object bestLock = new object();
                Parallel.For(0, colonySize, i =>
                {
                    int currentCmax = CalculateCmax(foodSources[i], n, m);
                    lock (bestLock)
                    {
                        if (currentCmax < bestCmax)
                        {
                            bestCmax = currentCmax;
                            bestSolution = foodSources[i].Select(t => t.DeepCopy()).ToList();
                        }
                    }
                });
            }

            return (bestSolution, bestCmax);
        }

        static (List<TaskItem>, int) BruteForce(List<TaskItem> tasks, int n, int m)
        {
            List<TaskItem> bestOrder = null;
            int bestCmax = int.MaxValue;

            foreach (var perm in GetPermutations(tasks, n))
            {
                int currentCmax = CalculateCmax(perm, n, m);
                if (currentCmax < bestCmax)
                {
                    bestCmax = currentCmax;
                    bestOrder = perm.Select(t => t.DeepCopy()).ToList();
                }
            }

            return (bestOrder, bestCmax);
        }

        static IEnumerable<List<TaskItem>> GetPermutations(List<TaskItem> list, int length)
        {
            if (length == 1)
            {
                yield return list;
            }
            else
            {
                for (int i = 0; i < length; i++)
                {
                    foreach (var perm in GetPermutations(list, length - 1))
                    {
                        yield return perm;
                    }

                    // Rotate
                    if (length % 2 == 0)
                    {
                        TaskItem temp = list[i];
                        list[i] = list[length - 1];
                        list[length - 1] = temp;
                    }
                    else
                    {
                        TaskItem temp = list[0];
                        list[0] = list[length - 1];
                        list[length - 1] = temp;
                    }
                }
            }
        }

        static void Main(string[] args)
        {
            int n = 10;
            int m = 5;
            long seed = 123456;

            Console.WriteLine($"Generowanie zadań dla n = {n}, m = {m}, seed = {seed}");
            List<TaskItem> tasks = GeneratingTasks(n, m, seed);

            Console.WriteLine("\nWygenerowane zadania:");
            for (int i = 0; i < tasks.Count; i++)
            {
                Console.WriteLine($"Zadanie {i + 1}: {tasks[i]}");
            }

            Console.WriteLine($"\nStart problemu przepływowego dla n = {n}, m = {m}, seed = {seed}");

            // ABC z równoległością
            Stopwatch swAbc = Stopwatch.StartNew();
            var (bestOrderAbc, bestCmaxAbc) = ArtificialBeeColony(
                tasks.Select(t => t.DeepCopy()).ToList(), n, m);
            swAbc.Stop();

            Console.WriteLine($"\n[ABC - RÓWNOLEGŁY] Najlepszy znaleziony Cmax: {bestCmaxAbc}");
            Console.WriteLine($"[ABC - RÓWNOLEGŁY] Czas działania: {swAbc.Elapsed.TotalSeconds:F4} sekundy");

            // Brute Force
            Stopwatch swBf = Stopwatch.StartNew();
            var (bestOrderBf, bestCmaxBf) = BruteForce(
                tasks.Select(t => t.DeepCopy()).ToList(), n, m);
            swBf.Stop();

            Console.WriteLine($"\n[Brute Force] Optymalny Cmax: {bestCmaxBf}");
            Console.WriteLine($"[Brute Force] Czas działania: {swBf.Elapsed.TotalSeconds:F4} sekundy");

            Console.WriteLine("\nPorównanie wyników:");
            if (bestCmaxAbc == bestCmaxBf)
            {
                Console.WriteLine("ABC znalazł optymalne rozwiązanie.");
            }
            else
            {
                Console.WriteLine($"ABC NIE znalazł optymalnego rozwiązania. Różnica: {bestCmaxAbc - bestCmaxBf}");
            }

            Console.WriteLine("\nKolejność zadań wg ABC:");
            for (int i = 0; i < bestOrderAbc.Count; i++)
            {
                Console.WriteLine($"Zadanie {i + 1}: {bestOrderAbc[i]}");
            }

            Console.WriteLine("\nKolejność zadań wg Brute Force:");
            for (int i = 0; i < bestOrderBf.Count; i++)
            {
                Console.WriteLine($"Zadanie {i + 1}: {bestOrderBf[i]}");
            }

            Console.WriteLine("\nKoniec programu.");
            Console.ReadKey();
        }
    }
}
