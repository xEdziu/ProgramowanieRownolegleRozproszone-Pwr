using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace ABCAlgorithm
{
    public class TaskItem
    {
        public int[] MachineTimes { get; set; }
        public TaskItem(int[] machineTimes) { MachineTimes = machineTimes; }
        public TaskItem DeepCopy() { return new TaskItem((int[])MachineTimes.Clone()); }
        public override string ToString() { return string.Join(", ", MachineTimes); }
    }

    class Program
    {
        private static readonly object lockObject = new object();
        private static Random threadSafeRandom = new Random();
        private static Context gpuContext;
        private static Accelerator gpuAccelerator;

        // GPU Kernel do obliczania Cmax
        static void CalculateCmaxKernel(
            Index1D index,
            ArrayView<int> allTasksFlat,
            ArrayView<int> results,
            ArrayView<int> allCValues,
            int n,
            int m)
        {
            int taskSetIndex = index;

            // offset w spłaszczonych tablicach
            int taskOffset = taskSetIndex * n * m;
            int cOffset    = taskSetIndex * n * m;

            // "Widok" na fragment cValues tylko dla tego rozwiązania
            var cValues = allCValues.SubView(cOffset, n * m);

            // LICZENIE Cmax bez lokalnych tablic
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < m; j++)
                {
                    int idx = i * m + j;

                    // C[i-1, j]
                    int above = i > 0 ? cValues[(i - 1) * m + j] : 0;
                    // C[i, j-1]
                    int left  = j > 0 ? cValues[i * m + (j - 1)] : 0;

                    int start;
                    if (i == 0 && j == 0)
                        start = 0;
                    else
                        start = above > left ? above : left;

                    int taskTime = allTasksFlat[taskOffset + idx];
                    cValues[idx] = start + taskTime;
                }
            }

            // Cmax = ostatni element
            results[taskSetIndex] = cValues[n * m - 1];
        }


        static void InitializeGPU()
        {
            try
            {
                gpuContext = Context.Create(builder => builder.Cuda().EnableAlgorithms());
                gpuAccelerator = gpuContext.GetPreferredDevice(preferCPU: false).CreateAccelerator(gpuContext);
                Console.WriteLine($"GPU zainicjalizowane: {gpuAccelerator.Name}");
                Console.WriteLine($"Pamięć: {gpuAccelerator.MemorySize / (1024 * 1024)} MB");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Nie można zainicjalizować GPU: {ex.Message}");
                Console.WriteLine("Program będzie działał tylko na CPU.");
                gpuAccelerator = null;
            }
        }

        static void CleanupGPU()
        {
            gpuAccelerator?.Dispose();
            gpuContext?.Dispose();
        }

        static List<TaskItem> GeneratingTasks(int n, int m, long seed)
        {
            RandomNumberGenerator rng = new RandomNumberGenerator(seed);
            List<TaskItem> tasks = new List<TaskItem>();
            for (int i = 0; i < n; i++)
            {
                int[] machineTimes = new int[m];
                for (int j = 0; j < m; j++) machineTimes[j] = rng.NextInt(1, 29);
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
                    if (i == 0 && j == 0) sValues[i, j] = 0;
                    else if (i == 0) sValues[i, j] = cValues[i, j - 1];
                    else if (j == 0) sValues[i, j] = cValues[i - 1, j];
                    else sValues[i, j] = Math.Max(cValues[i - 1, j], cValues[i, j - 1]);
                    cValues[i, j] = sValues[i, j] + tasks[i].MachineTimes[j];
                }
            }
            return cValues[n - 1, m - 1];
        }

        static int[] CalculateCmaxBatchGPU(List<List<TaskItem>> taskSets, int n, int m)
        {
            if (gpuAccelerator == null)
                return taskSets.Select(ts => CalculateCmax(ts, n, m)).ToArray();

            int batchSize = taskSets.Count;
            int totalSize = batchSize * n * m;

            int[] flatData = new int[totalSize];
            for (int b = 0; b < batchSize; b++)
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < m; j++)
                        flatData[b * n * m + i * m + j] = taskSets[b][i].MachineTimes[j];

            using var inputBuffer  = gpuAccelerator.Allocate1D<int>(flatData);
            using var outputBuffer = gpuAccelerator.Allocate1D<int>(batchSize);
            using var cValuesBuf   = gpuAccelerator.Allocate1D<int>(totalSize);

            var kernel = gpuAccelerator
                .LoadAutoGroupedStreamKernel<
                    Index1D,
                    ArrayView<int>,
                    ArrayView<int>,
                    ArrayView<int>,
                    int,
                    int>(CalculateCmaxKernel);

            kernel(batchSize, inputBuffer.View, outputBuffer.View, cValuesBuf.View, n, m);
            gpuAccelerator.Synchronize();

            return outputBuffer.GetAsArray1D();
        }


        static double CalculateFitness(List<TaskItem> order, int n, int m) { return 1.0 / (1.0 + CalculateCmax(order, n, m)); }

        static List<TaskItem> GetNeighbor(List<TaskItem> solution, int n)
        {
            List<TaskItem> neighbor = solution.Select(t => t.DeepCopy()).ToList();
            int i, j;
            lock (lockObject)
            {
                i = threadSafeRandom.Next(n);
                j = threadSafeRandom.Next(n);
                while (i == j) j = threadSafeRandom.Next(n);
            }
            TaskItem temp = neighbor[i]; neighbor[i] = neighbor[j]; neighbor[j] = temp;
            return neighbor;
        }

        static int RouletteSelection(double[] fitness, double totalFitness)
        {
            double pick;
            lock (lockObject) { pick = threadSafeRandom.NextDouble() * totalFitness; }
            double current = 0;
            for (int i = 0; i < fitness.Length; i++)
            {
                current += fitness[i];
                if (current > pick) return i;
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
                    TaskItem temp = shuffled[i]; shuffled[i] = shuffled[j]; shuffled[j] = temp;
                }
            }
            return shuffled;
        }

        static (List<TaskItem>, int) ArtificialBeeColonyGPU(List<TaskItem> tasks, int n, int m, int colonySize = 30, int limit = 100, int maxIter = 500)
        {
            List<List<TaskItem>> foodSources = new List<List<TaskItem>>();
            double[] fitness = new double[colonySize];
            int[] trialCounters = new int[colonySize];

            object initLock = new object();
            Parallel.For(0, colonySize, i =>
            {
                List<TaskItem> solution = ShuffleTasks(tasks);
                double fit = CalculateFitness(solution, n, m);
                lock (initLock) { foodSources.Add(solution); fitness[i] = fit; trialCounters[i] = 0; }
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
                        lock (lockObject) { foodSources[i] = neighbor; fitness[i] = neighborFitness; trialCounters[i] = 0; }
                    }
                    else { lock (lockObject) { trialCounters[i]++; } }
                });

                double totalFitness = fitness.Sum();
                Parallel.For(0, colonySize, _ =>
                {
                    int selectedIndex = RouletteSelection(fitness, totalFitness);
                    List<TaskItem> neighbor = GetNeighbor(foodSources[selectedIndex], n);
                    double neighborFitness = CalculateFitness(neighbor, n, m);
                    if (neighborFitness > fitness[selectedIndex])
                    {
                        lock (lockObject) { foodSources[selectedIndex] = neighbor; fitness[selectedIndex] = neighborFitness; trialCounters[selectedIndex] = 0; }
                    }
                    else { lock (lockObject) { trialCounters[selectedIndex]++; } }
                });

                Parallel.For(0, colonySize, i =>
                {
                    if (trialCounters[i] > limit)
                    {
                        List<TaskItem> newSolution = ShuffleTasks(tasks);
                        double newFitness = CalculateFitness(newSolution, n, m);
                        lock (lockObject) { foodSources[i] = newSolution; fitness[i] = newFitness; trialCounters[i] = 0; }
                    }
                });

                // GPU batch processing co 10 iteracji
                if (iteration % 10 == 0 && gpuAccelerator != null)
                {
                    int[] cmaxValues = CalculateCmaxBatchGPU(foodSources, n, m);
                    for (int i = 0; i < colonySize; i++)
                    {
                        if (cmaxValues[i] < bestCmax)
                        {
                            bestCmax = cmaxValues[i];
                            bestSolution = foodSources[i].Select(t => t.DeepCopy()).ToList();
                        }
                    }
                }
                else
                {
                    object bestLock = new object();
                    Parallel.For(0, colonySize, i =>
                    {
                        int currentCmax = CalculateCmax(foodSources[i], n, m);
                        lock (bestLock)
                        {
                            if (currentCmax < bestCmax) { bestCmax = currentCmax; bestSolution = foodSources[i].Select(t => t.DeepCopy()).ToList(); }
                        }
                    });
                }
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
                if (currentCmax < bestCmax) { bestCmax = currentCmax; bestOrder = perm.Select(t => t.DeepCopy()).ToList(); }
            }
            return (bestOrder, bestCmax);
        }

        static IEnumerable<List<TaskItem>> GetPermutations(List<TaskItem> list, int length)
        {
            if (length == 1) yield return list;
            else
            {
                for (int i = 0; i < length; i++)
                {
                    foreach (var perm in GetPermutations(list, length - 1)) yield return perm;
                    if (length % 2 == 0) { TaskItem temp = list[i]; list[i] = list[length - 1]; list[length - 1] = temp; }
                    else { TaskItem temp = list[0]; list[0] = list[length - 1]; list[length - 1] = temp; }
                }
            }
        }

        static void Main(string[] args)
        {
            Console.WriteLine("=== Inicjalizacja GPU ===");
            InitializeGPU();
            
            int n = 10, m = 5;
            long seed = 123456;

            Console.WriteLine($"\nGenerowanie zadań dla n = {n}, m = {m}, seed = {seed}");
            List<TaskItem> tasks = GeneratingTasks(n, m, seed);

            Console.WriteLine("\nWygenerowane zadania:");
            for (int i = 0; i < tasks.Count; i++) Console.WriteLine($"Zadanie {i + 1}: {tasks[i]}");

            Console.WriteLine($"\nStart problemu przepływowego dla n = {n}, m = {m}, seed = {seed}");

            Stopwatch swAbc = Stopwatch.StartNew();
            var (bestOrderAbc, bestCmaxAbc) = ArtificialBeeColonyGPU(tasks.Select(t => t.DeepCopy()).ToList(), n, m);
            swAbc.Stop();

            Console.WriteLine($"\n[ABC - CPU + GPU] Najlepszy Cmax: {bestCmaxAbc}");
            Console.WriteLine($"[ABC - CPU + GPU] Czas: {swAbc.Elapsed.TotalSeconds:F4}s");

            Stopwatch swBf = Stopwatch.StartNew();
            var (bestOrderBf, bestCmaxBf) = BruteForce(tasks.Select(t => t.DeepCopy()).ToList(), n, m);
            swBf.Stop();

            Console.WriteLine($"\n[Brute Force] Optymalny Cmax: {bestCmaxBf}");
            Console.WriteLine($"[Brute Force] Czas: {swBf.Elapsed.TotalSeconds:F4}s");

            Console.WriteLine("\nPorównanie:");
            if (bestCmaxAbc == bestCmaxBf) Console.WriteLine("✓ ABC znalazł optymalne rozwiązanie.");
            else Console.WriteLine($"✗ Różnica: {bestCmaxAbc - bestCmaxBf}");

            Console.WriteLine("\nABC kolejność:");
            for (int i = 0; i < bestOrderAbc.Count; i++) Console.WriteLine($"Zadanie {i + 1}: {bestOrderAbc[i]}");

            Console.WriteLine("\nBrute Force kolejność:");
            for (int i = 0; i < bestOrderBf.Count; i++) Console.WriteLine($"Zadanie {i + 1}: {bestOrderBf[i]}");

            Console.WriteLine("\n=== Czyszczenie GPU ===");
            CleanupGPU();
            Console.WriteLine("\nKoniec programu.");
        }
    }
}
