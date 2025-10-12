namespace Lista1
{
    internal class Program
    {
        static readonly int[] MinersCount = { 1, 2, 3, 4, 5, 6 };
        const int InitialOre = 2000;
        const int VehicleCapacity = 200;
        const int TimeAquiringOneUnit = 10; //ms
        const int TimeUnloadingOneUnit = 10; //ms
        const int TimeToTravelToDestination = 10000; //ms => 10s

        class SimulationResult
        {
            public int NumberOfMiners { get; set; }
            public int TotalTime { get; set; }
            public int FinalWarehouse { get; set; }
        }


        static void Main(string[] args)
        {
            Console.Clear();

            var results = new List<SimulationResult>();
            double initialTime = 0;

            foreach (int count in MinersCount)
            {
                var result = RunSimulation(count);
                results.Add(result);

                if (count == MinersCount[0])
                {
                    initialTime = result.TotalTime;
                }
            }

            Console.WriteLine("\n=== SUMMARY ===");
            foreach (var r in results)
            {
                double time = r.TotalTime;
                double speedup = initialTime / time; // speedup = T1 / Tp
                double efficiency = speedup / r.NumberOfMiners; // efficiency = speedup / p
                Console.WriteLine($"{r.NumberOfMiners} miners → time {r.TotalTime}s; acceleration: {speedup:F2}; efficiency: {efficiency:F2}");
            }

        }

        static SimulationResult RunSimulation(int NumberOfMiners)
        {
            int Ore1 = InitialOre;
            int Warehouse = 0;
            int minerId = 0;

            var lockMine = new object();
            var lockWarehouse = new object();

            SemaphoreSlim mine = new SemaphoreSlim(2, 2);
            SemaphoreSlim warehouse = new SemaphoreSlim(1, 1);

            object consoleLock = new object();
            string[] minerStatus = new string[NumberOfMiners];

            Task[] miners = new Task[NumberOfMiners];
            Console.CursorVisible = false;
            for (int i = 0; i < NumberOfMiners; i++)
            {
                minerStatus[i] = "Starting...";
            }
            var cts = new CancellationTokenSource();
            var displayTask = Task.Run(() => DisplayLoop(cts.Token, mine, warehouse, consoleLock, minerStatus, () => Ore1, () => Warehouse, NumberOfMiners));
            // Start time
            int startTime = Environment.TickCount;
            for (int i = 0; i < NumberOfMiners; i++)
            {
                miners[i] = Task.Run(() => DoMining(mine, warehouse, lockMine, lockWarehouse, minerStatus, ref Ore1, ref Warehouse, ref minerId));
            }
            Task.WaitAll(miners);
            // End time
            int endTime = Environment.TickCount;
            cts.Cancel();
            try { displayTask.Wait(); } catch { }

            return new SimulationResult
            {
                NumberOfMiners = NumberOfMiners,
                TotalTime = (endTime - startTime) / 1000,
                FinalWarehouse = Warehouse
            };

        }

        static void DoMining(
            SemaphoreSlim mine,
            SemaphoreSlim warehouse,
            object lockMine,
            object lockWarehouse,
            string[] minerStatus,
            ref int Ore1,
            ref int Warehouse,
            ref int minerId
        )
        {
            int idx = Interlocked.Increment(ref minerId) - 1;
            while (Ore1 > 0)
            {
                // Mining
                minerStatus[idx] = "Waiting for mine access";
                mine.Wait();
                minerStatus[idx] = "Mining";
                int amountToMine;
                lock (lockMine)
                {
                    amountToMine = Math.Min(VehicleCapacity, Ore1);
                    for (int i = 0; i < amountToMine; i++)
                    {
                        Thread.Sleep(TimeAquiringOneUnit);
                        Ore1--;
                    }
                }
                mine.Release();
                // Traveling to warehouse
                minerStatus[idx] = "Traveling to warehouse";
                Thread.Sleep(TimeToTravelToDestination);
                // Unloading
                minerStatus[idx] = "Waiting for warehouse access";
                warehouse.Wait();
                minerStatus[idx] = "Unloading";
                lock (lockWarehouse)
                {
                    for (int i = 0; i < amountToMine; i++)
                    {
                        Thread.Sleep(TimeUnloadingOneUnit);
                        Warehouse++;
                    }
                    minerStatus[idx] = "Unloaded";
                }
                warehouse.Release();
                // Don't continue if no ore left
                if (Ore1 <= 0)
                {
                    minerStatus[idx] = "Finished";
                    break;
                }
                // Traveling back to mine
                minerStatus[idx] = "Traveling back to mine";
                Thread.Sleep(TimeToTravelToDestination);
            }
        }

        static void DisplayLoop(
            CancellationToken token,
            SemaphoreSlim mine,
            SemaphoreSlim warehouse,
            object consoleLock,
            string[] minerStatus,
            Func<int> getOre1,
            Func<int> getWarehouse,
            int NumberOfMiners
            )
        {
            while (!token.IsCancellationRequested)
            {
                lock (consoleLock)
                {
                    Console.SetCursorPosition(0, 0);
                    Console.WriteLine("=== LIVE SIMULATION STATUS ===".PadRight(60));

                    Console.SetCursorPosition(0, 2);
                    Console.WriteLine($"Ore remaining: {getOre1()} units".PadRight(60));

                    Console.SetCursorPosition(0, 3);
                    Console.WriteLine($"Warehouse: {getWarehouse()} units".PadRight(60));

                    Console.SetCursorPosition(0, 4);
                    Console.WriteLine($"Mine semaphore free slots: {mine.CurrentCount}".PadRight(60));

                    Console.SetCursorPosition(0, 5);
                    Console.WriteLine($"Warehouse semaphore free slots: {warehouse.CurrentCount}".PadRight(60));

                    Console.SetCursorPosition(0, 6);
                    Console.WriteLine(new string('-', 60));

                    for (int i = 0; i < NumberOfMiners; i++)
                    {
                        Console.SetCursorPosition(0, 7 + i);
                        var st = minerStatus[i] ?? "";
                        Console.WriteLine($"Miner {i + 1}: {st}".PadRight(60));
                    }
                }
                Thread.Sleep(50);
            }
        }
    }
}
