namespace Lista1
{
    internal class Program
    {
        static readonly int[] MinersCount = { 2, 3, 4, 5, 6 };
        static readonly (string name, int ore, int slots, int travelMs)[] SitesConfig = new[]
        {
            ("Ore A", 2000, 2, 10000),
            ("Ore B", 2600,  1,  16000),
        };
        const int WarehouseSlots = 2;
        const int VehicleCapacity = 200;
        const int TimeAquiringOneUnit = 10; //ms
        const int TimeUnloadingOneUnit = 10; //ms
        static readonly ThreadLocal<Random> _rng = new ThreadLocal<Random>(() => new Random());

        enum SelectionMode
        {
            Random,
            MostFreeSlots
        }

        class OreSite
        {
            public string Name { get; }
            public int TravelMs { get; }
            public SemaphoreSlim Mine { get; }
            public object Lock { get; } = new object();
            public int OreRemaining;

            public OreSite(string name, int ore, int slots, int travelMs)
            {
                Name = name;
                OreRemaining = ore;
                Mine = new SemaphoreSlim(slots, slots);
                TravelMs = travelMs;
            }
        }


        class SimulationResult
        {
            public int NumberOfMiners { get; set; }
            public int TotalTime { get; set; }
        }


        static void Main(string[] args)
        {
            Console.Clear();

            var resultsRandom = new List<SimulationResult>();
            var resultsMostSlots = new List<SimulationResult>();

            int totalStartTime = Environment.TickCount;

            int[] winsPerMode = { 0, 0 };
            foreach (int count in MinersCount)
            {
                var resultRandom = RunSimulation(count, SelectionMode.Random);
                var resultMostSlots = RunSimulation(count, SelectionMode.MostFreeSlots);

                resultsRandom.Add(resultRandom);
                resultsMostSlots.Add(resultMostSlots);

                if (resultRandom.TotalTime < resultMostSlots.TotalTime)
                {
                    winsPerMode[0]++;
                }
                else if (resultRandom.TotalTime > resultMostSlots.TotalTime)
                {
                    winsPerMode[1]++;
                }
            }
            int totalEndTime = Environment.TickCount;

            Console.WriteLine("\n=== SUMMARY ===");
            Console.WriteLine("\n--- Selection mode: Random ---");
            foreach (var r in resultsRandom)
            {
                double time = r.TotalTime;
                double speedup = resultsRandom[0].TotalTime / time; // speedup = T1 / Tp
                double efficiency = speedup / r.NumberOfMiners; // efficiency = speedup / p
                Console.WriteLine($"{r.NumberOfMiners} miners → time {r.TotalTime}s; acceleration: {speedup:F2}; efficiency: {efficiency:F2}");
            }
            Console.WriteLine("\n--- Selection mode: Most free slots ---");
            foreach (var r in resultsMostSlots)
            {
                double time = r.TotalTime;
                double speedup = resultsMostSlots[0].TotalTime / time; // speedup = T1 / Tp
                double efficiency = speedup / r.NumberOfMiners; // efficiency = speedup / p
                Console.WriteLine($"{r.NumberOfMiners} miners → time {r.TotalTime}s; acceleration: {speedup:F2}; efficiency: {efficiency:F2}");
            }

            Console.WriteLine("\nAmounts each mode performed better:");
            Console.WriteLine($"Random: {winsPerMode[0]}");
            Console.WriteLine($"Most free slots: {winsPerMode[1]}");
            Console.WriteLine($"\nTotal time it took to run the simulations: {(totalEndTime - totalStartTime) / 1000}s");

        }

        static SimulationResult RunSimulation(int NumberOfMiners, SelectionMode mode)
        {
            int Warehouse = 0;
            int minerId = 0;

            var lockWarehouse = new object();
            SemaphoreSlim warehouseSem = new SemaphoreSlim(WarehouseSlots, WarehouseSlots);

            var ores = SitesConfig.Select(cfg => new OreSite(cfg.name, cfg.ore, cfg.slots, cfg.travelMs)).ToList();

            object consoleLock = new object();
            string[] minerStatus = new string[NumberOfMiners];

            Task[] miners = new Task[NumberOfMiners];
            Console.CursorVisible = false;
            for (int i = 0; i < NumberOfMiners; i++)
            {
                minerStatus[i] = "Starting...";
            }
            var cts = new CancellationTokenSource();
            var displayTask = Task.Run(() => DisplayLoop(
                cts.Token,
                mode,
                warehouseSem,
                consoleLock,
                minerStatus,
                ores,
                () => Warehouse));

            // Start time
            int startTime = Environment.TickCount;
            for (int i = 0; i < NumberOfMiners; i++)
            {
                miners[i] = Task.Run(() =>
            DoMining(
                ores,
                mode,
                warehouseSem,
                lockWarehouse,
                minerStatus,
                ref Warehouse,
                ref minerId
            ));
            }
            Task.WaitAll(miners);
            // End time
            int endTime = Environment.TickCount;
            cts.Cancel();
            try { displayTask.Wait(); } catch { }

            return new SimulationResult
            {
                NumberOfMiners = NumberOfMiners,
                TotalTime = (endTime - startTime) / 1000
            };

        }

        static void DoMining(
            List<OreSite> ores,
            SelectionMode mode,
            SemaphoreSlim warehouseSem,
            object lockWarehouse,
            string[] minerStatus,
            ref int Warehouse,
            ref int minerId
        )
        {
            int idx = Interlocked.Increment(ref minerId) - 1;
            while (true)
            {
                // Choosing ore
                var ore = ChooseOre(ores, mode);
                if (ore == null)
                {
                    minerStatus[idx] = "Finished";
                    break;
                }
                // Accesing the mine
                minerStatus[idx] = $"Waiting for mine access for {ore.Name}";
                ore.Mine.Wait();
                // Mining
                minerStatus[idx] = $"Mining at {ore.Name}";
                int amountToMine;
                lock (ore.Lock)
                {
                    amountToMine = Math.Min(VehicleCapacity, ore.OreRemaining);
                    if (amountToMine == 0)
                    {
                        ore.Mine.Release();
                        continue;
                    }
                    for (int i = 0; i < amountToMine; i++)
                    {
                        Thread.Sleep(TimeAquiringOneUnit);
                        ore.OreRemaining--;
                    }
                }
                ore.Mine.Release();
                // Traveling to warehouse
                minerStatus[idx] = "Traveling to warehouse";
                Thread.Sleep(ore.TravelMs);
                // Unloading
                minerStatus[idx] = "Waiting for warehouse access";
                warehouseSem.Wait();
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
                warehouseSem.Release();
                // Traveling back to mine
                minerStatus[idx] = $"Traveling to {ore.Name}";
                Thread.Sleep(ore.TravelMs);
            }
        }

        static OreSite? ChooseOre(List<OreSite> ores, SelectionMode mode)
        {
            var candidates = ores.Where(o => Volatile.Read(ref o.OreRemaining) > 0).ToList();
            if (candidates.Count == 0) return null;

            switch (mode)
            {
                case SelectionMode.Random:
                    return candidates[_rng.Value!.Next(candidates.Count)];

                case SelectionMode.MostFreeSlots:
                    return candidates
                        .OrderByDescending(o => o.Mine.CurrentCount)
                        .ThenByDescending(o => o.OreRemaining)
                        .ThenBy(o => o.TravelMs)
                        .First();

                default:
                    return candidates[0];
            }
        }


        static void DisplayLoop(
            CancellationToken token,
            SelectionMode mode,
            SemaphoreSlim warehouseSem,
            object consoleLock,
            string[] minerStatus,
            List<OreSite> ores,
            Func<int> getWarehouse
            )
        {
            while (!token.IsCancellationRequested)
            {
                lock (consoleLock)
                {
                    Console.SetCursorPosition(0, 0);
                    Console.WriteLine("=== LIVE SIMULATION STATUS ===".PadRight(60));

                    int row = 2;
                    Console.SetCursorPosition(0, row++);
                    Console.WriteLine($"--- Selection mode: {mode} ---".PadRight(60));

                    foreach (var o in ores)
                    {
                        Console.SetCursorPosition(0, row++);
                        Console.WriteLine($"{o.Name}: ore={Volatile.Read(ref o.OreRemaining)} | mine free slots={o.Mine.CurrentCount} | travel={o.TravelMs}ms".PadRight(80));
                    }

                    Console.SetCursorPosition(0, row++);
                    Console.WriteLine($"Warehouse: {getWarehouse()} units (free slots: {warehouseSem.CurrentCount})".PadRight(80));

                    Console.SetCursorPosition(0, row++);
                    Console.WriteLine(new string('-', 64));

                    for (int i = 0; i < minerStatus.Length; i++)
                    {
                        Console.SetCursorPosition(0, row++);
                        var st = minerStatus[i] ?? "";
                        Console.WriteLine($"Miner {i + 1}: {st}".PadRight(80));
                    }
                }
                Thread.Sleep(100);
            }
        }
    }
}
