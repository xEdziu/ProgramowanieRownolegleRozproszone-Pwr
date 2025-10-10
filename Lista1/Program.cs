namespace Lista1
{
    internal class Program
    {
        const int NumberOfMiners = 5;
        static int Ore1 = 2000;
        static int Magazine = 0;
        const int VehicleCapacity = 200;
        const int TimeAquiringOneUnit = 10; //ms
        const int TimeUnloadingOneUnit = 10; //ms
        const int TimeToTravelToDestination = 10000; //ms => 10s
        static SemaphoreSlim mine = new SemaphoreSlim(2, 2);
        static SemaphoreSlim magazine = new SemaphoreSlim(1, 1);
        static object lockMine = new object();
        static object lockMagazine = new object();
        static int minerId = 0;
        static readonly object consoleLock = new object(); 
        static readonly string[] minerStatus = new string[NumberOfMiners];


        static void Main(string[] args)
        {
            Task[] miners = new Task[NumberOfMiners];
            Console.CursorVisible = false;
            for (int i = 0; i < NumberOfMiners; i++)
            {
                minerStatus[i] = "Starting...";
            }
            var cts = new CancellationTokenSource();
            var displayTask = Task.Run(() => DisplayLoop(cts.Token));
            // Start time
            int startTime = Environment.TickCount;
            for (int i = 0; i < NumberOfMiners; i++)
            {
                miners[i] = Task.Run(() => DoMining());
            }
            Task.WaitAll(miners);
            // End time
            int endTime = Environment.TickCount;
            cts.Cancel();
            try { displayTask.Wait(); } catch { }
            Console.WriteLine("Resources left in Ore 1: " + Ore1);
            Console.WriteLine("All mining completed.");
            Console.WriteLine("Total time taken: " + (endTime - startTime)/1000 + "s");
            lock (consoleLock)
            {
                Console.SetCursorPosition(0, 7 + NumberOfMiners);
                Console.WriteLine($"Final Ore: {Ore1}".PadRight(60));
                Console.WriteLine($"Final Magazine: {Magazine}".PadRight(60));
                Console.CursorVisible = true;
            }
        }

        static void DoMining()
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
                // Traveling to magazine
                minerStatus[idx] = "Traveling to magazine";
                Thread.Sleep(TimeToTravelToDestination);
                // Unloading
                minerStatus[idx] = "Waiting for magazine access";
                magazine.Wait();
                minerStatus[idx] = "Unloading";
                lock (lockMagazine)
                {
                    for (int i = 0; i < amountToMine; i++)
                    {
                        Thread.Sleep(TimeUnloadingOneUnit);
                        Magazine++;
                    }
                    minerStatus[idx] = "Unloaded";
                }
                magazine.Release();
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

        static void DisplayLoop(CancellationToken token)
        {
            while (!token.IsCancellationRequested)
            {
                lock (consoleLock)
                {
                    Console.SetCursorPosition(0, 0);
                    Console.WriteLine("=== LIVE SIMULATION STATUS ===".PadRight(60));

                    Console.SetCursorPosition(0, 1);
                    Console.WriteLine($"Ore remaining: {Volatile.Read(ref Ore1)} units".PadRight(60));

                    Console.SetCursorPosition(0, 2);
                    Console.WriteLine($"Magazine: {Volatile.Read(ref Magazine)} units".PadRight(60));

                    Console.SetCursorPosition(0, 3);
                    Console.WriteLine($"Mine semaphore free slots: {mine.CurrentCount}".PadRight(60));

                    Console.SetCursorPosition(0, 4);
                    Console.WriteLine($"Magazine semaphore free slots: {magazine.CurrentCount}".PadRight(60));

                    Console.SetCursorPosition(0, 5);
                    Console.WriteLine(new string('-', 60));

                    for (int i = 0; i < NumberOfMiners; i++)
                    {
                        Console.SetCursorPosition(0, 6 + i);
                        var st = minerStatus[i] ?? "";
                        Console.WriteLine($"Miner {i + 1}: {st}".PadRight(60));
                    }
                }
                Thread.Sleep(50);
            }
        }
    }
}
