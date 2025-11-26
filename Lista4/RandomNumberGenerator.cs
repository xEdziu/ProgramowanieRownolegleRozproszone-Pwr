using System;

namespace ABCAlgorithm
{
    public class RandomNumberGenerator
    {
        private long seed;

        public RandomNumberGenerator(long seedValue)
        {
            this.seed = seedValue;
        }

        public int NextInt(int low, int high)
        {
            long m = 2147483647;
            long a = 16807;
            long b = 127773;
            long c = 2836;

            long k = seed / b;
            seed = a * (seed % b) - k * c;

            if (seed < 0)
            {
                seed = seed + m;
            }

            double value_0_1 = (double)seed / m;
            return low + (int)Math.Floor(value_0_1 * (high - low + 1));
        }

        public double NextFloat(double low, double high)
        {
            long lowInt = (long)(low * 100000);
            long highInt = (long)(high * 100000);
            double val = NextInt((int)lowInt, (int)highInt) / 100000.0;
            return val;
        }
    }
}
