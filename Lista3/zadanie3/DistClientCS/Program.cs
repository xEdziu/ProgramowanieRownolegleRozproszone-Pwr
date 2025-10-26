using System;
using System.Buffers.Binary;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;

// Klient kompatybilny z protokołem: [4B length BE][JSON payload].
// JSON niesie tablicę 3D: H x W x 3 (uint8), tak jak w Pythonie: np.array(image_fragment).tolist()

class Program
{
    static int[][][] Deserialize3D(byte[] json)
    {
        // Jagged arrays dobrze współpracują z System.Text.Json
        var data = JsonSerializer.Deserialize<int[][][]>(json) ?? throw new Exception("Deserialization returned null");
        return data;
    }

    static byte[] Serialize3D(int[][][] data)
    {
        return JsonSerializer.SerializeToUtf8Bytes(data);
    }

    static byte[] ReadExactly(NetworkStream ns, int n)
    {
        byte[] buf = new byte[n];
        int offset = 0;
        while (offset < n)
        {
            int read = ns.Read(buf, offset, n - offset);
            if (read <= 0) throw new IOException("Socket closed while reading.");
            offset += read;
        }
        return buf;
    }

    static byte[] ReceiveMessage(NetworkStream ns)
    {
        // 4 bajty długości (big-endian), potem payload
        byte[] lenBytes = ReadExactly(ns, 4);
        int len = BinaryPrimitives.ReadInt32BigEndian(lenBytes);
        if (len < 0 || len > 1_500_000_000) throw new Exception("Invalid length.");
        return ReadExactly(ns, len);
    }

    static void SendMessage(NetworkStream ns, byte[] payload)
    {
        Span<byte> lenBytes = stackalloc byte[4];
        BinaryPrimitives.WriteInt32BigEndian(lenBytes, payload.Length);
        ns.Write(lenBytes);
        ns.Write(payload);
        ns.Flush();
    }

    static double[,] ToGray(int[][][] rgb)
    {
        int h = rgb.Length;
        int w = rgb[0].Length;
        var gray = new double[h, w];
        for (int i = 0; i < h; i++)
        {
            var row = rgb[i];
            for (int j = 0; j < w; j++)
            {
                var px = row[j]; // [R,G,B]
                // średnia jak w Pythonie: np.mean(axis=2)
                gray[i, j] = (px[0] + px[1] + px[2]) / 3.0;
            }
        }
        return gray;
    }

    static int Clip255(double v) => v < 0 ? 0 : (v > 255 ? 255 : (int)v);

    static int[][][] Sobel(int[][][] fragment)
    {
        int h = fragment.Length;
        int w = fragment[0].Length;
        var gray = ToGray(fragment);

        // Sobel
        int[,] Kx = { { 1, 0, -1 }, { 2, 0, -2 }, { 1, 0, -1 } };
        int[,] Ky = { { 1, 2, 1 }, { 0, 0, 0 }, { -1, -2, -1 } };

        var gx = new double[h, w];
        var gy = new double[h, w];

        for (int i = 1; i < h - 1; i++)
        {
            for (int j = 1; j < w - 1; j++)
            {
                double sx = 0, sy = 0;
                for (int ki = -1; ki <= 1; ki++)
                {
                    for (int kj = -1; kj <= 1; kj++)
                    {
                        double val = gray[i + ki, j + kj];
                        sx += Kx[ki + 1, kj + 1] * val;
                        sy += Ky[ki + 1, kj + 1] * val;
                    }
                }
                gx[i, j] = sx;
                gy[i, j] = sy;
            }
        }

        var output = new int[h][][];
        for (int i = 0; i < h; i++)
        {
            output[i] = new int[w][];
            for (int j = 0; j < w; j++)
            {
                double g = Math.Sqrt(gx[i, j] * gx[i, j] + gy[i, j] * gy[i, j]);
                int v = Clip255(g);
                // zwracamy 3 kanały (R=G=B)
                output[i][j] = [v, v, v];
            }
        }
        return output;
    }

    static void Main(string[] args)
    {
        string host = args.Length > 0 ? args[0] : "127.0.0.1";
        int port = args.Length > 1 ? int.Parse(args[1]) : 2040;

        Console.WriteLine($"[C#] Connecting to {host}:{port} ...");
        using var client = new TcpClient();
        client.Connect(host, port);
        using NetworkStream ns = client.GetStream();

        // odbierz fragment
        byte[] request = ReceiveMessage(ns);
        Console.WriteLine($"[C#] Received fragment bytes: {request.Length}");

        // 2) JSON → tablica → Sobel
        var fragment = Deserialize3D(request);
        var processed = Sobel(fragment);

        // 3) wyślij wynik (ten sam format)
        byte[] payload = Serialize3D(processed);
        SendMessage(ns, payload);
        Console.WriteLine($"[C#] Sent processed fragment: {payload.Length} bytes");

        // koniec
        ns.Close();
        client.Close();
        Console.WriteLine("[C#] Done");
    }
}
