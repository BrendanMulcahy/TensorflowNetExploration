using System;
using NumSharp;
using Tensorflow;

namespace HelloWorld
{
    public class Program
    {
        public static void Main(string[] args)
        {
            Tensor hello = tf.constant("Hello, world!");

            using (Session session = tf.Session())
            {
                NDArray result = session.run(hello);
                Console.WriteLine(result.ToString());
            }
        }
    }
}