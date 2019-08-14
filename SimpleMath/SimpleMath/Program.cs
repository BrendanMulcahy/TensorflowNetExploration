using System;
using NumSharp;
using Tensorflow;

namespace SimpleMath
{
    public class Program
    {
        public static void Main(string[] args)
        {
            Tensor x = tf.Variable(3, name: "x");
            Tensor y = tf.Variable(4, name: "y");
            Tensor f = x * x * y + y + 2;

            Operation init = tf.global_variables_initializer();

            using (Session session = tf.Session())
            {
                session.run(init);
                NDArray result = f.eval();
                Console.WriteLine(result.ToString());
            }
        }
    }
}