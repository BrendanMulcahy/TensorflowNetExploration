using System;
using NumSharp;
using Tensorflow;

namespace SimpleMath
{
    public class Program
    {
        public static void Main(string[] args)
        {
            RefVariable x = tf.Variable(3, name: "x");
            RefVariable y = tf.Variable(4, name: "y");
            Tensor f = (Tensor) x * x * y + y + 2;

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