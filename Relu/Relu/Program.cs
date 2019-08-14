using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using Tensorflow;

namespace Relu
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            const int features = 3;
            Tensor x = tf.placeholder(tf.float32, new TensorShape(-1, features), "X");

            var relus = new List<Tensor>();
            for (var i = 0; i < 5; i++)
            {
                relus.Add(Relu(x));
            }

            Tensor output = relus.Aggregate(tf.add);

            Operation init = tf.global_variables_initializer();
            using (Session session = tf.Session())
            {
                session.run(init);

                NDArray nd = np.array(new[]
                {
                    new[] {3f, 1f, 1f},
                    new[] {2f, 3f, 1f}
                });

                NDArray result = output.eval(new FeedItem(x, nd));

                Console.WriteLine(result.ToString());
            }
        }

        public static Tensor Relu(Tensor x)
        {
            using (tf.name_scope("relu"))
            {
                int[] w_shape = {x.shape[1], 1};
                RefVariable w = tf.Variable(tf.random_normal(w_shape), name: "weights");
                RefVariable b = tf.Variable(0.0f, name: "bias");
                Tensor z = tf.add(tf.matmul(x, w), b);
                return tf.maximum(z, 0, "relu");
            }
        }
    }
}