using NumSharp;
using Tensorflow;
using Tensorflow.Operations;
using Tensorflow.Operations.Activation;

namespace Dnn
{
    public class Program
    {
        public static void Main(string[] args)
        {
            int nInputs = 28 * 28; // MNIST image dimensions
            var nHidden1 = 300;
            var nHidden2 = 100;
            var nOutputs = 10;

            Tensor X = tf.placeholder(tf.float32, (-1, nInputs), "X");
            Tensor y = tf.placeholder(tf.int64, new TensorShape(-1), "y");

            using (tf.name_scope("dnn"))
            {
                Tensor hidden1 = tf.layers.dense(X, nHidden1, name: "hidden1");
                Tensor hidden2 = tf.layers.dense(hidden1, nHidden2, name: "hidden2", activation: tf.nn.relu());
                Tensor logits = tf.layers.dense(hidden2, nOutputs, name: "outputs");

                using (tf.name_scope("loss"))
                {
                    Tensor xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y, logits);
                    Tensor loss = tf.reduce_mean(xentropy, name: "loss");


                    const float learningRate = 0.01f;

                    using (tf.name_scope("train"))
                    {
                        Optimizer optimizer = tf.train.GradientDescentOptimizer(learningRate);
                        Operation trainingOp = optimizer.minimize(loss);

                        using (tf.name_scope("eval")) // this nesting is getting crazy (?)
                        {
                            Tensor correct = gen_ops.in_top_k(logits, y, 1);
                            Tensor accuracy = tf.reduce_mean(tf.cast(correct, tf.float32));

                            var init = tf.global_variables_initializer();
                            var saver = tf.train.Saver();
                        }
                    }
                }
            }
        }

        // Example of how to create a neuron layer from scratch, use tf.layers.dense instead
        public static Tensor NeuronLayer(Tensor X, int nNeurons, string name, IActivation activation = null)
        {
            using (tf.name_scope(name))
            {
                int nInputs = X.shape[1];
                NDArray stddev = 2 / np.sqrt(nInputs);
                Tensor init = tf.truncated_normal(new[] {nInputs, nNeurons}, stddev: stddev);
                RefVariable W = tf.Variable(init, name: "kernel");
                RefVariable b = tf.Variable(tf.zeros(new[] {nNeurons}), name: "bias");
                Tensor Z = tf.matmul(X, W) + b;

                if (activation != null)
                {
                    return activation.Activate(Z);
                }

                return Z;
            }
        }
    }
}