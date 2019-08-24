using NumSharp;
using Tensorflow;
using Tensorflow.Operations;
using Tensorflow.Operations.Activation;

namespace Dnn
{
    public class Program
    {
        const int nInputs = 28 * 28; // MNIST image dimensions
        const float learningRate = 0.01f;
        const int sizeHidden1 = 300;
        const int sizeHidden2 = 100;
        const int numOutputs = 10;

        public static void Main(string[] args)
        {
            Tensor features = tf.placeholder(tf.float32, (-1, nInputs), "Features");
            Tensor labels = tf.placeholder(tf.int64, new TensorShape(-1), "Labels");

            var (trainingOp, loss) = MakeGraph(features, labels);

            Operation init = tf.global_variables_initializer();
            Saver saver = tf.train.Saver();

            var epochs = 20;
            var batches = 50;

            using (var session = tf.Session())
            {
                session.run(init);
                for (int i = 0; i < epochs; i++)
                {
                    for (int j = 0; j < batches; j++)
                    {
                        session.run(trainingOp); // todo feed data
                    }
                }
            }
        }

        public static (Operation, Tensor) MakeGraph(Tensor features, Tensor labels)
        {
            Tensor hidden1 = tf.layers.dense(features, sizeHidden1, name: "Hidden1", activation: tf.nn.relu());
            Tensor hidden2 = tf.layers.dense(hidden1, sizeHidden2, name: "Hidden2", activation: tf.nn.relu());
            Tensor logits = tf.layers.dense(hidden2, numOutputs, name: "outputs");
            Tensor label_probability = tf.nn.softmax(logits);

            Tensor xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits);
            Tensor loss = tf.reduce_mean(xentropy, name: "loss");

            Optimizer optimizer = tf.train.GradientDescentOptimizer(learningRate);
            Operation trainingOp = optimizer.minimize(loss);

            return (trainingOp, loss);
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