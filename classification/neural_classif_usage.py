import tensorflow as tf


class Classification:

    def __init__(self, model_path, file):
        # Network Parameters
        con_n_hidden_1 = 20  # 1st layer number of features
        con_n_hidden_2 = 20  # 2nd layer number of features
        qos_n_hidden_1 = 50  # 1st layer number of features
        qos_n_hidden_2 = 50  # 2nd layer number of features
        n_input = 4 # data input
        n_classes = 1 # function output

        # tf Graph input
        self.x = tf.placeholder("float", [None, n_input])

        # Create model
        def multilayer_perceptron(x, weights, biases):
            # Hidden layer with RELU activation
            layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
            layer_1 = tf.nn.relu(layer_1)
            # Hidden layer with RELU activation
            layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
            layer_2 = tf.nn.relu(layer_2)
            # Output layer with linear activation
            out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
            return out_layer

        # Store layers weight & bias
        con_weights = {
            'h1': tf.Variable(tf.random_normal([n_input, con_n_hidden_1]), name='h1c'),
            'h2': tf.Variable(tf.random_normal([con_n_hidden_1, con_n_hidden_2]), name='h2c'),
            'out': tf.Variable(tf.random_normal([con_n_hidden_2, n_classes]), name='houtc')
        }

        qos_weights = {
            'h1': tf.Variable(tf.random_normal([n_input, qos_n_hidden_1]), name='h1q'),
            'h2': tf.Variable(tf.random_normal([qos_n_hidden_1, qos_n_hidden_2]), name='h2q'),
            'out': tf.Variable(tf.random_normal([qos_n_hidden_2, n_classes]), name='houtq')
        }
        con_biases = {
            'b1': tf.Variable(tf.random_normal([con_n_hidden_1]), name='b1c'),
            'b2': tf.Variable(tf.random_normal([con_n_hidden_2]), name='b2c'),
            'out': tf.Variable(tf.random_normal([n_classes]), name='boutc')
        }
        qos_biases = {
            'b1': tf.Variable(tf.random_normal([qos_n_hidden_1]), name='b1q'),
            'b2': tf.Variable(tf.random_normal([qos_n_hidden_2]), name='b2q'),
            'out': tf.Variable(tf.random_normal([n_classes]), name='boutq')
        }




        # Construct model
        self.pred_con = multilayer_perceptron(self.x, con_weights, con_biases)
        self.pred_qos = multilayer_perceptron(self.x, qos_weights, qos_biases)

        saver = tf.train.Saver()
        self.sess = tf.Session()

        saver.restore(self.sess, model_path+file+'.ckpt')

    def getConsumption(self, val):
        return self.sess.run([self.pred_con], feed_dict={self.x: val})[0]

    def getQoS(self, val):
        return self.sess.run([self.pred_qos], feed_dict={self.x: val})[0]

    def closeModel(self):
        self.sess.close()

