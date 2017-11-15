import numpy as np
import tensorflow as tf

class Online_class_mab:

    def __init__(self, nPicos, ABSval, CREval, initExploration, epsilon_0, batch_size=1):

        nActivePicosVal = np.arange(0, (nPicos+1))
        ABSval = 0
        CREval = 0
        self.controlSpace = np.array(np.meshgrid(nActivePicosVal, ABSval, CREval)).T.reshape(-1, 3)
        self.nControls = len(self.controlSpace[:, 0])
        # self.nControls = 1

        # Network Parameters
        n_hidden_1 = 50  # 1st layer number of features
        n_hidden_2 = 50  # 2nd layer number of features
        n_input = 4 # data input
        n_output = 1 # function output
        learning_rate = 0.001
        self.batch_size = batch_size

        # self.batch_count = np.zeros((self.nControls))
        # self.batch_buffer = np.zeros((self.nControls, self.batch_size))

        self.initExploration = initExploration


        self.neuralArms = list()
        self.armCost = list()
        self.armOptimizer = list()

        self.x = tf.placeholder("float", [None, n_input])
        self.y = tf.placeholder("float", [None, n_output])

        def multilayer_perceptron(x, weights, biases):
            layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
            layer_1 = tf.nn.relu(layer_1)
            layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
            layer_2 = tf.nn.relu(layer_2)
            out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
            return out_layer

        weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name='h1_'),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='h2_'),
            'out': tf.Variable(tf.random_normal([n_hidden_2, n_output]), name='hout_')
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='b1_'),
            'b2': tf.Variable(tf.random_normal([n_hidden_2]), name='b2_'),
            'out': tf.Variable(tf.random_normal([n_output]), name='bout_')
        }
        self.pred = multilayer_perceptron(self.x, weights, biases)
        self.predict_class = tf.greater(self.pred, 0)

        self.cost_qos = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.pred, labels=self.y))
        self.optimizer_qos = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost_qos)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

        self.reward = np.zeros((self.nControls))
        self.count = np.zeros((self.nControls))
        self.epsilon_0 = epsilon_0

        self.algConf = {'epsilon_0': epsilon_0,
                   'initExploration': initExploration,
                    'batch_size': batch_size,
                    'n_hidden_1': n_hidden_1,
                    'n_hidden_2': n_hidden_2,
                    'learning_rate': learning_rate}


    def getControl(self, inputData):
        x = inputData['state']

        indexes = np.where(self.count < self.initExploration)[0]

        if len(indexes) > 0:
            array_index = np.random.randint(0, len(indexes))
            selectedIndex = indexes[array_index]
        else:
            states_controls = np.append(np.tile(x[0], (self.nControls, 1)), self.controlSpace.copy(), axis=1)
            QoS_class = self.sess.run([self.pred], feed_dict={self.x: states_controls})[0].transpose()[0]
            validActions = np.where(QoS_class > 0)[0]
            if len(validActions) == 0:
                validActions = np.array([np.argmax(QoS_class)])

            epsilon_desc = self.epsilon_0 / np.sum(self.count)
            if np.random.rand() < epsilon_desc:
                selectedIndex = np.random.randint(0, self.nControls, 1)[0]
            else:
                selectedIndex_aux = np.argmin(self.reward[validActions])
                selectedIndex = validActions[selectedIndex_aux]

        return self.controlSpace[selectedIndex, :], selectedIndex


    def updateAlg(self, inputData):
        index = inputData['index']
        x = inputData['state']
        control = inputData['control']
        con = inputData['currentConsumption']
        qos = inputData['currentQoS']

        state_control = np.append(x[0], control)
        qos_sample = 1 if qos > 0 else 0
        qos_sample = np.array([qos_sample])
        _, c_qos = self.sess.run([self.optimizer_qos, self.cost_qos], feed_dict={self.x: np.expand_dims(state_control, axis=0), self.y: np.expand_dims(qos_sample, axis=0)})

        self.count[index] += 1
        nk = self.count[index]
        self.reward[index] = (nk-1)/nk * self.reward[index] + 1/nk*con
        return c_qos

    def getConf(self):
        return self.algConf

    def closeAlg(self):
        self.sess.close()
