import numpy as np
import tensorflow as tf

class NeuralBandit:

    def __init__(self, nPicos, ABSval, CREval, initExploration, epsilon_0, batch_size=1):

        nActivePicosVal = np.arange(0, (nPicos+1))
        self.controlSpace = np.array(np.meshgrid(nActivePicosVal, ABSval, CREval)).T.reshape(-1, 3)
        self.nControls = len(self.controlSpace[:, 0])
        # self.nControls = 10

        # Network Parameters
        n_hidden_1 = 20  # 1st layer number of features
        n_hidden_2 = 20  # 2nd layer number of features
        n_input = 2 # data input
        n_output = 1 # function output
        learning_rate = 0.001
        self.batch_size = batch_size

        # self.batch_count = np.zeros((self.nControls))
        # self.batch_buffer = np.zeros((self.nControls, self.batch_size))

        self.count = np.zeros((self.nControls))
        self.current_cost = np.zeros((self.nControls))
        self.initExploration = initExploration
        self.epsilon_0 = epsilon_0

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

        for i in range(self.nControls):
            weights = {
                'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name='h1_'+str(i)),
                'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='h2_'+str(i)),
                'out': tf.Variable(tf.random_normal([n_hidden_2, n_output]), name='hout_'+str(i))
            }
            biases = {
                'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='b1_'+str(i)),
                'b2': tf.Variable(tf.random_normal([n_hidden_2]), name='b2_'+str(i)),
                'out': tf.Variable(tf.random_normal([n_output]), name='bout_'+str(i))
            }
            pred = multilayer_perceptron(self.x, weights, biases)
            cost = tf.reduce_sum(tf.pow(pred - self.y, 2)) / self.batch_size
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

            self.neuralArms.append(pred)
            self.armCost.append(cost)
            self.armOptimizer.append(optimizer)
            if np.mod(i, 20) == 0:
                print('NeuralBandit: Created NN number ' + str(i) + ' of '+str(self.nControls))

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

        self.algConf = {'epsilon_0': epsilon_0,
                   'initExploration': initExploration,
                    'batch_size': batch_size,
                    'n_hidden_1': n_hidden_1,
                    'n_hidden_2': n_hidden_2,
                    'learning_rate': learning_rate,
                    'Common_layer': 'no'}

    def getControl(self, inputData):
        x = inputData['state']

        indexes = np.where(self.count < self.initExploration)[0]

        if len(indexes) > 0:
            array_index = np.random.randint(0, len(indexes))
            selectedIndex = indexes[array_index]
        else:
            epsilon_desc = self.epsilon_0 / np.sum(self.count)
            if np.random.rand() < epsilon_desc:
                selectedIndex = np.random.randint(0, self.nControls, 1)[0]
            else:
                estimatedReward = np.zeros(self.nControls)
                for i in range(self.nControls):

                    estimatedReward[i] = self.sess.run([self.neuralArms[i]], feed_dict={self.x: np.expand_dims(x, axis=0)})[0][0][0]
                # print(estimatedReward)
                selectedIndex = np.argmin(estimatedReward)

        return self.controlSpace[selectedIndex, :], selectedIndex


    def updateAlg(self, inputData):
        index = inputData['index']
        x = inputData['state']
        reward = inputData['utilityFunctionVal']

        self.count[index] += 1
        # self.batch_buffer[index, self.batch_count[index]] = reward
        # self.batch_count[index] += 1
        #
        # if self.batch_count[index] == self.batch_size:
        #     _, self.current_cost[index] = self.sess.run([self.armOptimizer[index], self.armCost[index]], feed_dict={self.x: np.expand_dims(x, axis=0), self.y: np.expand_dims(reward, axis=0)})
        _, self.current_cost[index] = self.sess.run([self.armOptimizer[index], self.armCost[index]], feed_dict={self.x: np.expand_dims(x, axis=0), self.y: np.expand_dims(reward, axis=0)})

        return self.current_cost


    def getConf(self):
        return self.algConf


    def closeAlg(self):
        self.sess.close()
