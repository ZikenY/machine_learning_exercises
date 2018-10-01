## ----------------------------------------------------------------
## use Tensorflow and BaseEstimator to implement Logistic Regression
## ----------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

from sklearn.base import BaseEstimator

# ------------------- prepare the data --------------------------
# 100个2D samples, random points
N = 100
D = 2
trainX = np.random.randn(N, D)

# separate the points into 2 clusters by centroid distance of 1.75*2
delta = 1.75
trainX[:N//2] += np.array([delta, delta])
trainX[N//2:] += np.array([-delta, -delta])

labels = np.array([0] * (N//2) + [1] * (N//2))
print('labels: ', labels) 

plt.scatter(trainX[:,0], trainX[:,1], s=100, c=labels, alpha=0.5)
plt.show()

# save the none-one-hot labels for drawing the boundary
original_labels = np.array([0] * (N//2) + [1] * (N//2))
# ------------------- prepare the data --------------------------


from sklearn.metrics import accuracy_score
class LogisticRegressionTF(BaseEstimator):
    def __init__(self, learning_rate, training_epochs, display_step, annotate=False):
        self.annotate = annotate
        self.sess = tf.Session()
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs
        self.display_step = display_step
        
    def fit(self, trainX, labels):
        N, D = trainX.shape  # N samples, D features each
        c = labels.shape[1]  # c is the number of label classes
        
        # ------------------ define the graph ---------------------
        self.X = tf.placeholder(tf.float64, shape=[None, D])   # X:[NxD], N is pending
        self.Y = tf.placeholder(tf.float64, shape=[None, c])   # T:[Nxc],

        # initialize the trainable variables 
        self.W = tf.Variable(np.random.randn(D, c), name='weight') # [X]*[W]: [NxD]*[Dxc]. so W is [Dxc].
        self.b = tf.Variable(np.random.randn(c), name='bias')
        
        
        # logistic prediction   
        output_logits = tf.add(tf.matmul(self.X, self.W), self.b)
        
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output_logits, labels=self.Y))
        # or loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
        # ------------------ define the graph ---------------------

        # optimizer
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)

        # turn logits to probability
        self.pred = tf.sigmoid(output_logits)
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
        # draw boundary
        if self.annotate:
            assert len(trainX.shape) == 2, "Only 2d points are allowed!!"
            plt.scatter(trainX[:,0], trainX[:,1], s=100, c=original_labels, alpha=0.5) 

            h = .02 
            x_min, x_max = trainX[:, 0].min() - 1, trainX[:, 0].max() + 1
            y_min, y_max = trainX[:, 1].min() - 1, trainX[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

            Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
            plt.title("This is where model starts to learn!!")
            plt.show()


        # start to train
        for epoch in range(self.training_epochs):
            for (x, y) in zip(trainX, labels):
                self.sess.run(optimizer, feed_dict={self.X:np.asmatrix(x), self.Y:np.asmatrix(y)})

            # show the intermediate trainning result (cost value and the fitting line)
            if (epoch+1) % self.display_step == 0:
                current_cost = self.sess.run(loss, feed_dict={self.X:trainX, self.Y:labels})
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(current_cost), \
                  "W=", self.sess.run(self.W), "b=", self.sess.run(self.b))
                
                # show the fitted line
                if self.annotate:
                    assert len(trainX.shape) == 2, "Only 2d points are allowed!!"
                    plt.scatter(trainX[:,0], trainX[:,1], s=100, c=original_labels, alpha=0.5) 
                    h = .02 
                    x_min, x_max = trainX[:, 0].min() - 1, trainX[:, 0].max() + 1
                    y_min, y_max = trainX[:, 1].min() - 1, trainX[:, 1].max() + 1
                    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

                    Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
                    Z = Z.reshape(xx.shape)
                    plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
                    plt.show()


        print("Optimization Finished!")
        training_cost = self.sess.run(loss, feed_dict={self.X:trainX, self.Y:labels})
        print("Training cost=", training_cost, "W=", self.sess.run(self.W), "b=", self.sess.run(self.b), '\n')

    def predict(self, testX):
        prediction = self.sess.run(self.pred, feed_dict={self.X:testX})
        return np.argmax(prediction, axis=1)
    
    # testY should be one-hot
    def score(self, testX, testY):
        result = self.predict(testX)

        # suppose the testY has been one hot encoded
        #eg:#0: [1,0]  -> (id, 0)
            #1: [1,0]  -> (id, 0)
            #2: [0,1]  -> (id, 1)
        _ , true_result = np.where(testY==1)
        return accuracy_score(true_result, result)


# ---------------------  one-hot the labels!!!  -----------------------------------
from sklearn.preprocessing import OneHotEncoder
one_hot_encoder = OneHotEncoder()
one_hot_encoder.fit(labels.reshape(N, -1))
labels = one_hot_encoder.transform(labels.reshape(N, -1)).toarray() # 别忘了将sparse_matrix转换为nparray!!
# ---------------------  one-hot the labels!!!  -----------------------------------

logisticTF = LogisticRegressionTF(learning_rate=0.01, training_epochs=1000, display_step=200, annotate=True)
logisticTF.fit(trainX, labels)

from sklearn.model_selection import cross_val_score
logisticTF = LogisticRegressionTF(learning_rate=0.01, training_epochs=1000, display_step=300, annotate=False)
cross_val_score(logisticTF, trainX, labels, cv=3).mean()

tf.reset_default_graph()
