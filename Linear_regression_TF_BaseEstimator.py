## ----------------------------------------------------------------
## use Tensorflow and BaseEstimator to implement Linear Regression
## ----------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

from sklearn.base import BaseEstimator


"""
    the function we want to fit： y = 3x + 4
"""
def f(X):
    return 3*X + 4

N = 40            
noise_level = 0.8 

trainX = np.linspace(-4.0, 4.0, N)
np.random.shuffle(trainX)

# irreducible error
noise = np.random.randn(N) * noise_level
labels = f(trainX) + noise

# peek the train_set:
plt.scatter(trainX, labels)

learning_rate = 0.01
training_epochs = 1000
display_step = 50


class LinearRegressionTF(BaseEstimator):
    def __init__(self, learning_rate, training_epochs, display_step, annotate=False):
        self.annotate = annotate   # display or not in process
        self.sess = tf.Session()
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs
        self.display_step = display_step        

        
    '''
        perform graph operations on tensorflow
    '''
    def fit(self, trainX, labels):
        # the total number of samples
        N = trainX.shape[0]

        # -------- implement graph definition on tensorflow -----------
        # input samples and labels
        self.X = tf.placeholder('float')
        self.Y = tf.placeholder('float')        
        
        # trainable variables 'W, b' for linear regression (only 2 variables)
        self.W = tf.Variable(np.random.randn(), name='weight')
        self.b = tf.Variable(np.random.randn(), name='bias')
        
        # linear regression model (y~ = W*x + b)
        self.pred = tf.add(tf.multiply(self.X, self.W), self.b)
        # -------- implement graph definition on tensorflow -----------
        
        # mean squre error loss
        #                           self.pred is the model
        loss = tf.reduce_sum(tf.pow(self.pred - self.Y, 2)) / (N * 2)
        
        # optimizer 
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
        
        # initialize W, b
        init = tf.global_variables_initializer()
        self.sess.run(init)
                
        if self.annotate:
            # original (X, Y)s
            plt.plot(trainX, labels, 'ro', label='Original data')
            # result from our model: y~ = W*X + b
            plt.plot(trainX, self.sess.run(self.W) * trainX + self.sess.run(self.b), label='Fitted line')
            plt.legend()
            plt.title("This is where model starts to learn!!")
            plt.show()
            
        # start to train
        for epoch in range(self.training_epochs):
            for (x, y) in zip(trainX, labels):
                self.sess.run(optimizer, feed_dict={self.X:x, self.Y:y})

            # peek the intermediate result
            if (epoch+1) % self.display_step == 0:
                current_cost = self.sess.run(loss, feed_dict={self.X:trainX, self.Y:labels})
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(current_cost), \
                    "W=", self.sess.run(self.W), "b=", self.sess.run(self.b))
                
                # peek the regression line
                if self.annotate:
                    plt.plot(trainX, labels, 'ro', label='Original data')
                    plt.plot(trainX, self.sess.run(self.W) * trainX + self.sess.run(self.b), label='Fitted line')
                    plt.legend()
                    plt.show()
                #plt.pause(0.5)

        print("Optimization Finished!")
        training_cost = self.sess.run(loss, feed_dict={self.X:trainX, self.Y:labels})
        print("Training cost=", training_cost, "W=", self.sess.run(self.W), "b=", self.sess.run(self.b), '\n')

    def predict(self, testX):
        # run我们定义的pred 线性model，feed into testX，传出prediction y~
        prediction = self.sess.run(self.pred, feed_dict={self.X:testX})
        return prediction
    
    def score(self, testX, testY):
        result = self.predict(testX)
        return r2_score(testY, result)

lr = LinearRegressionTF(learning_rate=0.01, training_epochs=1000, display_step=300, annotate=True)
lr.fit(trainX, labels)

from sklearn.model_selection import cross_val_score
lrtf = LinearRegressionTF(learning_rate=0.01, training_epochs=1000, display_step=200, annotate=False)
cv_mean = cross_val_score(lrtf, trainX, labels, cv=3).mean()
print('cross_val_score.mean(): ', cv_mean)

tf.reset_default_graph()
