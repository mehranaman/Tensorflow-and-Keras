import tensorflow as tf

import numpy as np
from numpy import pi

x = np.linspace(0, 15*pi, 500)
np.random.shuffle(x)
y = np.sin(x)

x_train = x[:400]
x_test = x[400:]

y_train = y[:400]
y_test = y[400:]


feature_column = tf.feature_column.numeric_column(key = 'variable')

train_input_fn = tf.estimator.inputs.numpy_input_fn(x = {'variable':x_train}, y=y_train,batch_size = 100, num_epochs = 10, shuffle = True)
test_input_fn = tf.estimator.inputs.numpy_input_fn(x = {'variable' : x_test},y = y_test, shuffle = False)

estimator = tf.estimator.DNNRegressor([1,10,1], feature_columns = [feature_column])

estimator.train(input_fn = train_input_fn, max_steps = 100)


eval_result = estimator.evaluate(input_fn = test_input_fn)
print('result is: ', eval_result)


x_pred = np.arange(0.07,20, 0.07)
x_pred = pi*x_pred
print('sdsd')
pred_input_fn = tf.estimator.inputs.numpy_input_fn(x = {'variable':x_pred},num_epochs = 1, shuffle = False)
print('dasdax')

predictions = estimator.predict(input_fn = pred_input_fn)


#tensorboard: use to plot how training is happening with time (iterations epoch).
