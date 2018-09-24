import pandas as pd
import tensorflow as tf

TRAIN_URL = 'http://download.tensorflow.org/data/iris_training.csv'
TEST_URL = 'http://download.tensorflow.org/data/iris_test.csv'

CSV_COLUMN_NAMES = ['SepalLength','SepalWidth',PetalLength',PetalWidth','Species']


SPECIES = ['Setosa' , 'Versicolor','Virginica']

def maybe_download():

    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)


    return train_path, test_path

def load_data(y_name = 'Species'):
    train_path, test_path = maybe_download()

    train = pd.read_csv(test_path, names = CVS_COLUMN_NAMES, header = 0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)

def 
