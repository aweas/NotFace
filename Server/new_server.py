import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import socket
import scipy.misc
import base64
import time
import skimage.transform
from os import listdir
from os import environ
import warnings

warnings.filterwarnings("ignore")
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

shape = (64, 64, 3)
patch_size = 32
stride = 1/2


class PredictionModel:

    def __init__(self, path_to_model=None, model_dir='../trained_models'):
        self.session = None

        if path_to_model is not None:
            self.path_to_model = path_to_model
        else:
            self._get_latest_model(model_dir)

    def _get_latest_model(self, model_dir):
        print('Looking for latest model')

        all_files = ['.'.join(f.split('.')[:-1])
                     for f in listdir(model_dir) if f.split('_')[0] == 'model']
        duplicates_removed = list(set(all_files))

        self.path_to_model = f'{model_dir}/{max(duplicates_removed)}'

        print('Latest model found:', self.path_to_model)
        date = self.path_to_model.split('/')[-1].split('_')[1]
        time = self.path_to_model.split('/')[-1].split('_')[2].split('.')[0].split('-')
        print(f'Training time: {date} {time[0]}:{time[1]}')

    def inference(self, inputs):
        inputs = tf.cast(inputs, tf.float32)

        # ARCHITECTURE
        with tf.name_scope("convolutional"):
            conv1 = tf.layers.conv2d(
                inputs=inputs,
                filters=32,
                kernel_size=3,
                padding='same',
                activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(
                inputs=conv1, pool_size=2, strides=2)

            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=32,
                kernel_size=3,
                padding='same',
                activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(
                inputs=conv2, pool_size=2, strides=2)

            conv3 = tf.layers.conv2d(
                inputs=pool2,
                filters=64,
                kernel_size=3,
                padding='same',
                activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(
                inputs=conv3, pool_size=2, strides=2)
            flatten = tf.reshape(
                pool3, [-1, int((shape[0] / 8) * (shape[1] / 8) * 64)])

        with tf.name_scope("dense"):
            dense1 = tf.layers.dense(
                inputs=flatten,
                units=512)
            dense1_batch = tf.layers.batch_normalization(dense1, training=True)
            dense1_relu = tf.nn.relu(dense1_batch)

            dense2 = tf.layers.dense(
                inputs=dense1_relu,
                units=256,
                activation=tf.nn.relu)

            dense3 = tf.layers.dense(
                inputs=dense2,
                units=128)
            dense3_batch = tf.layers.batch_normalization(dense3, training=True)
            dense3_relu = tf.nn.relu(dense3_batch)

            dense4 = tf.layers.dense(
                inputs=dense3_relu,
                units=64,
                activation=tf.nn.relu)

        with tf.name_scope("logits"):
            logits = tf.layers.dense(inputs=dense4, units=2)
        return logits

    def predict_proba(self, X):
        pred_x = tf.convert_to_tensor(X, dtype=np.float32)

        with self.session.as_default():
            with tf.variable_scope("DNN", reuse=True) as scope:
                logits = self.inference(pred_x)
                predictions = tf.nn.softmax(logits)

            return predictions.eval()

    def predict(self, X):
        result = self.predict_proba(X)
        return np.argmax(result, axis=1)

    def restore(self):
        print('Restoring model')
        tf.reset_default_graph()
        dummy_features = np.zeros((1, *shape), dtype=np.float32)

        with tf.variable_scope("DNN") as scope:
            _ = self.inference(dummy_features)

        self.session = tf.Session()

        with self.session.as_default():
            new_saver = tf.train.Saver()
            new_saver.restore(self.session, self.path_to_model)
        print('Model restored')


class FaceFinder:
    def __init__(self, prediction_model):
        self.img = None
        self.img_not_processed = None
        self.img_orig = None
        self.occurences_map = None
        self.patches = []
        self.preds = None
        self.proba = None
        self.prediction_model = prediction_model

    def process_photo(self, img_name, patch_size=64, stride=1 / 2):
        self.patches = []
        self.img = plt.imread(img_name)
        self.img_not_processed = self.img.copy()
        self.img = preprocess_image(self.img)
        self.img_orig = self.img.copy()
        self.occurences_map = np.zeros(
            shape=(self.img.shape[0], self.img.shape[1]))

        print('Splitting photo into frames')
        self._split_pic_into_frames(size=patch_size, stride=stride)
        print('Running predictions')
        self._run_predictions()
        self._rectangify_patches()
        print('Marking photo')
        return self._mark_photo()

    def _split_pic_into_frames(self, size, stride):
        sh_x = self.img.shape[1]
        sh_y = self.img.shape[0]
        len_x = size
        len_y = size
        stride_x = np.ceil(len_x * stride).astype(int)
        stride_y = np.ceil(len_y * stride).astype(int)

        for y in range(0, sh_y, stride_y):
            for x in range(0, sh_x, stride_x):
                patch = self.img[y:y + len_y, x:x + len_x]
                if patch.shape[0] * patch.shape[1] != 0:
                    self.patches.append(
                        [patch, (y, x), (y + len_y, x + len_x)])
        print(f'Split into {len(self.patches)} frames')

    def _rectangify_patches(self):
        columns = np.ceil(self.img.shape[1]/patch_size/stride).astype(int)
        rows = np.ceil(self.img.shape[0]/patch_size/stride).astype(int)

        cont = True
        while cont:
            cont = False
            for r in range(rows - 1):
                for c in range(columns - 1):
                    if self.preds[r * columns + c] == 0:
                        LD = self.preds[(r + 1) * columns + c - 1]
                        SD = self.preds[(r + 1) * columns + c]
                        RD = self.preds[(r + 1) * columns + c + 1]
                        L = self.preds[r * columns + c - 1]
                        R = self.preds[r * columns + c + 1]
                        LG = self.preds[(r - 1) * columns + c - 1]
                        SG = self.preds[(r - 1) * columns + c]
                        RG = self.preds[(r - 1) * columns + c + 1]

                        answ = L * LD * SD + SD * RD * R + R * RG * SG + SG * LG * L
                        if answ >= 1:
                            cont = True
                            self.preds[r * columns + c] = 1


    def _mark_photo(self):
        coord_start = np.array(self.patches)[:, 1]
        coord_end = np.array(self.patches)[:, 2]

        self.img.setflags(write=1)
        for start, end, pred in zip(coord_start, coord_end, self.preds):
            if pred == 1:
                self.occurences_map[max(0, start[0]):end[0], max(
                    0, start[1]):end[1]] += 1

        for start, end, pred, proba in zip(coord_start, coord_end, self.preds, self.proba):
            if pred == 1:
                weight = (proba - 1 / 2) * 2
                addition = int(
                    255 * weight) / self.occurences_map[max(0, start[0]):end[0], max(0, start[1]):end[1]]
                added = (self.img_not_processed[max(0, start[0]):end[0], max(
                    0, start[1]):end[1]][:, :, 1].astype(np.float64) + addition)
                added = np.floor(added / 255) * 255 + \
                    (1 - np.floor(added / 255)) * added
                self.debug = added

                self.img_not_processed[max(0, start[0]):end[0], max(
                    0, start[1]):end[1]][:, :, 1] = added.astype(np.uint8)

        return self.img, self.img_not_processed

    def _run_predictions(self):
        a = skimage.transform.resize(
            self.patches[0][0], shape).reshape((1, *shape))
        for i in self.patches[1:]:
            a = np.vstack((a, skimage.transform.resize(
                i[0], shape).reshape((1, *shape))))
        self.proba = np.array(
            self.prediction_model.predict_proba(a))[:, 1]
        self.preds = [round(abs(i - 0.3)) for i in self.proba]


class Server:
    def __init__(self):
        self.model = None
        self.socket = None
        self.conn = None
        self.addr = None

    def start_server(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(('', 5007))
        self.socket.listen()

    def read_data(self, msg_length=1024):
        self.conn, self.addr = self.socket.accept()
        print('Connection accepted')
        length = self.conn.recv(msg_length)
        length = int.from_bytes(length, byteorder='big')
        while length == 0:
            length = self.conn.recv(msg_length)
            length = int.from_bytes(length, byteorder='big')

        data = b''
        print(length, 'bytes read')
        self.conn.sendall(b'OK\r\n')

        while len(data) < length:
            msg = self.conn.recv(length - len(data))
            data += msg
            if not msg:
                break

        return data

    def stop_server(self):
        # self.socket.shutdown(SHUT_RDWR)
        self.socket.close()


def preprocess_image(img):
    return img


srv = Server()
try:
    srv.start_server()
    print('Server started')

    prediction_model = PredictionModel()
    prediction_model.restore()

    facefinder = FaceFinder(prediction_model)

    while True:
        image = open('image.jpeg', 'wb')
        data = srv.read_data()
        image.write(data)
        image.close()

        plt.imshow(facefinder.process_photo(
            'image.jpeg', patch_size=patch_size, stride=stride)[1])
        plt.show()
        srv.conn.sendall(b'0')
        print('')

except Exception as e:
    srv.stop_server()
    print('Exception ocurred: ')
    print(e)

finally:
    print('Server stopped')
