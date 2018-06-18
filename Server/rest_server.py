import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64
import skimage.transform
from os import listdir
from os import environ
from flask import Flask, request
from flask_restful import reqparse, Resource, Api
import warnings


warnings.filterwarnings("ignore")
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

shape = (64, 64, 3)
# patch_size = 4
# stride = 1/2


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
        time = self.path_to_model.split(
            '/')[-1].split('_')[2].split('.')[0].split('-')
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
            # with self.session.graph.as_default():
            [print(n.name) for n in tf.get_default_graph().as_graph_def().node]

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
        self.session = tf.Session()
        dummy_features = np.zeros((1, *shape), dtype=np.float32)

        with tf.variable_scope("DNN") as scope:
            infer = self.inference(dummy_features)

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

    def process_photo(self, img, patch_size=64, stride=1 / 2, leave_patches=False):
        if not leave_patches:
            self.patches = []

        self.img = img
        self.img_not_processed = self.img.copy()
        self.img = preprocess_image(self.img)
        self.img_orig = self.img.copy()
        self.occurences_map = np.zeros(
            shape=(self.img.shape[0], self.img.shape[1]))

        print('Splitting photo into frames')
        self._split_pic_into_frames(size=patch_size, stride=stride)
        print('Running predictions')
        self._run_predictions()
        # self._rectangify_patches()
        print('Marking photo')
        return self._mark_photo()

    def _split_pic_into_frames(self, size, stride):
        sh_x = self.img.shape[1]
        sh_y = self.img.shape[0]
        len_x = int(sh_x / size)
        len_y = int(sh_x / size)
        stride_x = np.ceil(len_x * stride).astype(int)
        stride_y = np.ceil(len_y * stride).astype(int)

        for y in range(0, sh_y, stride_y):
            for x in range(0, sh_x, stride_x):
                patch = self.img[y:y + len_y, x:x + len_x]
                if patch.shape[0] * patch.shape[1] != 0:
                    self.patches.append(
                        [patch, (y, x), (y + len_y, x + len_x)])
        print(f'Split into {len(self.patches)} frames')

    def _run_predictions(self):
        a = skimage.transform.resize(
            self.patches[0][0], shape).reshape((1, *shape))
        for i in self.patches[1:]:
            a = np.vstack((a, skimage.transform.resize(
                i[0], shape).reshape((1, *shape))))
        print("-------")
        [print(n.name) for n in tf.get_default_graph().as_graph_def().node]
        self.proba = np.array(
            self.prediction_model.predict_proba(a))[:, 1]
        self.preds = [round(abs(i - 0.3)) for i in self.proba]

    def _rectangify_patches(self):
        columns = np.ceil(self.img.shape[1] / patch_size / stride).astype(int)
        rows = np.ceil(self.img.shape[0] / patch_size / stride).astype(int)

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

                # added = np.floor(added / 255) * 255 + \
                #     (1 - np.floor(added / 255)) * added
                added = np.clip(added, 0, 255)
                self.debug = added

                self.img_not_processed[max(0, start[0]):end[0], max(
                    0, start[1]):end[1]][:, :, 1] = added.astype(np.uint8)

        return self.img, self.img_not_processed


class AIServer(Resource):
    def __init__(self):
        super().__init__()
        # Load AI model
        self.prediction_model = PredictionModel()
        self.prediction_model.restore()

        # Initialize algorithm
        self.facefinder = FaceFinder(self.prediction_model)

        self.parser = reqparse.RequestParser()
        self.parser.add_argument('image')

    def post(self):
        # Retrieve image from args
        args = self.parser.parse_args()

        image_bytes = base64.b64decode(str(args['image']))
        image = Image.open(io.BytesIO(image_bytes))

        # Process it
        self.facefinder.patches = []
        for size, stride in zip([2, 4, 5], [1/4, 1/2, 1]):
            processed_img_base = self.facefinder.process_photo(np.asarray(
                image), patch_size=size, stride=stride, leave_patches=True)[1]
        processed_img = Image.fromarray(processed_img_base, 'RGB')

        # Send back to user
        buff = io.BytesIO()
        processed_img.save(buff, format='JPEG')
        img_str = base64.b64encode(buff.getvalue())

        print("Answer sent")
        return img_str.decode("utf-8")


def preprocess_image(img):
    return img


if __name__ == "__main__":
    # Initialize rest API
    app = Flask(__name__)
    api = Api(app)
    api.add_resource(AIServer, '/')

    app.run(host='192.168.1.8', port='5007')
