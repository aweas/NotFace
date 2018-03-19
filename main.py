import numpy as np
import keras
import matplotlib.pyplot as plt
import socket
import scipy.misc
import base64
import time

shape = (64, 64, 3)


class Server:
    def __init__(self):
        self.model = None
        self.socket = None
        self.conn = None
        self.addr = None

    def load_model(self, model_json='model.json', model_weights="weights_1.dat"):
        json_file = open(model_json, 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        self.model = keras.models.model_from_json(loaded_model_json)
        self.model.load_weights(model_weights)

    def test_model(self, face='face.jpg', not_face='notFace.jpg'):
        not_face = scipy.misc.imresize(plt.imread(not_face), shape)
        face = scipy.misc.imresize(plt.imread(face), shape)

        prediction = np.round(self.model.predict(np.array([not_face, face])))

        if np.array_equal(prediction[:, 0], [0, 1]):
            print("Passed test")
        else:
            print("FAILED")

    def start_server(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(('', 5007))
        self.socket.listen()

    def read_data(self, msg_length=1024):
        self.conn, self.addr = self.socket.accept()

        length = self.conn.recv(msg_length)
        length = int.from_bytes(length, byteorder='big')
        while length == 0:
            length = self.conn.recv(msg_length)
            length = int.from_bytes(length, byteorder='big')

        data = b''
        print(length)
        self.conn.sendall(b'OK\r\n')

        while len(data) < length:
            msg = self.conn.recv(length-len(data))
            data += msg
            if not msg:
                break

        return data

    def predict(self, data):
        res = self.model.predict(data)
        print('Face : notFace probability: %d%% : %d%%' % (res[0, 0]*100, res[0, 1]*100))

        prediction = np.round(res)[0, 0]
        if prediction == 1:
            return b'Face\r\n'
        else:
            return b'notFace\r\n'

if __name__ == "__main__":
    srv = Server()

    srv.load_model()
    srv.test_model()
    srv.start_server()

    while True:
        image = open('image.jpeg', 'wb')
        data = srv.read_data()
        image.write(data)
        image.close()

        img = plt.imread('image.jpeg')
        img = scipy.misc.imresize(img, (64, 64, 3))

        prediction = srv.predict(np.array([img]))

        srv.conn.sendall(prediction)