from red import *
test = Network()
test.retrieve('../Desktop')
test.keep('../Desktop')
train = Data()
types, data, _ = train.grayscale_jpg_load('../Desktop/data/handwritten-characters/Train', (32,32))
labels = data
test.train(epochs=2, batch_size=128, data=data, labels=labels, types=types, learning_rate=0.001)
