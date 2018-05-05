import sys, os
from vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

from model_train_and_test import evaluate_model, training_model


os.environ['CUDA_VISIBLE_DEVICES'] = '2'

model = VGG19(weights='imagenet')
model.load_weights('./weights/vgg19_weights.07.h5')


evaluate_model(model)
#training_model(model, epoches=10, batch_size=64)

