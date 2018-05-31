import sys, os
from vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import keras
from keras.layers.convolutional import Conv2D
from termcolor import cprint

from model_modify import modify_model, cluster_model_kernels, save_cluster_result, load_cluster_result
from vgg19_modified import VGG19_Modified

from model_train_and_test import evaluate_model, training_model, fine_tune_model

def print_conv_layer_info(model):
    f = open("./tmp/conv_layers_info.txt", "w")
    f.write("layer index   filter number   filter shape(HWCK)\n")

    cprint("conv layer information:", "red")
    for i, l in enumerate(model.layers):
        if isinstance(l, Conv2D):
            print i, l.filters, l.kernel.shape.as_list()
            print >> f, i, l.filters, l.kernel.shape.as_list()
    f.close()

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    model = VGG19()
    #model.summary()
    model.load_weights('./weights/vgg19_weights_71_08.h5')

    keras.utils.plot_model(model, to_file="./tmp/vgg19.png")
    print_conv_layer_info(model)

    kmeans_k = 512
    file = "./tmp/vgg19_" + str(kmeans_k)

    #cluster_id, temp_kernels = cluster_model_kernels(model, k=kmeans_k, t=3)
    #save_cluster_result(cluster_id, temp_kernels, file)
    cluster_id, temp_kernels = load_cluster_result(file)

    model_new = modify_model(model, cluster_id, temp_kernels)

    print "start fine-tuneing"
    # print "start testing"
    # print "start training"
    #evaluate_model(model)
    evaluate_model(model_new)
    #fine_tune_model(model_new, epochs=10)
    # training_model(model, epoches=10, batch_size=64)


if __name__ == "__main__":
    main()

