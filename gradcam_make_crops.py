'''Script taking non-detected bird images and old model,
Outputting cropped regions around high-probability CAM areas
Author: Group 8 CS4245 - Jan Peter Simons'''


############# IMPORTS ##############
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input,
    decode_predictions)
import cv2

class GradCamG8:
    def __init__(self, image_dir, show=False) -> None:
        self.__ROOT_DIR = os.getcwd()
        self.image = np.array(load_img(f"{self.__ROOT_DIR}{image_dir}", target_size=(224, 224, 3)))
        self.show = show
        self.model = ResNet50()
        self.gradcam = None
        self.counter_image = None

    def generate_heatmap(self):
        # Make model of last layer of ResNet50 network:
        last_conv_layer = self.model.get_layer("conv5_block3_out")
        last_conv_layer_model = tf.keras.Model(self.model.inputs, last_conv_layer.output)
        
        # Get final predictions
        classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input
        for layer_name in ["avg_pool", "predictions"]:
            x = self.model.get_layer(layer_name)(x)
        classifier_model = tf.keras.Model(classifier_input, x)

        # Get output till last convolution layer
        # Calculate predictions of target class wrt output of this model
        with tf.GradientTape() as tape:
            inputs = self.image[np.newaxis, ...]
            last_conv_layer_output = last_conv_layer_model(inputs)
            tape.watch(last_conv_layer_output)
            preds = classifier_model(last_conv_layer_output)
            top_pred_index = tf.argmax(preds[0])
            top_class_channel = preds[:, top_pred_index]
        
        # Gradients of model wrt feature map activations of convolution layer:
        grads = tape.gradient(top_class_channel, last_conv_layer_output)
        # To which pixels do these accord?
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        # Multiply gradients with actual feature map
        last_conv_layer_output = last_conv_layer_output.numpy()[0]
        pooled_grads = pooled_grads.numpy()
        for i in range(pooled_grads.shape[-1]):
            last_conv_layer_output[:, :, i] *= pooled_grads[i]

        # Average over all the filters to get a single 2D array
        self.gradcam = np.mean(last_conv_layer_output, axis=-1)
        # Clip the values (equiv to applying ReLU) + normalize
        self.gradcam = np.clip(self.gradcam, 0, np.max(self.gradcam)) / np.max(self.gradcam)
        self.gradcam = cv2.resize(self.gradcam, (224, 224))
        # self.gradcam = self.gradcam.astype('uint8')

        # Show result
        # if self.show:
        #     plt.title('Image + Heatmap')
        #     plt.imshow(self.image)
        #     plt.imshow(self.gradcam, alpha=0.5)
        #     plt.show()
        
    def gen_bbox_of_heatmap(self):
        
        if self.gradcam == None:
            self.generate_heatmap()
            if self.show:
                print("Generating Heatmap")
        
        norm_gradcam = np.zeros((224,224))
        norm_gradcam = cv2.normalize(self.gradcam, norm_gradcam, 0, 255, cv2.NORM_MINMAX)
        # print(f'{norm_gradcam = }')
        norm_gradcam = norm_gradcam.astype("uint8")

        # gray = cv2.cvtColor(self.gradcam, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(norm_gradcam, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Find contours
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        self.image_bbox = self.image.copy()
        self.cropped_imgs = []
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(self.image_bbox, (x, y), (x + w, y + h), (36,255,12), 2)
            cropped_img = self.image[y:y+h, x:x+w]
            self.cropped_imgs.append(cropped_img)
        
        if len(self.cropped_imgs) > 1:
            print(f'No of crops: {len(self.cropped_imgs)}')
            multiple = True

        if self.show:
            plt.figure(figsize=(10,10))
            plt.suptitle('Grad-CAM results')
            plt.subplot(221)
            plt.title("Original image")
            plt.imshow(self.image)
            plt.subplot(222)
            plt.title("Heatmap + Boundingbox")
            plt.imshow(self.image_bbox)
            plt.imshow(self.gradcam, alpha=0.5)
            plt.subplot(223)
            plt.title("Threshold")
            plt.imshow(thresh)
            plt.subplot(224)
            plt.title('Cropped image')
            if multiple == True: 
                plt.imshow(self.cropped_imgs[1])
            else: 
                plt.imshow(self.cropped_imgs)
            plt.show()
        
    def generate_countermap(self):
        # Make model of last layer of ResNet50 network:
        last_conv_layer = self.model.get_layer("conv5_block3_out")
        last_conv_layer_model = tf.keras.Model(self.model.inputs, last_conv_layer.output)
        
        # Get final predictions
        classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input
        for layer_name in ["avg_pool", "predictions"]:
            x = self.model.get_layer(layer_name)(x)
        classifier_model = tf.keras.Model(classifier_input, x)
        
        with tf.GradientTape() as tape:
            last_conv_layer_output = last_conv_layer_model(self.image[np.newaxis, ...])
            tape.watch(last_conv_layer_output)
            preds = classifier_model(last_conv_layer_output)
            top_pred_index = tf.argmax(preds[0])
            top_class_channel = preds[:, top_pred_index]
        
        grads = tape.gradient(top_class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(-1 * grads, axis=(0, 1, 2))
    
        last_conv_layer_output = last_conv_layer_output.numpy()[0]
        pooled_grads = pooled_grads.numpy()
        for i in range(pooled_grads.shape[-1]):
            last_conv_layer_output[:, :, i] *= pooled_grads[i]
        
        # Average over all the filters to get a single 2D array
        ctfcl_gradcam = np.mean(last_conv_layer_output, axis=-1)
        # Normalise the values
        ctfcl_gradcam = np.clip(ctfcl_gradcam, 0, np.max(ctfcl_gradcam)) / np.max(ctfcl_gradcam)
        ctfcl_gradcam = cv2.resize(ctfcl_gradcam, (224, 224))

        if self.show:
            plt.imshow(self.image)
            plt.imshow(ctfcl_gradcam, alpha=0.5)
            plt.show()
        
        mask = cv2.resize(ctfcl_gradcam, (224, 224))
        mask[mask > 0.1] = 255
        mask[mask != 255] = 0
        mask = mask.astype(bool)

        ctfctl_image = self.image.copy()
        ctfctl_image[mask] = (0, 0, 0)

        self.counter_image = ctfctl_image

        if self.show:
            plt.imshow(ctfctl_image)
            plt.show()
        
############# TESTS for executing this file ###################

if __name__ == "__main__":
    image_path = './old_dataset/birds_images/bird_1.png' 
    
    
    gradcam = GradCamG8(image_path, show=True)
    # gradcam.generate_countermap()
    gradcam.gen_bbox_of_heatmap()
    cropped_images = gradcam.cropped_imgs
    
    # SAVE HERE:
    # for images in cropped_images:
    #     np.save()
    
    # all_images_paths = []
    # for image in all_images_paths:
    #     gradcam = GradCamG8
    #     crops = gradcam.cropped_imgs
    #     del gradcam



    del gradcam