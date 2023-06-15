'''Script taking non-detected bird images and old model,
Outputting cropped regions around high-probability CAM areas
Author: Group 8 CS4245 - Jan Peter Simons'''


############# IMPORTS ##############
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions
import cv2
from resnet50_classes import all_classes



class GradCamG8:
    def __init__(self, image_dir, show=False) -> None:
        self.image = np.array(load_img(f"{image_dir}", target_size=(224, 224, 3)))
        self.show = show
        self.model = ResNet50()
        self.gradcam = None
        self.counter_image = None
        self.index_list = np.array([7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,80,81,82,83,84,
                                    85,86,87,88,89,90,100,127,128,129,130,131,132,133,134,135,
                                    136,137,138,139,140,141,142,143,144,145,146])

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
            
            preds_birds = np.take(np.array(preds),self.index_list)
            top_pred_bird_index = self.index_list[tf.argmax(preds_birds)]
            top_class_channel = preds[:, top_pred_bird_index]
            resnet_prediction = list(all_classes.values())[top_pred_bird_index]
            print(f'Resnet50 pred = {resnet_prediction}\nScore: {np.max(preds_birds)}\n')
        
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
        
        multiple = False
        if len(self.cropped_imgs) > 1:
            multiple = True
            if self.show:
                print(f'No of crops: {len(self.cropped_imgs)}')
            
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
            inputs = last_conv_layer_model(self.image[np.newaxis, ...])
            last_conv_layer_output = last_conv_layer_model(inputs)
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

    ROOT_DIR = os.getcwd()
    
    folder_dir = f"{ROOT_DIR}/old_dataset/missed_mrcnn"
    for image_name in os.listdir(folder_dir):

        if (image_name.endswith(".jpg") or image_name.endswith(".jfif")): #or image_name.endswith(".jfif")
            print(f'{image_name = }')

            image_path = f'{folder_dir}/{image_name}'
            image_name = image_path.split('/')[-1]

            gradcam = GradCamG8(image_path, show=False)
            gradcam.gen_bbox_of_heatmap()
            cropped_images = gradcam.cropped_imgs

            # SAVE heatmap:
            plt.figure()
            plt.imshow(gradcam.image_bbox)
            plt.imshow(gradcam.gradcam, alpha=0.5)
            if image_name.endswith(".jfif"):
                name = image_name.split('.')[0]
                image_name = name + '.jpg'
            save_path_heatmap = f'{ROOT_DIR}/heatmap_crops/heatmap_{image_name}'
            plt.savefig(save_path_heatmap)

            # SAVE crops:
            count = 0
            for image in cropped_images:
                plt.figure()
                plt.imshow(image)
                save_path_crops = f'{ROOT_DIR}/heatmap_crops/crop{count}_{image_name}'
                plt.savefig(save_path_crops)
                count += 1

            del gradcam