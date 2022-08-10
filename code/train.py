
import tensorflow as tf

import setting
import time
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from preprocessing import preprocessing
from tensorflow.keras.optimizers import Adam
from model import get_unet, print_summary_model

class Session():
    def __init__(self, save_loss, save_val):
        self.unet_model = get_unet()
        print_summary_model(self.unet_model)
        self.unet_model.compile(optimizer=Adam(learning_rate=1e-4), loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])

        self.start_step = 0
        self.end_step = setting.Iteration

        self.dataset = preprocessing()

        self.opt_save_fig_loss = save_loss
        self.opt_save_fig_val = save_val

        self.val_input_path = setting.Val_data_path + setting.Aug_path + setting.Image_path
        self.val_label_path = setting.Val_data_path + setting.Aug_path + setting.Label_path

    def training(self):
        loss_ary = []
        ep_ary = []
        loss_avg = 0.0
        print('# Start training, with %d iteration' %(self.end_step))
        start_time = time.time()
        for episode in range(self.start_step, self.end_step):
            if not episode:
                train_image, train_label = self.dataset.get_dataset(check_opt=True, random=True)
            else:
                train_image, train_label = self.dataset.get_dataset(check_opt=False, random=True)
            history = self.unet_model.fit(train_image, train_label, verbose=0)
            loss = history.history['loss']
            loss_avg += loss[0]
            if (episode+1) % setting.Val_num == 0:
                if not os.path.exists(setting.Fig_path):
                    os.mkdir(setting.Fig_path)
                print('########## Average of loss: %f in %d iteration' %((loss_avg/setting.Val_num), episode))
                loss_avg = 0.0
                if self.opt_save_fig_val:
                    if (episode+1) == setting.Val_num:
                        self.validation(episode, True)
                    else:
                        self.validation(episode, False)
                loss_ary.append(loss[0])
                ep_ary.append(episode)
        end_time = time.time()
        train_time = end_time - start_time

        print('# Total training time : %d' %(train_time))

        if self.opt_save_fig_loss:
            self.save_fig_loss(ep_ary, loss_ary)

    def validation(self, ep, save_opt):
        if save_opt:
            self.dataset.save_val_dataset()

        val_images = []
        val_labels = []
        for i in range(setting.Batch_size):
            val_image = cv2.imread(self.val_input_path + 'aug_val_input_%d.png' %(i), cv2.IMREAD_GRAYSCALE)
            val_label = cv2.imread(self.val_label_path + 'aug_val_label_%d.png' %(i), cv2.IMREAD_GRAYSCALE)

            val_image = val_image/255.0
            val_label = val_label/255.0

            val_image = np.expand_dims(np.asarray(val_image), axis=2)
            val_label = np.expand_dims(np.asarray(val_label), axis=2)

            val_images.append(val_image)
            val_labels.append(val_label)

        val_images = np.stack(val_images)
        val_labels = np.stack(val_labels)
        output_label = self.unet_model.predict(val_images)
        for i in range(len(output_label)):
            output_label[i][output_label[i]<=0.5] = 0.0
            output_label[i][output_label[i]> 0.5] = 1.0

        plt.figure(figsize=(7, 7))
        titles = ['Input image', 'Target image', 'Output image']
        num_fig = setting.Batch_size
        for i in range(num_fig):
            plt.subplot(3, 3, i*num_fig+1)
            plt.imshow(val_images[i], cmap='gray')
            plt.title(titles[0], fontdict={'fontsize':9})
            plt.axis('off')

            plt.subplot(3, 3, i*num_fig+2)
            plt.imshow(val_labels[i], cmap='gray')
            plt.title(titles[1], fontdict={'fontsize':9})
            plt.axis('off')

            plt.subplot(3, 3, i*num_fig+3)
            plt.imshow(output_label[i], cmap='gray')
            plt.title(titles[2], fontdict={'fontsize':9})
            plt.axis('off')

        plt.savefig(setting.Fig_path + 'Fig_val_%d iteration.png' %(ep))
        plt.close()

    def save_fig_loss(self, ep_ary, loss_ary):
        plt.plot(ep_ary, loss_ary)
        plt.xlabel('Episode', loc='center')
        plt.ylabel('Loss', loc='center')

        plt.savefig(setting.Fig_path + 'Loss_graph.png')

if __name__ == '__main__':
    sess = Session(save_loss=True, save_val=True)
    sess.training()