
import setting
import glob
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

class preprocessing():
    def __init__(self):
        self.train_origin_path = setting.Training_data_path + setting.Origin_path
        self.test_path = setting.Test_data_path

        self.input_image_path = self.train_origin_path + setting.Image_path
        self.input_label_path = self.train_origin_path + setting.Label_path

        self.input_images = glob.glob(self.input_image_path + '*.png')
        self.input_labels = glob.glob(self.input_label_path + '*.png')
        self.classes = len(self.input_images)

        self.data_aug = data_augmentation()

    def get_dataset(self, check_opt, random):
        ary_check_dataset = []
        ary_x, ary_y = [], []
        for i in range(setting.Batch_size):
            origin_x, origin_y, aug_x, aug_y = self.make_dataset(random, len(self.input_images)-i-1)
            ary_x.append(aug_x)
            ary_y.append(aug_y)

            if check_opt:
                ary_check_dataset.append([origin_x, origin_y, aug_x, aug_y])

        if check_opt:
            self.check_pair(ary_check_dataset, ['Origin image', 'Origin label', 'Aug image', 'Aug label'])
            print('# Augmentation: %d x %d(before) >> %d x %d(after)' %(origin_x.shape[0], origin_x.shape[1], aug_x.shape[0], aug_x.shape[1]))

        ary_x = np.stack(ary_x)
        ary_y = np.stack(ary_y)

        return ary_x, ary_y

    def make_dataset(self, random, num):
        if random:
            idx = np.random.randint(0, self.classes, size=1)[0]
        else:
            idx = num

        image_name = self.input_images[idx]
        label_name = self.input_labels[idx]

        input_image = Image.open(image_name)
        input_image = np.asarray(input_image)
        input_label = Image.open(label_name)
        input_label = np.asarray(input_label)

        data_aug = self.data_aug
        aug_image = data_aug.augmentations(input_image, True)
        aug_label = data_aug.augmentations(input_label, False)

        return input_image, input_label, aug_image, aug_label

    def check_pair(self, images, titles):
        if len(images[0]) == len(titles):
            plt.figure(figsize=(5, 5))

            for i in range(3):
                for j in range(len(images[0])):
                    plt.subplot(3, 4, i*len(images[0])+j+1)
                    plt.imshow(images[i][j], cmap='gray')
                    plt.title(titles[j], fontdict={'fontsize':10})
                    plt.axis('off')
            plt.show()
        else:
            print('# Error: Number of images and number of titles are different.')
            exit()

    def save_val_dataset(self):
        self.make_val_folder()

        for i in range(setting.Batch_size):
            idx = np.random.randint(0, self.classes, size=1)[0]

            image_name = self.input_images[idx]
            label_name = self.input_labels[idx]

            input_image = Image.open(image_name)
            input_image.save(setting.Val_data_path+setting.Origin_path+setting.Image_path+'origin_val_input_%d.png' %(i))
            input_image = np.asarray(input_image)

            input_label = Image.open(label_name)
            input_label.save(setting.Val_data_path + setting.Origin_path + setting.Label_path + 'origin_val_label_%d.png' %(i))
            input_label = np.asarray(input_label)

            data_aug = self.data_aug
            aug_image = data_aug.augmentations(input_image, True)
            aug_label = data_aug.augmentations(input_label, False)

            cv2.imwrite(setting.Val_data_path + setting.Aug_path+setting.Image_path + 'aug_val_input_%d.png' %(i), aug_image*255.0)
            cv2.imwrite(setting.Val_data_path + setting.Aug_path + setting.Label_path + 'aug_val_label_%d.png' %(i), aug_label*255.0)

    def make_val_folder(self):
        if not os.path.exists(setting.Val_data_path):
            os.mkdir(setting.Val_data_path)

        if not os.path.exists(setting.Val_data_path+setting.Origin_path):
            os.mkdir(setting.Val_data_path+setting.Origin_path)

        if not os.path.exists(setting.Val_data_path+setting.Aug_path):
            os.mkdir(setting.Val_data_path+setting.Aug_path)

        if not os.path.exists(setting.Val_data_path+setting.Origin_path+setting.Image_path):
            os.mkdir(setting.Val_data_path+setting.Origin_path+setting.Image_path)

        if not os.path.exists(setting.Val_data_path + setting.Origin_path + setting.Label_path):
            os.mkdir(setting.Val_data_path + setting.Origin_path + setting.Label_path)

        if not os.path.exists(setting.Val_data_path + setting.Aug_path+setting.Image_path):
            os.mkdir(setting.Val_data_path + setting.Aug_path+setting.Image_path)

        if not os.path.exists(setting.Val_data_path + setting.Aug_path + setting.Label_path):
            os.mkdir(setting.Val_data_path + setting.Aug_path + setting.Label_path)


class data_augmentation():
    def __init__(self):
        """
        * In paper 'Averaged over 7 rotated versions of the input data'
         (1) Flip
         (2) Add Gaussian noise
         (3) Add Uniform noise
         (4) Add Brightness
         (5) Crop
         (6) Add Pad
        """

        self.input_size = setting.Image_size
        self.output_size = setting.Patch_size

        self.flip_rand = np.random.randint(0, 3, size=1)[0]
        self.flip_opt = ['vertical', 'horizontal', 'vertical_and_horizontal']

        self.noise_rand = np.random.randint(0, 2, size=1)[0]
        self.gaussian_mean = 0
        self.gaussian_std_rand = np.random.randint(0, 21, size=1)[0]
        self.uniform_low_rand = np.random.randint(-20, 1, size=1)[0]
        self.uniform_high_rand = np.random.randint(0, 21, size=1)[0]

        self.brightness_rand = np.random.randint(-50, 50, size=1)[0]

        self.crop_sy_rand = np.random.randint(0, setting.Image_size-self.output_size+1, size=1)[0]
        self.crop_sx_rand = np.random.randint(0, setting.Image_size-self.output_size+1, size=1)[0]


    def augmentations(self, x, opt):
        x = self.aug_flip(x, self.flip_opt[self.flip_rand])

        if opt:
            if self.noise_rand == 0:
                x = self.aug_gaussian_noise(x, self.gaussian_mean, self.gaussian_std_rand)
            elif self.noise_rand == 1:
                x = self.aug_uniform_noise(x, self.uniform_low_rand, self.uniform_high_rand)

            x = self.aug_brightness(x, self.brightness_rand)

            pad_size = int((self.input_size - self.output_size) / 2)
            x = np.pad(x, pad_size, mode='symmetric')
        x = self.aug_crop(x, size=self.output_size, sx=self.crop_sx_rand, sy=self.crop_sy_rand)
        x = self.normalization(x, max=1, min=0)
        x = np.expand_dims(x, axis=2)
        return x

    def aug_flip(self, x, opt):
        if opt == 'vertical':
            x = np.flip(x, 0)
        elif opt == 'horizontal':
            x = np.flip(x, 1)
        elif opt == 'vertical_and_horizontal':
            x = np.flip(x, 0)
            x = np.flip(x, 1)

        return x

    def aug_gaussian_noise(self, x, mean=0, std=1):
        gaussian_noise = np.random.normal(mean, std, x.shape)
        x = x.astype('float64')
        x += gaussian_noise
        x[x>255] = 255.0
        x[x<0] = 0.0
        x = x.astype('uint8')

        return x

    def aug_uniform_noise(self, x, low=10, high=10):
        uniform_noise = np.random.uniform(low, high, x.shape)
        x = x.astype('float64')
        x ++ uniform_noise
        x[x>255] = 255.0
        x[x<0] = 0.0
        x = x.astype('uint8')

        return x

    def aug_brightness(self, x, value):
        x = x.astype('int16')
        x += value

        x[x>255] = 255.0
        x[x<0] = 0.0
        x = x.astype('uint8')
        return x

    def aug_crop(self, x, size, sx, sy):
        x = x[sx:sx+size, sy:sy+size]
        return x

    def aug_pad(self, x, size1, size2, opt):
        pad_size = int((size1-size2) / 2)
        x = np.pad(x, pad_size, mode=opt)
        return x

    def normalization(self, x, max, min):
        if (np.max(x) - np.min(x)) != 0:
            x = (x-np.min(x)) * (max-min) / (np.max(x) - np.min(x)) + min
        return x