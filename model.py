
import tensorflow as tf

import setting
from preprocessing import preprocessing


class UNet(tf.Module):
    def __init__(self):
        super().__init__()
        self.pool_size = 2
        self.dropout_rate = 0.5
        self.filter = 64
        self.kernel_init = 'he_normal'
        self.padding = 'same'

    def build(self, input):
        x = input
        x, num_filters, features = self.contracting_path(x)

        for _ in range(2):
            x = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=3, kernel_initializer=self.kernel_init, padding=self.padding)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)

        num_filters /= 2

        x = self.expanding_path(x, num_filters, features)

        return x

    def contracting_path(self, x):
        num_filters = self.filter
        features = []

        for n in range(4):
            for _ in range(2):
                x = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=3, kernel_initializer=self.kernel_init, padding=self.padding)(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.ReLU()(x)
            features.append(x)
            x = tf.keras.layers.MaxPool2D(pool_size=(self.pool_size, self.pool_size), strides=2)(x)
            num_filters *= 2

        return x, num_filters, features

    def expanding_path(self, x, num_filters, features):
        for n in range(4):
            x = tf.keras.layers.Conv2DTranspose(filters=num_filters, kernel_size=2, kernel_initializer=self.kernel_init, padding=self.padding, strides=2)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.concatenate([features.pop(), x], axis=3)
            for _ in range(2):
                x = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=3, kernel_initializer=self.kernel_init,padding=self.padding)(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.ReLU()(x)
            num_filters /= 2
        x = tf.keras.layers.Conv2D(filters=1, kernel_size=1)(x)
        x = tf.keras.activations.sigmoid(x)

        return x


def get_unet():
    unet_model = UNet()

    input = tf.keras.layers.Input(shape=(setting.Patch_size, setting.Patch_size, 1))
    output = unet_model.build(input)
    model = tf.keras.Model(input, output)

    return model


def print_summary_model(model):
    model.summary()


if __name__ == '__main__':
    prd_data = preprocessing()
    md = get_unet()
    print_summary_model(md)
    val_image, val_label = prd_data.get_dataset(check_opt=False)
    unet_model = UNet()
    output = unet_model.build(val_image)