# ResNet50 model set up reference: https://blog.csdn.net/sinom21/article/details/128126163


from keras import layers
from keras.layers import Input, Activation, BatchNormalization, Flatten
from keras.layers import Dense, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, Dropout
from keras.models import Model

# Set up ResNet50 model

def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters

    name_base = str(stage) + block + '_identity_block_'

    x = Conv2D(filters1, (1, 1), name=name_base + 'conv1')(input_tensor)
    x = BatchNormalization(name=name_base + 'bn1')(x)
    x = Activation('relu', name=name_base + 'relu1')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=name_base + 'conv2')(x)
    x = BatchNormalization(name=name_base + 'bn2')(x)
    x = Activation('relu', name=name_base + 'relu2')(x)

    x = Conv2D(filters3, (1, 1), name=name_base + 'conv3')(x)
    x = BatchNormalization(name=name_base + 'bn3')(x)

    x = layers.add([x, input_tensor], name=name_base + 'add')
    x = Activation('relu', name=name_base + 'relu4')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters

    res_name_base = str(stage) + block + '_conv_block_res_'
    name_base = str(stage) + block + '_conv_block_'

    x = Conv2D(filters1, (1, 1), strides=strides, name=name_base + 'conv1')(input_tensor)
    x = BatchNormalization(name=name_base + 'bn1')(x)
    x = Activation('relu', name=name_base + 'relu1')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=name_base + 'conv2')(x)
    x = BatchNormalization(name=name_base + 'bn2')(x)
    x = Activation('relu', name=name_base + 'relu2')(x)

    x = Conv2D(filters3, (1, 1), name=name_base + 'conv3')(x)
    x = BatchNormalization(name=name_base + 'bn3')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=res_name_base + 'conv')(input_tensor)
    shortcut = BatchNormalization(name=res_name_base + 'bn')(shortcut)

    x = layers.add([x, shortcut], name=name_base + 'add')
    x = Activation('relu', name=name_base + 'relu4')(x)
    return x


def ResNet50(classes):
    img_input = Input(shape=[224, 224, 3])
    x = ZeroPadding2D((3, 3))(img_input)

    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(classes, activation='softmax', name='breastHis')(x)

    model = Model(img_input, x, name='resnet50')
    return model
