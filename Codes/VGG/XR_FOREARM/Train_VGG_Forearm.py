# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 16:06:59 2018

@author: prant
"""

import time
import pickle
import warnings

from keras import Model
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from keras.utils import to_categorical
from keras import optimizers
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.engine.topology import get_source_inputs
from keras.layers import Input
from keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")

t1=time.time()
x_train = pickle.load(open("Splited_X_Train_Forearm_HIST_256.pickle","rb"))
y_train = pickle.load(open("Splited_Y_Train_Forearm_HIST_256.pickle","rb"))
y_train = to_categorical(y_train,2)
t2=time.time()
print("time to load :",(t2-t1))

t1=time.time()
x_test = pickle.load(open("Splited_X_Test_Forearm_HIST_256.pickle","rb"))
y_test = pickle.load(open("Splited_Y_Test_Forearm_HIST_256.pickle","rb"))
y_test_ROC = y_test
y_test = to_categorical(y_test,2)
t2=time.time()
print("time to load :",(t2-t1))

WEIGHTS_PATH = '../../../../Imagenet Pretrained Weight/VGG - 19/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = '../../../../Imagenet Pretrained Weight/VGG - 19//vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'

def _obtain_input_shape(input_shape,
                        default_size,
                        min_size,
                        data_format,
                        require_flatten,
                        include_top,
                        weights=None):

    if weights != 'imagenet' and input_shape and len(input_shape) == 3:
        if data_format == 'channels_first':
            if input_shape[0] not in {1, 3}:
                warnings.warn(
                    'This model usually expects 1 or 3 input channels. '
                    'However, it was passed an input_shape with ' +
                    str(input_shape[0]) + ' input channels.')
            default_shape = (input_shape[0], default_size, default_size)
        else:
            if input_shape[-1] not in {1, 3}:
                warnings.warn(
                    'This model usually expects 1 or 3 input channels. '
                    'However, it was passed an input_shape with ' +
                    str(input_shape[-1]) + ' input channels.')
            default_shape = (default_size, default_size, input_shape[-1])
    else:
        if data_format == 'channels_first':
            default_shape = (3, default_size, default_size)
        else:
            default_shape = (default_size, default_size, 3)
    if weights == 'imagenet' and require_flatten:
        if input_shape is not None:
            if input_shape != default_shape:
                raise ValueError('When setting`include_top=True` '
                                 'and loading `imagenet` weights, '
                                 '`input_shape` should be ' +
                                 str(default_shape) + '.')
        return default_shape
    if input_shape:
        if data_format == 'channels_first':
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError(
                        '`input_shape` must be a tuple of three integers.')
                if input_shape[0] != 3 and weights == 'imagenet':
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape=' + str(input_shape) + '`')
                if ((input_shape[1] is not None and input_shape[1] < min_size) or
                   (input_shape[2] is not None and input_shape[2] < min_size)):
                    raise ValueError('Input size must be at least ' +
                                     str(min_size) + 'x' + str(min_size) +
                                     '; got `input_shape=' +
                                     str(input_shape) + '`')
        else:
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError(
                        '`input_shape` must be a tuple of three integers.')
                if input_shape[-1] != 3 and weights == 'imagenet':
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape=' + str(input_shape) + '`')
                if ((input_shape[0] is not None and input_shape[0] < min_size) or
                   (input_shape[1] is not None and input_shape[1] < min_size)):
                    raise ValueError('Input size must be at least ' +
                                     str(min_size) + 'x' + str(min_size) +
                                     '; got `input_shape=' +
                                     str(input_shape) + '`')
    else:
        if require_flatten:
            input_shape = default_shape
        else:
            if data_format == 'channels_first':
                input_shape = (3, None, None)
            else:
                input_shape = (None, None, 3)
    if require_flatten:
        if None in input_shape:
            raise ValueError('If `include_top` is True, '
                             'you should specify a static `input_shape`. '
                             'Got `input_shape=' + str(input_shape) + '`')
    return input_shape


def VGG19(include_top=True, weights='imagenet',
          input_tensor=None, input_shape=None,
          pooling=None,
          classes=1000):
   
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')
        
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      require_flatten=True,
                                      include_top=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='vgg19')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models')
        else:
            weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='block5_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    return model

#################################################################################################################


epoch = 50

#set early stopping criteria
pat = 10 #this is the number of epochs with no improvment after which the training will stop
early_stopping = EarlyStopping(monitor='val_loss', patience=pat, verbose=1)
#define the model checkpoint callback -> this will keep on saving the model as a physical file
model_checkpoint = ModelCheckpoint('Forearm_VGG_Model_by_Machine.h5', verbose=1, save_best_only=True)

def final_model():
    img_width, img_height = 256, 256
    model = VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
    
     # Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
    for layer in model.layers[:5]:
        layer.trainable = False
    
    #Adding custom Layers 
    x = model.output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(2, activation="softmax")(x)
    
    # creating the final model 
    model_final = Model(inputs = model.input, output = predictions)
    
    # compile the model 
    model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])    
    return model_final


def fit_and_evaluate(X_Train, X_Test, Y_Train, Y_Test, y_test_ROC):
    
    model = None
    model = final_model()
    
    results = model.fit(X_Train, Y_Train, batch_size=53, epochs = epoch, callbacks=[early_stopping, model_checkpoint], verbose=1, validation_split=0.05)
    model.save('Forearm_VGG19_Model.h5')
    
    ################## Result Evaluation Start ###################
    
    score = model.evaluate(X_Test, Y_Test, batch_size=53)
    print(score)
    
    # Make prediction from model
    predictions = model.predict(X_Test)
    Y_Pred = (predictions > 0.5)
    Y_Test = (Y_Test > 0.5)
    
    
    ###########  ROC Curve Generate    
    
    probs = predictions
    
    # keep probabilities for the positive outcome only
    probs = probs[:, 1]
    
    # calculate AUC
    auc = roc_auc_score(y_test_ROC, probs)
    auc = '%.3f' % auc
    print('AUC: ' +  auc)
    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(y_test_ROC, probs)
    
    pyplot.figure()
    # plot no skill
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange', lw=lw, label='ROC Curve ( Area = ' + auc +' )')
    pyplot.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    
    pyplot.plot(fpr, tpr, marker='.')
    # show the plot
    pyplot.title('Receiver Operating Characteristic ( ROC )')
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.legend()
    pyplot.show()
    pyplot.savefig('ROC_' + auc + '.png')
        
                        ############################################
                        #       Overal: Abnormal + Normal          #
                        ############################################
    Y_Pred_all = Y_Pred
    print(Y_Pred_all.shape)
    Y_Pred_all = Y_Pred_all.reshape(-1,1)
    
    Y_Test_all = Y_Test
    print(Y_Test_all.shape)
    Y_Test_all = Y_Test_all.reshape(-1,1)
    
    file = open("Result_of_Model.txt","w") 
    
    
    con_mat = confusion_matrix(Y_Test_all, Y_Pred_all)
    print("Confusion Matrix => Overal: Abnormal + Normal : \n" + str(con_mat))
    file.write("Confusion Matrix => Overal: Abnormal + Normal : \n" + str(con_mat) + "\n\n")
    
                        ############################################
                        #               Abnormal                   #
                        ############################################
    Y_Pred_abn = Y_Pred
    print(Y_Pred_abn.shape)
    Y_Pred_abn = Y_Pred[:, 0]
    
    Y_Test_abn = Y_Test
    print(Y_Test_abn.shape)
    Y_Test_abn = Y_Test_abn[:, 0]
    
    con_mat = confusion_matrix(Y_Test_abn, Y_Pred_abn)
    print("Confusion Matrix => Abnormal : \n" + str(con_mat))
    file.write("Confusion Matrix => Abnormal : \n" + str(con_mat) + "\n\n")
    
                        ############################################
                        #               Normal                    #
                        ############################################
    Y_Pred_nor = Y_Pred
    print(Y_Pred_nor.shape)
    Y_Pred_nor = Y_Pred[:, 1]
    
    Y_Test_nor = Y_Test
    print(Y_Test_nor.shape)
    Y_Test_nor = Y_Test_nor[:, 1]
    
    con_mat = confusion_matrix(Y_Test_nor, Y_Pred_nor)
    print("Confusion Matrix => Normal : \n" + str(con_mat))
    file.write("Confusion Matrix => Normal : \n" + str(con_mat))
    
    file.close() 
    
    ################## Result Evaluation END ###################
    print('\n\n\n')
    
    return results


model_history = [] 
model_history.append(fit_and_evaluate(x_train, x_test, y_train, y_test, y_test_ROC))

plt.figure()
plt.title('Accuracies vs Epochs On Training Data')
plt.plot(model_history[0].history['acc'], label='Training Data')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.savefig('Accuracies_vs_Epochs_training.png')

plt.figure()
plt.title('Accuracies vs Epochs On Validation Data')
plt.plot(model_history[0].history['val_acc'], label='Validation Data')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.savefig('Accuracies_vs_Epochs_validation.png')

plt.figure()
plt.title('Train Accuracy vs Validation Accuracy')
plt.plot(model_history[0].history['acc'], label='Training Accuracy', color='black')
plt.plot(model_history[0].history['val_acc'], label='Validation Accuracy', color='black', linestyle = "dashdot")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.savefig('Train_vs_Val.png')