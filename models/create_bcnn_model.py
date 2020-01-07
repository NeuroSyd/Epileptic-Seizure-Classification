import keras
import numpy as np
from keras import backend as K
from keras.models import Sequential, Model, Input
from keras.layers import Input, ConvLSTM2D, Lambda, MaxPooling3D
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda, Reshape, Permute
from keras.layers.convolutional import Convolution1D, Convolution2D, MaxPooling1D, MaxPooling2D
from keras.layers.recurrent import GRU, LSTM
from keras.layers.advanced_activations import ELU
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.constraints import max_norm
from numpy import newaxis
from keras.optimizers import SGD,Adam,Adagrad,Adadelta,RMSprop
from utils.customCallbacks import MyEarlyStopping, MyModelCheckpoint
from datetime import datetime
import tensorflow as tf

def outer_product(x):
	"""
	x list of 2 tensors, assuming each of which has shape = (size_minibatch, total_pixels, size_filter)
	"""
	return keras.backend.batch_dot(x[0], x[1], axes=[1,1]) / x[0].get_shape().as_list()[1] 
	#return tf.einsum('bom,bpm->bmop', x[0], x[1])

# loading cnn model
def load_cnn_model(weights_path, X_train_shape, nb_classes):
	from .create_cnn_model import create_cnn_model
	model = create_cnn_model(X_train_shape, nb_classes)
	model.load_weights(weights_path)
	return model

# Create BCNN
def create_bcnn_model(X_train_shape, fold, nb_classes, ds, pp='specific'):
	cnn_weights = './{}_{}cls_{}_saved_models/stft_cnn_{}.hdf5'.format(ds, int(nb_classes), pp, fold)
	cnn_model = load_cnn_model(cnn_weights, X_train_shape, nb_classes)

	x_detector = cnn_model.layers[-8].output
	shape_detector = cnn_model.layers[-8].output_shape
	x_detector = Reshape((shape_detector[1]*shape_detector[2], shape_detector[3]))(x_detector)
	x_extractor = x_detector
	
	x = keras.layers.Lambda(outer_product)([x_detector, x_extractor])
	x = Flatten()(x)
	##
	x = BatchNormalization(axis=-1)(x)
	##
	x = Dropout(0.5)(x)
	x = Dense(256, activation='sigmoid')(x)
	x = Dropout(0.5)(x)
	preds = Dense(nb_classes, activation = 'softmax')(x)                                
	model = Model(inputs=cnn_model.input,outputs=preds)
	
	# make sure none of the CNN or RNN layers are trained, only final FC layer is trained
	for layer in cnn_model.layers:
		layer.trainable = False
	
	return model

def train_bcnn_model(n_folds, nb_classes, ds, pp):
	summary = []
	fold = 1

	for X_train, Y_train, X_val, Y_val, X_test, Y_test in n_folds:
		bcnn_model = create_bcnn_model(X_train.shape, fold, nb_classes, ds, pp)
		bcnn_model.summary()
		adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
		bcnn_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
		class_weights = {}
		for i in range(nb_classes):
			class_weights[i] = X_train.shape[0] / (np.sum(Y_train==i) + 1e-6)

		Y_train = Y_train.astype('uint8')
		Y_train = np_utils.to_categorical(Y_train, nb_classes)
		Y_val = np_utils.to_categorical(Y_val, nb_classes)
		print('Shape: x_train, y_train, X_val, y_val, X_test, y_test')
		print(X_train.shape, Y_train.shape, X_val.shape, Y_val.shape, X_test.shape, Y_test.shape)

		filename = "./{}_{}cls_{}_saved_models/stft_bcnn_{}.hdf5".format(ds, int(nb_classes), pp, fold)

		early_stop = MyEarlyStopping(patience=10, verbose=1)
		checkpointer = MyModelCheckpoint(filename, verbose=1, save_best_only=True)

		bcnn_model.fit(X_train, Y_train, batch_size=32, epochs=200, class_weight=class_weights, 
			validation_data=(X_val, Y_val), callbacks=[early_stop, checkpointer])

		bcnn_model.load_weights(filename)
		predictions = bcnn_model.predict(X_test, verbose=1)
		y_pred = np_utils.to_categorical(np.argmax(predictions, axis=1), nb_classes)
		y_true = np_utils.to_categorical(Y_test, nb_classes)

		from sklearn.metrics import f1_score
		f1_test = f1_score(y_true, y_pred, average='weighted')
		print('Test F1-weighted score is:', f1_test)

		summary.append(f1_test)
		print (summary)

		now = datetime.now()
		date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
		summary.append(date_time)   
		with open('./{}_{}cls_{}_training_history/bcnn_training_history.txt'.format(ds, int(nb_classes), pp), 'w') as f:
			for item in summary:
				f.write("%s\n" % item)
				
		fold += 1