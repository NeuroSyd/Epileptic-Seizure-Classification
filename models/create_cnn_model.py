import keras
import numpy as np
from keras import backend as K
from keras.models import Sequential, Model, Input
from keras.layers import Input, ConvLSTM2D, Lambda, MaxPooling3D
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda, Reshape, Permute
from keras.layers.convolutional import Convolution1D, Convolution2D, Convolution3D, MaxPooling1D, MaxPooling2D
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

def create_cnn_model(X_train_shape, nb_classes):

	inputs = Input(shape=X_train_shape[1:])

	normal1 = BatchNormalization(axis=-1)(inputs)
	reshape1 = Lambda(lambda x: keras.backend.expand_dims(x, axis=-1))(normal1)
	conv1 = Convolution3D(
		32, (3 ,3, X_train_shape[-1]), data_format = 'channels_last',
		padding='valid', strides=(1,1,1))(reshape1)

	reshape2 = Lambda(lambda x: keras.backend.squeeze(x, axis=-2))(conv1)
	
	# conv1 = Convolution2D(
	# 	32, (3 ,3), data_format = 'channels_last',
	# 	padding='valid', strides=(1,1),
	# 	name='conv1')(normal1)
	relu1 = Activation('relu')(reshape2)
	pool1 = MaxPooling2D(pool_size=(2, 1), data_format = 'channels_last')(relu1)

	normal2 = BatchNormalization(axis=-1)(pool1)

	conv2 = Convolution2D(
		64, (3, 3), data_format = 'channels_last',
		padding='valid', strides=(1,1))(normal2)
	relu2 = Activation('relu')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 1), data_format = 'channels_last')(relu2)

	normal3 = BatchNormalization(axis=-1)(pool2)

	
	conv3 = Convolution2D(
		64, (3, 3), data_format = 'channels_last',
		padding='valid', strides=(1,1))(normal3)
	relu3 = Activation('relu')(conv3)
	# pool3 = MaxPooling2D(pool_size=(2, 1), data_format = 'channels_last')(relu3)
	
	flat = Flatten()(relu3)
	drop1 = Dropout(0.5)(flat)
	dens1 = Dense(256, activation='sigmoid')(drop1)
	drop2 = Dropout(0.5)(dens1)
	dens2 = Dense(nb_classes)(drop2)
	# option to include temperature in softmax

	temp = 1.0
	temperature = Lambda(lambda x: x / temp)(dens2)
	last = Activation('softmax')(temperature)

	model = Model(input=inputs, output=last)
	return model

def train_cnn_model(n_folds, nb_classes, ds, pp):
	summary = []
	fold = 1

	for X_train, Y_train, X_val, Y_val, X_test, Y_test in n_folds:
		cnn_model = create_cnn_model(X_train.shape, nb_classes)
		cnn_model.summary()
		adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
		cnn_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
		class_weights = {}
		for i in range(nb_classes):
			class_weights[i] = X_train.shape[0] / (np.sum(Y_train==i) + 1e-6)

		Y_train = Y_train.astype('uint8')
		Y_train = np_utils.to_categorical(Y_train, nb_classes)
		Y_val = np_utils.to_categorical(Y_val, nb_classes)
		print('Shape: x_train, y_train, X_val, y_val, X_test, y_test')
		print(X_train.shape, Y_train.shape, X_val.shape, Y_val.shape, X_test.shape, Y_test.shape)

		filename = "./{}_{}cls_{}_saved_models/stft_cnn_{}.hdf5".format(ds, int(nb_classes), pp, fold)

		early_stop = MyEarlyStopping(patience=10, verbose=1)
		checkpointer = MyModelCheckpoint(filename, verbose=1, save_best_only=True)

		cnn_model.fit(X_train, Y_train, batch_size=32, epochs=200, class_weight=class_weights,
					   validation_data=(X_val,Y_val), callbacks=[early_stop,checkpointer])

		cnn_model.load_weights(filename)
		predictions = cnn_model.predict(X_test, verbose=1)
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
		with open('./{}_{}cls_{}_training_history/cnn_training_history.txt'.format(ds, int(nb_classes), pp), 'w') as f:
			for item in summary:
				f.write("%s\n" % item)

		fold += 1