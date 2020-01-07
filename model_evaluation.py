import argparse
import numpy as np
from utils.prep_data import train_val_test_nfold_split, train_val_test_stratified_nfold_split
from models.create_hybrid_model import create_hybrid_model
from models.create_bcnn_model import create_bcnn_model
from models.create_brnn_model import create_brnn_model
from models.create_cnn_model import create_cnn_model
from models.create_rnn_model import create_rnn_model
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix, f1_score
import collections
# import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def plot_save_cm(cm, model, plt_fname):
	'''
	Pass Normalised CM
	'''
	cm = cm.round(3)
	plt.figure(figsize = (10,7))
	ax = sns.heatmap(cm, annot=True, cmap="Blues")
	bottom, top = ax.get_ylim()
	ax.set_ylim(bottom + 0.5, top - 0.5)
	plt.ylabel('True Label', fontsize=16)
	plt.xlabel('\nPredicted Label', fontsize=16)
	plt.xticks(rotation=45, fontsize=12)
	plt.yticks(rotation=0, fontsize=12)
	plt.title('{} Confusion Matrix\n'.format(model.capitalize()), fontsize = 18)
	plt.savefig(plt_fname, dpi=1200)

def export_cm(cm, norm_cm, ds, model, nb_classes, class_map):
	import  csv
	df_cm = pd.DataFrame(cm, index = class_map, columns = class_map)
	df_norm_cm = pd.DataFrame(norm_cm, index = class_map, columns = class_map)

	csv_fname = "./model_evaluation/{}_{}_{}.csv".format(ds, model, int(nb_classes))
	csv_fname_norm = "./model_evaluation/{}_{}_{}_norm.csv".format(ds, model, int(nb_classes))
	plt_fname = "./model_evaluation/{}_{}_{}.pdf".format(ds, model, int(nb_classes))
	df_cm.to_csv(csv_fname)
	df_norm_cm.to_csv(csv_fname_norm)

	### Comment out to save plot, I cannot install seaborn
	# plot_save_cm(df_norm_cm, model, plt_fname)

def evaluate_hybrid_model(n_folds, nb_classes, ds, class_map, pp='specific'):
	total_pred = []
	total_true = []
	fold = 1
	for X_train, Y_train, X_val, Y_val, X_test, Y_test in n_folds:
		hybrid_model = create_hybrid_model(X_train.shape, fold, nb_classes, ds)
		filename = "./{}_{}cls_{}_saved_models/stft_hybrid_{}.hdf5".format(ds, int(nb_classes), pp, fold)
		hybrid_model.load_weights(filename)
		predictions = hybrid_model.predict([X_test, X_test], verbose=1)
		y_pred = np.argmax(predictions, axis=1)
		y_true = Y_test
		total_pred.extend(y_pred)
		total_true.extend(y_true)
		fold += 1
	cm = confusion_matrix(total_true, total_pred)
	norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	print('Results for Hybrid')
	print(cm)
	print(norm_cm)
	print(f1_score(y_true, y_pred, average=None))
	export_cm(cm, norm_cm, ds, 'hybrid', nb_classes, class_map)

def evaluate_bcnn_model(n_folds, nb_classes, ds, class_map, pp='specific'):
	total_pred = []
	total_true = []
	fold = 1
	for X_train, Y_train, X_val, Y_val, X_test, Y_test in n_folds:
		bcnn_model = create_bcnn_model(X_train.shape, fold, nb_classes, ds)
		filename = "./{}_{}cls_{}_saved_models/stft_bcnn_{}.hdf5".format(ds, int(nb_classes), pp, fold)
		bcnn_model.load_weights(filename)
		predictions = bcnn_model.predict(X_test, verbose=1)
		y_pred = np.argmax(predictions, axis=1)
		y_true = Y_test
		total_pred.extend(y_pred)
		total_true.extend(y_true)
		fold += 1
	cm = confusion_matrix(total_true, total_pred)
	norm_cm = cm / cm.astype(np.float).sum(axis=1)
	print('Results for B-CNN')
	print(cm)
	print(norm_cm)
	print(f1_score(y_true, y_pred, average=None))
	export_cm(cm, norm_cm, ds, 'bcnn', nb_classes, class_map)

def evaluate_brnn_model(n_folds, nb_classes, ds, class_map, pp='specific'):
	total_pred = []
	total_true = []
	fold = 1
	for X_train, Y_train, X_val, Y_val, X_test, Y_test in n_folds:
		brnn_model = create_brnn_model(X_train.shape, fold, nb_classes, ds)
		filename = "./{}_{}cls_{}_saved_models/stft_brnn_{}.hdf5".format(ds, int(nb_classes), pp, fold)
		brnn_model.load_weights(filename)
		predictions = brnn_model.predict(X_test, verbose=1)
		y_pred = np.argmax(predictions, axis=1)
		y_true = Y_test
		total_pred.extend(y_pred)
		total_true.extend(y_true)
		fold += 1
	cm = confusion_matrix(total_true, total_pred)
	norm_cm = cm / cm.astype(np.float).sum(axis=1)
	print('Results for B-RNN')
	print(cm)
	print(norm_cm)
	print(f1_score(y_true, y_pred, average=None))
	export_cm(cm, norm_cm, ds, 'brnn', nb_classes, class_map)

def evaluate_rnn_model(n_folds, nb_classes, ds, class_map, pp='specific'):
	total_pred = []
	total_true = []
	fold = 1
	for X_train, Y_train, X_val, Y_val, X_test, Y_test in n_folds:
		rnn_model = create_rnn_model(X_train.shape, nb_classes)
		filename = "./{}_{}cls_{}_saved_models/stft_rnn_{}.hdf5".format(ds, int(nb_classes), pp, fold)
		rnn_model.load_weights(filename)
		predictions = rnn_model.predict(X_test, verbose=1)
		y_pred = np.argmax(predictions, axis=1)
		y_true = Y_test
		total_pred.extend(y_pred)
		total_true.extend(y_true)
		fold += 1
	cm = confusion_matrix(total_true, total_pred)
	norm_cm = cm / cm.astype(np.float).sum(axis=1)
	print('Results for RNN')
	print(cm)
	print(norm_cm)
	print(f1_score(y_true, y_pred, average=None))
	export_cm(cm, norm_cm, ds, 'rnn', nb_classes, class_map)

def evaluate_cnn_model(n_folds, nb_classes, ds, class_map, pp='specific'):
	total_pred = []
	total_true = []
	fold = 1
	for X_train, Y_train, X_val, Y_val, X_test, Y_test in n_folds:
		cnn_model = create_cnn_model(X_train.shape, nb_classes)
		filename = "./{}_{}cls_{}_saved_models/stft_cnn_{}.hdf5".format(ds, int(nb_classes), pp, fold)
		cnn_model.load_weights(filename)
		predictions = cnn_model.predict(X_test, verbose=1)
		y_pred = np.argmax(predictions, axis=1)
		y_true = Y_test
		total_pred.extend(y_pred)
		total_true.extend(y_true)
		fold += 1
	cm = confusion_matrix(total_true, total_pred)
	norm_cm = cm / cm.astype(np.float).sum(axis=1)
	print('Results for CNN')
	print(cm)
	print(norm_cm)
	print(f1_score(y_true, y_pred, average=None))
	export_cm(cm, norm_cm, ds, 'cnn', nb_classes, class_map)

def prep_data(nb_classes, ds):
	if ds == 'tuh':
		pnt_path = '/mnt/data7_M2/Tennison_TUH_Reprocessed_STFT/stft_1s_64/'

		X = np.load(pnt_path + 'data_x.npy')
		y = np.load(pnt_path + 'data_y.npy')
		print('x shape: {}'.format(X.shape))
		print('y shape: {}'.format(y.shape))
		if nb_classes == 7:
			X = X[y!='MYSZ']
			y = y[y!='MYSZ']
			print ('x shape after removing MYSZ: {}'.format(X.shape))
			print ('y shape after removing MYSZ: {}'.format(y.shape))
	else:
		pnt_path = '/mnt/data7_M2/epilepsia_data/stft_data/'
		X = np.load(pnt_path + 'stft_x.npy') # Epilepsia
		y = np.load(pnt_path + 'stft_y.npy')
		print('x shape: {}'.format(X.shape))
		print('y shape: {}'.format(y.shape))

	print('number of unique y values: {}'.format(np.unique(y)))

	y, class_map = class_integer_encode(y)
	n_folds = train_val_test_stratified_nfold_split(X, y)
	return n_folds, class_map

def class_integer_encode(y):
	u, integer_indices = np.unique(y, return_inverse=True)
	integer_indices = integer_indices.reshape(len(integer_indices), 1)
	class_to_categorical = find_mapping(y, integer_indices)

	print(collections.Counter(y))

	print(class_to_categorical)
	return integer_indices, class_to_categorical

def find_mapping(y, integer_indices):
	class_to_categorical = {}
	for i in np.unique(y):
		index = np.where(y==i)[0][0]
		categorical = integer_indices[index][0]
		class_to_categorical[i] = categorical
	return class_to_categorical

def main(model, nb_classes, ds):
	call_dict = {
	'cnn': evaluate_cnn_model,
	'rnn': evaluate_rnn_model,
	'bcnn': evaluate_bcnn_model,
	'brnn': evaluate_brnn_model,
	'hybrid': evaluate_hybrid_model}

	for mod in model:
		n_folds, class_map = prep_data(nb_classes, ds)
		call_dict[mod](n_folds, nb_classes, ds, class_map)

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument('dataset', choices = ['tuh', 'epi'], help='dataset to use')
	ap.add_argument('model', choices = ['cnn', 'rnn', 'bcnn', 'brnn', 'hybrid'], nargs = '+',
					 help='model to be evaluated')
	ap.add_argument('nb_classes', help='number of classes')

	args = ap.parse_args()
	model = args.model
	nb_classes = int(args.nb_classes)
	ds = args.dataset

	print('Evaluating model(s) {} using dataset {} with {} classes'.format(model, ds, nb_classes))
	main(model, nb_classes, ds)

