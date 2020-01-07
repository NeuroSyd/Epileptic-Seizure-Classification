import argparse
import numpy as np
from utils.prep_data import train_val_test_nfold_split, train_val_test_stratified_nfold_split
from models.create_rnn_model import train_rnn_model
from models.create_cnn_model import train_cnn_model
from models.create_hybrid_model import train_hybrid_model
from models.create_bcnn_model import train_bcnn_model
from models.create_brnn_model import train_brnn_model

def class_integer_encode(y):
	_, integer_indices = np.unique(y, return_inverse=True)
	integer_indices = integer_indices.reshape(len(integer_indices), 1)

	print('length of encoded class integer vector: {}'.format(len(integer_indices[0])))
	return integer_indices

def prep_data(nb_classes, ds, pp):
	if ds == 'tuh':
		if pp == 'adaptive':
			# pnt_path = '/mnt/data7_M2/Tennison_TUH_Reprocessed_STFT/stft_1s_64_adaptive/'
			pnt_path = '/mnt/data7_M2/Tennison_TUH_Reprocessed_STFT/stft_1s_64_adaptive_cont/'
		elif pp == 'continuous':
			pass
		else:
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
		if pp == 'adaptive':
			pnt_path = '/mnt/data7_M2/epilepsia_data_adaptive/stft_data/'
		elif pp == 'continuous':
			pass
		else:
			pnt_path = '/mnt/data7_M2/epilepsia_data/stft_data/'
		X = np.load(pnt_path + 'stft_x.npy') # Epilepsia
		y = np.load(pnt_path + 'stft_y.npy')
		print('x shape: {}'.format(X.shape))
		print('y shape: {}'.format(y.shape))

	print('number of unique y values: {}'.format(np.unique(y)))

	y = class_integer_encode(y)
	n_folds = train_val_test_stratified_nfold_split(X, y)
	return n_folds

def main(model, nb_classes, ds, pp):
	call_dict = {
	'cnn': train_cnn_model,
	'rnn': train_rnn_model,
	'bcnn': train_bcnn_model,
	'brnn': train_brnn_model,
	'hybrid': train_hybrid_model}

	for mod in model:
		n_folds = prep_data(nb_classes, ds, pp)
		call_dict[mod](n_folds, nb_classes, ds, pp)

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument('dataset', choices = ['tuh', 'epi'], help='dataset to use')
	ap.add_argument('model', choices = ['cnn', 'rnn', 'bcnn', 'brnn', 'hybrid'], nargs = '+',
					 help='model to be trained')
	ap.add_argument('nb_classes', help='number of classes')
	ap.add_argument('preprocessing', choices = ['adaptive', 'continuous', 'specific'], help='preprocessing technique')

	args = ap.parse_args()
	model = args.model
	nb_classes = int(args.nb_classes)
	ds = args.dataset
	pp = args.preprocessing

	print('Training model(s) {} using dataset {} with {} classes and preprocessing method {}'.format(model, ds, nb_classes, pp))
	main(model, nb_classes, ds, pp)


