import numpy as np
import collections

def train_val_test_stratified_nfold_split(X, y, val_ratio=0.25, test_ratio=0.2):
	from sklearn.model_selection import StratifiedKFold
	from sklearn.model_selection import train_test_split
	seed = 19
	kfold = StratifiedKFold(n_splits=int(1/test_ratio), shuffle=True, random_state=seed)
	for index, [train, test] in enumerate(kfold.split(X, y)):
		X_train, X_val, y_train, y_val = train_test_split(X[train], y[train], test_size=val_ratio, stratify=y[train], random_state=seed)
		X_test, y_test = X[test], y[test]

		yield X_train, y_train, X_val, y_val, X_test, y_test

def train_val_test_nfold_split(X,y,val_ratio=0.2,test_ratio=0.2):
	'''
	Prepare data for n-fold cross-validation
	:param X:
	:param y:
	:param val_ratio:
	:param test_ratio:
	:return:
	'''

	# shuffle the data
	X_s, y_s = shuffle(X,y)
	n_fold = int(1.0/test_ratio)
	fold_len = int(test_ratio*X_s.shape[0])
	inds_test_sets = []
	for i in range(n_fold):
		inds_test_sets.append(np.arange(i*fold_len,(i+1)*fold_len))
	inds_test_sets = np.array(inds_test_sets)

	for i in range(n_fold):
		inds_fold = np.arange(n_fold)
		inds_test = inds_test_sets[i]
		inds_train = np.concatenate(inds_test_sets[np.delete(inds_fold,i,0)])
		X_test = X_s[inds_test]
		X_train_ = X_s[inds_train]
		X_train = X_train_[:-int(val_ratio*X.shape[0])]
		X_val = X_train_[-int(val_ratio*X.shape[0]):]
		y_test = y_s[inds_test]
		y_train_ = y_s[inds_train]
		y_train = y_train_[:-int(val_ratio*y.shape[0])]
		y_val = y_train_[-int(val_ratio*y.shape[0]):]

		yield X_train, y_train, X_val, y_val, X_test, y_test

def train_val_loo_split(ictal_X, ictal_y, interictal_X, interictal_y, val_ratio):
	'''
	Prepare data for leave-one-out cross-validation
	:param ictal_X:
	:param ictal_y:
	:param interictal_X:
	:param interictal_y:
	:return: (X_train, y_train, X_val, y_val, X_test, y_test)
	'''
	#For each fold, one seizure is taken out for testing, the rest for training
	#Interictal are concatenated and split into N (no. of seizures) parts,
	#each interictal part is combined with one seizure

	nfold = len(ictal_y)

	if isinstance(interictal_y, list):
		interictal_X = np.concatenate(interictal_X,axis=0)
		interictal_y = np.concatenate(interictal_y,axis=0)
	interictal_fold_len = int(round(1.0*interictal_y.shape[0]/nfold))
	print ('interictal_fold_len',interictal_fold_len)

	for i in range(nfold):
		X_test_ictal = ictal_X[i]
		y_test_ictal = ictal_y[i]

		X_test_interictal = interictal_X[i*interictal_fold_len:(i+1)*interictal_fold_len]
		y_test_interictal = interictal_y[i*interictal_fold_len:(i+1)*interictal_fold_len]

		if i==0:
			X_train_ictal = np.concatenate(ictal_X[1:],axis=0)
			y_train_ictal = np.concatenate(ictal_y[1:],axis=0)

			X_train_interictal = interictal_X[(i + 1) * interictal_fold_len + 1:]
			y_train_interictal = interictal_y[(i + 1) * interictal_fold_len + 1:]
		elif i < nfold-1:
			X_train_ictal = np.concatenate(ictal_X[:i] + ictal_X[i + 1:],axis=0)
			y_train_ictal = np.concatenate(ictal_y[:i] + ictal_y[i + 1:],axis=0)

			X_train_interictal = np.concatenate([interictal_X[:i * interictal_fold_len], interictal_X[(i + 1) * interictal_fold_len + 1:]],axis=0)
			y_train_interictal = np.concatenate([interictal_y[:i * interictal_fold_len], interictal_y[(i + 1) * interictal_fold_len + 1:]],axis=0)
		else:
			X_train_ictal = np.concatenate(ictal_X[:i],axis=0)
			y_train_ictal = np.concatenate(ictal_y[:i],axis=0)

			X_train_interictal = interictal_X[:i * interictal_fold_len]
			y_train_interictal = interictal_y[:i * interictal_fold_len]

		print (y_train_ictal.shape,y_train_interictal.shape)

		'''
		Downsampling interictal training set so that the 2 classes
		are balanced
		'''
		down_spl = int(np.floor(y_train_interictal.shape[0]/y_train_ictal.shape[0]))
		if down_spl > 1:
			X_train_interictal = X_train_interictal[::down_spl]
			y_train_interictal = y_train_interictal[::down_spl]
		elif down_spl == 1:
			X_train_interictal = X_train_interictal[:X_train_ictal.shape[0]]
			y_train_interictal = y_train_interictal[:X_train_ictal.shape[0]]

		print ('Balancing:', y_train_ictal.shape,y_train_interictal.shape)

		#X_train_ictal = shuffle(X_train_ictal,random_state=0)
		#X_train_interictal = shuffle(X_train_interictal,random_state=0)
		X_train = np.concatenate((X_train_ictal[:int(X_train_ictal.shape[0]*(1-val_ratio))],X_train_interictal[:int(X_train_interictal.shape[0]*(1-val_ratio))]),axis=0)
		y_train = np.concatenate((y_train_ictal[:int(X_train_ictal.shape[0]*(1-val_ratio))], y_train_interictal[:int(X_train_interictal.shape[0]*(1-val_ratio))]),axis=0)

		X_val = np.concatenate((X_train_ictal[int(X_train_ictal.shape[0]*(1-val_ratio)):],X_train_interictal[int(X_train_interictal.shape[0]*(1-val_ratio)):]),axis=0)
		y_val = np.concatenate((y_train_ictal[int(X_train_ictal.shape[0]*(1-val_ratio)):], y_train_interictal[int(X_train_interictal.shape[0]*(1-val_ratio)):]),axis=0)

		nb_val = X_val.shape[0] - X_val.shape[0]%4
		X_val = X_val[:nb_val]
		y_val = y_val[:nb_val]

		# let overlapped ictal samples have same labels with non-overlapped samples
		y_train[y_train==2] = 1
		y_val[y_val==2] = 1
		y_train[y_train==-1] = 0
		y_val[y_val==-1] = 0

		print (X_test_ictal.shape, X_test_interictal.shape)
		X_test = np.concatenate((X_test_ictal, X_test_interictal),axis=0)
		y_test = np.concatenate((y_test_ictal, y_test_interictal),axis=0)

		# remove overlapped ictal samples in test-set
		X_test = X_test[y_test != 2]
		y_test = y_test[y_test != 2]
		X_test = X_test[y_test != -1]
		y_test = y_test[y_test != -1]

		print ('X_train, X_val, X_test',X_train.shape, X_val.shape, X_test.shape)
		yield (X_train, y_train, X_val, y_val, X_test, y_test)


def train_val_test_split(ictal_X, ictal_y, interictal_X, interictal_y, val_ratio, test_ratio):

	num_sz = len(ictal_y)
	num_sz_test = int(test_ratio*num_sz)
	print ('Total %d seizures. Last %d is used for testing.' %(num_sz, num_sz_test))

	if isinstance(interictal_y, list):
		interictal_X = np.concatenate(interictal_X,axis=0)
		interictal_y = np.concatenate(interictal_y,axis=0)
	interictal_fold_len = int(round(1.0*interictal_y.shape[0]/num_sz))
	print ('interictal_fold_len',interictal_fold_len)


	X_test_ictal = np.concatenate(ictal_X[-num_sz_test:])
	y_test_ictal = np.concatenate(ictal_y[-num_sz_test:])

	X_test_interictal = interictal_X[-num_sz_test*interictal_fold_len:]
	y_test_interictal = interictal_y[-num_sz_test*interictal_fold_len:]

	X_train_ictal = np.concatenate(ictal_X[:-num_sz_test],axis=0)
	y_train_ictal = np.concatenate(ictal_y[:-num_sz_test],axis=0)

	X_train_interictal = interictal_X[:-num_sz_test*interictal_fold_len]
	y_train_interictal = interictal_y[:-num_sz_test*interictal_fold_len]

	print (y_train_ictal.shape,y_train_interictal.shape)

	'''
	Downsampling interictal training set so that the 2 classes
	are balanced
	'''
	down_spl = int(np.floor(y_train_interictal.shape[0]/y_train_ictal.shape[0]))
	if down_spl > 1:
		X_train_interictal = X_train_interictal[::down_spl]
		y_train_interictal = y_train_interictal[::down_spl]
	elif down_spl == 1:
		X_train_interictal = X_train_interictal[:X_train_ictal.shape[0]]
		y_train_interictal = y_train_interictal[:X_train_ictal.shape[0]]

	print ('Balancing:', y_train_ictal.shape,y_train_interictal.shape)

	#X_train_ictal = shuffle(X_train_ictal,random_state=0)
	#X_train_interictal = shuffle(X_train_interictal,random_state=0)
	X_train = np.concatenate((X_train_ictal[:int(X_train_ictal.shape[0]*(1-val_ratio))],X_train_interictal[:int(X_train_interictal.shape[0]*(1-val_ratio))]),axis=0)
	y_train = np.concatenate((y_train_ictal[:int(X_train_ictal.shape[0]*(1-val_ratio))], y_train_interictal[:int(X_train_interictal.shape[0]*(1-val_ratio))]),axis=0)

	X_val = np.concatenate((X_train_ictal[int(X_train_ictal.shape[0]*(1-val_ratio)):],X_train_interictal[int(X_train_interictal.shape[0]*(1-val_ratio)):]),axis=0)
	y_val = np.concatenate((y_train_ictal[int(X_train_ictal.shape[0]*(1-val_ratio)):], y_train_interictal[int(X_train_interictal.shape[0]*(1-val_ratio)):]),axis=0)

	nb_val = X_val.shape[0] - X_val.shape[0]%4
	X_val = X_val[:nb_val]
	y_val = y_val[:nb_val]

	# let overlapped ictal samples have same labels with non-overlapped samples
	y_train[y_train==2] = 1
	y_val[y_val==2] = 1
	y_train[y_train==-1] = 0
	y_val[y_val==-1] = 0

	X_test = np.concatenate((X_test_ictal, X_test_interictal),axis=0)
	y_test = np.concatenate((y_test_ictal, y_test_interictal),axis=0)

	# remove overlapped ictal samples in test-set
	X_test = X_test[y_test != 2]
	y_test = y_test[y_test != 2]
	X_test = X_test[y_test != -1]
	y_test = y_test[y_test != -1]

	print ('X_train, X_val, X_test',X_train.shape, X_val.shape, X_test.shape)
	return X_train, y_train, X_val, y_val, X_test, y_test

def shuffle(X,y):
	np.random.seed(2019)
	ind = X.shape[0]
	inds = np.arange(ind)
	np.random.shuffle(inds)
	return X[inds], y[inds]

