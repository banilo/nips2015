"""
HCP: Semi-supervised network decomposition by low-rank logistic regression
"""

print __doc__

import os
import os.path as op
import numpy as np
import glob
from scipy.linalg import norm
import nibabel as nib
from sklearn.grid_search import RandomizedSearchCV
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from nilearn.input_data import NiftiMasker
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import theano
import theano.tensor as T
from matplotlib import pylab as plt
print('Running THEANO on %s' % theano.config.device)
from nilearn.image import concat_imgs
import joblib
import time

RES_NAME = 'nips3mm_serial'
WRITE_DIR = op.join(os.getcwd(), RES_NAME)
if not op.exists(WRITE_DIR):
    os.mkdir(WRITE_DIR)

##############################################################################
# load+preprocess data
##############################################################################

mask_img = 'grey10_icbm_3mm_bin.nii.gz'
nifti_masker = NiftiMasker(mask_img=mask_img, smoothing_fwhm=False,
                           standardize=False)
nifti_masker.fit()
mask_nvox = nifti_masker.mask_img_.get_data().sum()

print('Loading data...')
X_task, labels = joblib.load('preload_HT_3mm')

labels = np.int32(labels)

# contrasts are IN ORDER -> shuffle!
new_inds = np.arange(0, X_task.shape[0])
np.random.shuffle(new_inds)
X_task = X_task[new_inds]
y = labels[new_inds]

from sklearn.cross_validation import StratifiedShuffleSplit
folder = StratifiedShuffleSplit(y, n_iter=1, test_size=0.50)
new_devs, inds_val = iter(folder).next()
X_dev, X_val = X_task[new_devs], X_task[inds_val]
y_dev, y_val = y[new_devs], y[inds_val]

del X_task

##############################################################################
# class defintion
##############################################################################

class AutoEncoder(BaseEstimator):
    def __init__(self, n_hidden, gain1, learning_rate, max_epochs=500):
        self.n_hidden = n_hidden
        self.gain1 = gain1
        self.max_epochs = max_epochs
        self.learning_rate = np.float32(learning_rate)

    # def rectify(X):
    #     return T.maximum(0., X)

    from theano.tensor.shared_randomstreams import RandomStreams

    def RMSprop(self, cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
        grads = T.grad(cost=cost, wrt=params)
        updates = []
        for p, g in zip(params, grads):
            acc = theano.shared(p.get_value() * 0.)
            acc_new = rho * acc + (1 - rho) * g ** 2
            gradient_scaling = T.sqrt(acc_new + epsilon)
            g = g / gradient_scaling
            updates.append((acc, acc_new))
            updates.append((p, p - lr * g))
        return updates
        
    def get_param_pool(self):
        cur_params = (
            self.W0s, self.W1s, self.bW0s, self.bW1s
        )
        return cur_params
        

    def fit(self, X_rest):
        DEBUG_FLAG = True

        # self.max_epochs = 333
        self.batch_size = 100
        n_input = X_rest.shape[1]  # sklearn-like structure
        n_output = n_input
        rng = np.random.RandomState(42)
        self.params_from_last_iters = []
        
        self.input_restdata = T.matrix(dtype='float32', name='input_restdata')

        index = T.iscalar(name='index')
        
        # prepare data for theano computation
        if DEBUG_FLAG:
            self.dbg_ae_cost_ = list()
            self.dbg_ae_nonimprovesteps = list()
        X_rest_s = theano.shared(value=np.float32(X_rest), name='X_rest_s')
        ae_train_samples = len(X_rest)

        # W -> unsupervised / auto-encoder

        # computational graph: auto-encoder
        W0_vals = rng.randn(n_input, self.n_hidden).astype(np.float32) * self.gain1

        self.W0s = theano.shared(W0_vals)
        self.W1s = self.W0s.T  # tied
        bW0_vals = np.zeros(self.n_hidden).astype(np.float32)
        self.bW0s = theano.shared(value=bW0_vals, name='bW0')
        bW1_vals = np.zeros(n_output).astype(np.float32)
        self.bW1s = theano.shared(value=bW1_vals, name='bW1')

        givens_ae = {
            self.input_restdata: X_rest_s[
                index * self.batch_size:(index + 1) * self.batch_size]
        }

        encoding = (self.input_restdata.dot(self.W0s) + self.bW0s).dot(self.W1s) + self.bW1s

        self.ae_loss = T.sum((self.input_restdata - encoding) ** 2, axis=1)

        self.ae_cost = (
            T.mean(self.ae_loss) / n_input
        )

        # combined loss for AE and LR
        ae_params = [self.W0s, self.bW0s, self.bW1s]
        ae_updates = self.RMSprop(
            cost=self.ae_cost,
            params=ae_params,
            lr=self.learning_rate)
        f_train_ae = theano.function(
            [index],
            [self.ae_cost],
            givens=givens_ae,
            updates=ae_updates, allow_input_downcast=True)

        # optimization loop
        start_time = time.time()
        ae_last_cost = np.inf
        no_improve_steps = 0
        acc_train, acc_val = 0., 0.
        for i_epoch in range(self.max_epochs):
            if i_epoch == 1:
                epoch_dur = time.time() - start_time
                total_mins = (epoch_dur * self.max_epochs) / 60
                hs, mins = divmod(total_mins, 60)
                print("Max estimated duration: %i hours and %i minutes" % (hs, mins))

            # AE
            # if i_epoch % 2 == 0:  # every second time
            #if False:
                # auto-encoder
            ae_n_batches = ae_train_samples // self.batch_size
            # for i in range(lr_n_batches):
            # for i in range(max(ae_n_batches, lr_n_batches)):
                # if i < ae_n_batches:
                #     ae_cur_cost = float(f_train_ae(i)[0])
                # ae_cur_cost = 0
                # if i < lr_n_batches:
                #     lr_cur_cost = float(f_train_lr(i)[0])
            # for i in range(lr_n_batches):
            for i in range(ae_n_batches):
                # lr_cur_cost = f_train_lr(i)[0]
                # ae_cur_cost = lr_cur_cost
                ae_cur_cost = f_train_ae(i)[0]

            # evaluate epoch cost
            if ae_last_cost - ae_cur_cost < 0.1:
                no_improve_steps += 1
            else:
                ae_last_cost = ae_cur_cost
                no_improve_steps = 0

            print('E:%i, ae_cost:%.4f, adsteps:%i' % (
                i_epoch + 1, ae_cur_cost, no_improve_steps))

            if (i_epoch % 10 == 0):
                self.dbg_ae_cost_.append(ae_cur_cost)
                self.dbg_ae_nonimprovesteps.append(no_improve_steps)

            # save paramters from last 100 iterations
            if i_epoch > -1: # (self.max_epochs - 100):
                print('Param pool!')
                param_pool = self.get_param_pool()
                self.params_from_last_iters.append(param_pool)

        total_mins = (time.time() - start_time) / 60
        hs, mins = divmod(total_mins, 60)
        print("Final duration: %i hours and %i minutes" % (hs, mins))

        return self

    def transform(self, newdata):
        compr_matrix = self.W0s.get_value().T  # currently best compression
        transformed_data = np.dot(compr_matrix, newdata.T).T
        return transformed_data


class SEncoder(BaseEstimator):
    def __init__(self, gain1, learning_rate, max_epochs=100,
                 l1=0.1, l2=0.1):
        """
        Parameters
        ----------
        lambda : float
            Mediates between AE and LR. lambda==1 equates with LR only.
        """
        self.gain1 = gain1
        self.max_epochs = max_epochs
        self.learning_rate = np.float32(learning_rate)
        self.penalty_l1 = np.float32(l1)
        self.penalty_l2 = np.float32(l2)

    # def rectify(X):
    #     return T.maximum(0., X)

    from theano.tensor.shared_randomstreams import RandomStreams

    def RMSprop(self, cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
        grads = T.grad(cost=cost, wrt=params)
        updates = []
        for p, g in zip(params, grads):
            acc = theano.shared(p.get_value() * 0.)
            acc_new = rho * acc + (1 - rho) * g ** 2
            gradient_scaling = T.sqrt(acc_new + epsilon)
            g = g / gradient_scaling
            updates.append((acc, acc_new))
            updates.append((p, p - lr * g))
        return updates
        
    def get_param_pool(self):
        cur_params = (
            self.V0s, self.bV0
        )
        return cur_params
        
    def fit(self, X_task, y):
        DEBUG_FLAG = True

        # self.max_epochs = 333
        self.batch_size = 100
        rng = np.random.RandomState(42)
        self.input_taskdata = T.matrix(dtype='float32', name='input_taskdata')
        self.input_restdata = T.matrix(dtype='float32', name='input_restdata')
        self.params_from_last_iters = []
        n_input = X_task.shape[1]

        index = T.iscalar(name='index')
        
        # prepare data for theano computation
        if not DEBUG_FLAG:
            X_train_s = theano.shared(
                value=np.float32(X_task), name='X_train_s')
            y_train_s = theano.shared(
                value=np.int32(y), name='y_train_s')
            lr_train_samples = len(X_task)
        else:
            from sklearn.cross_validation import StratifiedShuffleSplit
            folder = StratifiedShuffleSplit(y, n_iter=1, test_size=0.20)
            new_trains, inds_val = iter(folder).next()
            X_train, X_val = X_task[new_trains], X_task[inds_val]
            y_train, y_val = y[new_trains], y[inds_val]

            X_train_s = theano.shared(value=np.float32(X_train),
                                      name='X_train_s', borrow=False)
            y_train_s = theano.shared(value=np.int32(y_train),
                                      name='y_train_s', borrow=False)
            # X_val_s = theano.shared(value=np.float32(X_val),
            #                         name='X_train_s', borrow=False)
            # y_val_s = theano.shared(value=np.int32(y_val),
            #                         name='y_cal_s', borrow=False)
            lr_train_samples = len(X_train)
            self.dbg_epochs_ = list()
            self.dbg_acc_train_ = list()
            self.dbg_acc_val_ = list()
            self.dbg_ae_cost_ = list()
            self.dbg_lr_cost_ = list()
            self.dbg_ae_nonimprovesteps = list()
            self.dbg_acc_other_ds_ = list()
            self.dbg_prfs_ = list()
            self.dbg_prfs_other_ds_ = list()

        # computation graph: logistic regression
        clf_n_output = 18  # number of labels
        my_y = T.ivector(name='y')

        bV0_vals = np.zeros(clf_n_output).astype(np.float32)
        self.bV0 = theano.shared(value=bV0_vals, name='bV0')

        V0_vals = rng.randn(n_input, clf_n_output).astype(np.float32) * self.gain1
        self.V0s = theano.shared(V0_vals)

        self.p_y_given_x = T.nnet.softmax(
            T.dot(self.input_taskdata, self.V0s) + self.bV0
        )
        self.lr_cost = -T.mean(T.log(self.p_y_given_x)[T.arange(my_y.shape[0]), my_y])
        self.lr_cost = (
            self.lr_cost +
            T.mean(abs(self.V0s)) * self.penalty_l1 +
            T.mean(abs(self.bV0)) * self.penalty_l1 +

            T.mean((self.V0s ** np.float32(2))) * self.penalty_l2 +
            T.mean((self.bV0 ** np.float32(2))) * self.penalty_l2
        )
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        givens_lr = {
            self.input_taskdata: X_train_s[index * self.batch_size:(index + 1) * self.batch_size],
            my_y: y_train_s[index * self.batch_size:(index + 1) * self.batch_size]
        }

        params = [self.V0s, self.bV0]
        updates = self.RMSprop(cost=self.lr_cost, params=params,
                               lr=self.learning_rate)

        f_train_lr = theano.function(
            [index],
            [self.lr_cost],
            givens=givens_lr,
            updates=updates)

        # optimization loop
        start_time = time.time()
        lr_last_cost = np.inf
        ae_cur_cost = np.inf
        no_improve_steps = 0
        acc_train, acc_val = 0., 0.
        for i_epoch in range(self.max_epochs):
            if i_epoch == 1:
                epoch_dur = time.time() - start_time
                total_mins = (epoch_dur * self.max_epochs) / 60
                hs, mins = divmod(total_mins, 60)
                print("Max estimated duration: %i hours and %i minutes" % (hs, mins))

            lr_n_batches = lr_train_samples // self.batch_size
            for i in range(lr_n_batches):
                lr_cur_cost = f_train_lr(i)[0]

            # evaluate epoch cost
            if lr_last_cost - lr_cur_cost < 0.1:
                no_improve_steps += 1
            else:
                lr_last_cost = lr_cur_cost
                no_improve_steps = 0

            # logistic
            lr_last_cost = lr_cur_cost
            acc_train = self.score(X_train, y_train)
            acc_val, prfs_val = self.score(X_val, y_val, return_prfs=True)

            print('E:%i, ae_cost:%.4f, lr_cost:%.4f, train_score:%.2f, vald_score:%.2f, ae_badsteps:%i' % (
                i_epoch + 1, ae_cur_cost, lr_cur_cost, acc_train, acc_val, no_improve_steps))

            if (i_epoch % 10 == 0):
                self.dbg_ae_cost_.append(ae_cur_cost)
                self.dbg_lr_cost_.append(lr_cur_cost)

                self.dbg_epochs_.append(i_epoch + 1)
                self.dbg_ae_nonimprovesteps.append(no_improve_steps)
                self.dbg_acc_train_.append(acc_train)
                self.dbg_acc_val_.append(acc_val)
                self.dbg_prfs_.append(prfs_val)
                
            # if i_epoch > (self.max_epochs - 100):
            param_pool = self.get_param_pool()
            self.params_from_last_iters.append(param_pool)

        total_mins = (time.time() - start_time) / 60
        hs, mins = divmod(total_mins, 60)
        print("Final duration: %i hours and %i minutes" % (hs, mins))

        return self

    def predict(self, X):
        X_test_s = theano.shared(value=np.float32(X), name='X_test_s', borrow=True)

        givens_te = {
            self.input_taskdata: X_test_s
        }

        f_test = theano.function(
            [],
            [self.y_pred],
            givens=givens_te)
        predictions = f_test()
        del X_test_s
        del givens_te
        return predictions[0]

    def score(self, X, y, return_prfs=False):
        pred_y = self.predict(X)
        acc = np.mean(pred_y == y)
        prfs = precision_recall_fscore_support(pred_y, y)
        if return_prfs:
            return acc, prfs
        else:
            return acc

##############################################################################
# data processing
##############################################################################

n_comps = [5, 20, 50, 100]
for n_comp in n_comps:
    print('#' * 80)
    print('#components: %i' % n_comp)
    
    # 2-step approach 1: ICA + LR
    # print('Compressing...')
    # from sklearn.decomposition import FastICA
    # compressor = FastICA(n_components=n_comp, whiten=True)
    # compressor.fit(X_dev)
    # half2compr = compressor.transform(X_val)
    # 
    # print('Classifiying...')
    # l1 = 0.1
    # l2 = 0.1
    # my_title = r'LR: L1=%.1f L2=%.1f res=3mm' % (
    #     l1, l2
    # )
    # print(my_title)
    # estimator = SEncoder(
    #     gain1=0.004,  # empirically determined by CV
    #     learning_rate = np.float32(0.00001),  # empirically determined by CV,
    #     max_epochs=500, l1=l1, l2=l2)
    # 
    # estimator.fit(half2compr, y_val)
    # 
    # acc_ICA = estimator.dbg_acc_val_
    # print(acc_ICA)
    # print('ICA: %.4f' % acc_ICA[-1])
    # 
    # outpath = op.join(WRITE_DIR, 'ICA-LR_ncomp=%i' % n_comp)
    # np.save(outpath + '_acc', acc_ICA)
    # joblib.dump(estimator, outpath + '_est', compress=9)

    # 2-step approach: SPCA + LR
    # print('Compressing...')
    # from sklearn.decomposition import SparsePCA
    # compressor = SparsePCA(n_components=n_comp, alpha=1.0,  # big sparsity
    #                        n_jobs=1, verbose=0, tol=0.1)
    # compressor.fit(X_dev)
    # half2compr = compressor.transform(X_val)
    # 
    # print('Classifiying...')
    # l1 = 0.1
    # l2 = 0.1
    # my_title = r'LR: L1=%.1f L2=%.1f res=3mm' % (
    #     l1, l2
    # )
    # print(my_title)
    # estimator = SEncoder(
    #     gain1=0.004,  # empirically determined by CV
    #     learning_rate = np.float32(0.00001),  # empirically determined by CV,
    #     max_epochs=500, l1=l1, l2=l2)
    # 
    # estimator.fit(half2compr, y_val)
    # 
    # acc_SPCA = estimator.dbg_acc_val_
    # print(acc_SPCA)
    # print('SPCA: %.4f' % acc_SPCA[-1])
    # 
    # outpath = op.join(WRITE_DIR, 'SPCA-LR_ncomp=%i' % n_comp)
    # np.save(outpath + '_acc', acc_SPCA)
    # joblib.dump(estimator, outpath + '_est', compress=9)

    # 2-step approach 3: AE + LR
    # print('Compressing by autoencoder...')
    # compressor = AutoEncoder(
    #         n_hidden=n_comp,
    #         gain1=0.004,  # empirically determined by CV
    #         learning_rate = np.float32(0.00001),  # empirically determined by CV,
    #         max_epochs=500)
    # compressor.fit(X_dev)
    # 
    # half2compr = compressor.transform(X_val)
    # 
    # print('Classifiying...')
    # l1 = 0.1
    # l2 = 0.1
    # my_title = r'LR: L1=%.1f L2=%.1f res=3mm' % (
    #     l1, l2
    # )
    # print(my_title)
    # estimator = SEncoder(
    #     gain1=0.004,  # empirically determined by CV
    #     learning_rate = np.float32(0.00001),  # empirically determined by CV,
    #     max_epochs=500, l1=l1, l2=l2)
    # 
    # estimator.fit(half2compr, y_val)
    # 
    # acc_AE = estimator.dbg_acc_val_
    # print(acc_AE)
    # print('AE: %.4f' % acc_AE[-1])
    # 
    # outpath = op.join(WRITE_DIR, 'AE-LR_ncomp=%i' % n_comp)
    # np.save(outpath + '_acc', acc_AE)
    # joblib.dump(estimator, outpath + '_est', compress=9)

    # 2-step approach 4: PCA + LR
    from sklearn.decomposition import PCA
    print('Compressing by whitened PCA...')
    compressor = PCA(n_components=n_comp, whiten=True)
    compressor.fit(X_dev)

    half2compr = compressor.transform(X_val)

    print('Classifiying...')
    l1 = 0.1
    l2 = 0.1
    my_title = r'LR: L1=%.1f L2=%.1f res=3mm' % (
        l1, l2
    )
    print(my_title)
    estimator = SEncoder(
        gain1=0.004,  # empirically determined by CV
        learning_rate = np.float32(0.00001),  # empirically determined by CV,
        max_epochs=500, l1=l1, l2=l2)

    estimator.fit(half2compr, y_val)

    acc_PCA = estimator.dbg_acc_val_
    print(acc_PCA)
    print('wPCA: %.4f' % acc_PCA[-1])

    outpath = op.join(WRITE_DIR, 'nwPCA-LR_ncomp=%i' % n_comp)
    np.save(outpath + '_acc', acc_PCA)
    joblib.dump(estimator, outpath + '_est', compress=9)

