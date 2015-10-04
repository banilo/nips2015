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

RES_NAME = 'nips3mm_vanilla'
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

# ARCHI task
X_task, labels = joblib.load('preload_HT_3mm')

labels = np.int32(labels)

# contrasts are IN ORDER -> shuffle!
new_inds = np.arange(0, X_task.shape[0])
np.random.shuffle(new_inds)
X_task = X_task[new_inds]
labels = labels[new_inds]
# subs = subs[new_inds]

X_task = StandardScaler().fit_transform(X_task)

# ARCHI task
AT_niis, AT_labels, AT_subs = joblib.load('preload_AT_3mm')
AT_X = nifti_masker.transform(AT_niis)
AT_X = StandardScaler().fit_transform(AT_X)
print('done :)')

##############################################################################
# define computation graph
##############################################################################

class SSEncoder(BaseEstimator):
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
# plot figures
##############################################################################

def dump_comps(masker, compressor, components, threshold=2, fwhm=None,
               perc=None):
    from scipy.stats import zscore
    from nilearn.plotting import plot_stat_map
    from nilearn.image import smooth_img
    from scipy.stats import scoreatpercentile

    if isinstance(compressor, basestring):
        comp_name = compressor
    else:
        comp_name = compressor.__str__().split('(')[0]

    for i_c, comp in enumerate(components):
        path_mask = op.join(WRITE_DIR, '%s_%i-%i' % (comp_name,
                                                     n_comp, i_c + 1))
        nii_raw = masker.inverse_transform(comp)
        nii_raw.to_filename(path_mask + '.nii.gz')
        
        comp_z = zscore(comp)
        
        if perc is not None:
            cur_thresh = scoreatpercentile(np.abs(comp_z), per=perc)
            path_mask += '_perc%i' % perc
            print('Applying percentile %.2f (threshold: %.2f)' % (perc, cur_thresh))
        else:
            cur_thresh = threshold
            path_mask += '_thr%.2f' % cur_thresh
            print('Applying threshold: %.2f' % cur_thresh)

        nii_z = masker.inverse_transform(comp_z)
        gz_path = path_mask + '_zmap.nii.gz'
        nii_z.to_filename(gz_path)
        plot_stat_map(gz_path, bg_img='colin.nii', threshold=cur_thresh,
                      cut_coords=(0, -2, 0), draw_cross=False,
                      output_file=path_mask + 'zmap.png')
                      
        # optional: do smoothing
        if fwhm is not None:
            nii_z_fwhm = smooth_img(nii_z, fwhm=fwhm)
            plot_stat_map(nii_z_fwhm, bg_img='colin.nii', threshold=cur_thresh,
                          cut_coords=(0, -2, 0), draw_cross=False,
                          output_file=path_mask +
                          ('zmap_%imm.png' % fwhm))
l1 = 0.1
l2 = 0.1
my_title = r'LR: L1=%.1f L2=%.1f res=3mm' % (
    l1, l2
)
print(my_title)
estimator = SSEncoder(
    gain1=0.004,  # empirically determined by CV
    learning_rate = np.float32(0.00001),  # empirically determined by CV,
    max_epochs=500, l1=l1, l2=l2)

estimator.fit(X_task, labels)

# my_title = r'Low-rank LR: n_comp=%i L1=%.1f L2=%.1f res=10mm pca20RS' % (
#     n_comp, l1, l2
# )
# FONTS = 12
# plt.figure(facecolor='white', figsize=(8, 6))
# plt.plot(np.log(estimator.dbg_ae_cost_), label='cost autoencoder')
# plt.plot(estimator.dbg_lr_cost_, label='cost logistic')
# plt.plot(estimator.dbg_acc_train_, label='score training set')
# plt.plot(estimator.dbg_acc_val_, label='score validation set')
# plt.plot(estimator.dbg_acc_other_ds_, label='other-datset acc')
# plt.legend(loc='best', fontsize=12)
# plt.xlabel('epoch', fontsize=FONTS)
# plt.ylabel('misc', fontsize=FONTS)
# plt.yticks(np.arange(12), np.arange(12))
# plt.grid(True)
# plt.title(my_title)
# plt.show()

fname = my_title.replace(' ', '_').replace('+', '').replace(':', '').replace('__', '_').replace('%', '')
cur_path = op.join(WRITE_DIR, fname)
joblib.dump(estimator, cur_path)
# plt.savefig(cur_path + '_SUMMARY.png', dpi=200)

V0_mat = estimator.V0s.get_value().T
np.save(cur_path + 'V0comps', V0_mat)

# dump data also as numpy array
np.save(cur_path + 'dbg_epochs_', np.array(estimator.dbg_epochs_))
np.save(cur_path + 'dbg_acc_train_', np.array(estimator.dbg_acc_train_))
np.save(cur_path + 'dbg_acc_val_', np.array(estimator.dbg_acc_val_))
np.save(cur_path + 'dbg_lr_cost_', np.array(estimator.dbg_lr_cost_))
np.save(cur_path + 'dbg_prfs_', np.array(estimator.dbg_prfs_))


dump_comps(nifti_masker, fname, comps, threshold=0.25)

import re
pkgs = glob.glob(RES_NAME + '/*dbg_epochs*.npy')
dbg_epochs_ = np.load(pkgs[0])
d = {
    'training accuracy': '/*dbg_acc_train*.npy',
    'accuracy val': '/*dbg_acc_val_*.npy',
    'loss lr': '/*dbg_lr_cost_*.npy'
}
for k, v in d.iteritems():
    pkgs = glob.glob(RES_NAME + v)
    p = pkgs[0]
    cur_values = np.load(p)

    cur_label = ''
    if not '_AE' in p:
        cur_label += 'LR only!'
    elif 'subRS' in p:
        cur_label += 'RSnormal'
    elif 'spca20RS' in p:
        cur_label += 'RSspca20'
    elif 'pca20RS' in p:
        cur_label += 'RSpca20'
    cur_label += '/'
    cur_label += 'separate decomp.' if 'decomp_separate' in p else 'joint decomp.'
    cur_label += '' if '_AE' in p else '/LR only!'
    plt.figure()
    plt.plot(
        dbg_epochs_,
        cur_values,
        label=cur_label)
    plt.title('LR L1=0.1 L2=0.1 res=3mm')
    plt.yticks(np.linspace(0., 1., 11))
    plt.ylabel(k)
    plt.xlabel('epochs')
    plt.ylim(0., 1.05)
    plt.grid(True)
    plt.show()
    plt.savefig(op.join(WRITE_DIR, k.replace(' ', '_') + '.png'))


# print class weights
pkgs = ['nips3mm_vanilla/V0comps.npy']
comps = np.load(pkgs[0])
new_fname = 'comps_perc75'
dump_comps(nifti_masker, new_fname, comps, perc=75, fwhm=None)