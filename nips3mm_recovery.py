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
from scipy.stats import zscore

LR_AE_DIR = 'nips3mm'
LR_DIR = 'nips3mm_vanilla'

##############################################################################
# abc
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
# abc
##############################################################################



means_path = '/git/cohort/archi/compr2task_means'

mask_img = 'grey10_icbm_3mm_bin.nii.gz'
nifti_masker = NiftiMasker(mask_img=mask_img, smoothing_fwhm=False,
                           standardize=False)
nifti_masker.fit()
mask_nvox = nifti_masker.mask_img_.get_data().sum()

# get HCP data
HCP_contrasts = [
    'REWARD-PUNISH', 'PUNISH-REWARD', 'SHAPES-FACES', 'FACES-SHAPES',
    'RANDOM-TOM', 'TOM-RANDOM',

    'MATH-STORY', 'STORY-MATH',
    'T-AVG', 'F-H', 'H-F',
    'MATCH-REL', 'REL-MATCH',

    'BODY-AVG', 'FACE-AVG', 'PLACE-AVG', 'TOOL-AVG',
    '2BK-0BK'
]
mean_supp = np.zeros((18, mask_nvox))
from nilearn.image import resample_img
for itask, task in enumerate(HCP_contrasts):
    cur_nii = op.join(means_path, 'mean_%s.nii.gz' % (task))
    print(cur_nii)
    res_nii = resample_img(cur_nii,
        target_affine=nifti_masker.mask_img_.get_affine(),
        target_shape=nifti_masker.mask_img_.shape)
    task_mean = nifti_masker.transform(res_nii)
    mean_supp[itask, :] = task_mean
mean_supp_z = zscore(mean_supp, axis=1)
    
# get classification weights
lr_supp = np.load(op.join(LR_DIR, 'V0comps.npy'))
lr_supp_z = zscore(lr_supp, axis=1)

# get LR/AE weights
WRITE_DIR = 'nips3mm_recovery'
lambs = [0.25, 0.5, 0.75, 1]
import re
from scipy.stats import pearsonr
for n_comp in [5]:
    corr_means_lr = np.zeros((len(lambs), 18))
    corr_means_lr_ae = np.zeros((len(lambs), 18))
    for ilamb, lamb in enumerate(lambs):
        pkgs = glob.glob(op.join(LR_AE_DIR, '*comp=%i_*V1comps*' % n_comp))
        for p in pkgs:
            lambda_param = np.float(re.search('lambda=(.{4})', p).group(1))
            # n_hidden = int(re.search('comp=(.{1,3})_', p).group(1))
            if lamb != lambda_param:
                continue

            print(p)
            q = p.replace('V1comp', 'W0comp')

            try:
                mat_V1 = np.load(p)
                mat_W0 = np.load(q)
            except:
                print('Did not find %s oder %s!' % (p, q))

            lr_ae_supp = np.dot(mat_V1, mat_W0)
            lr_ae_supp_z = zscore(lr_ae_supp, axis=1)

            for i in np.arange(18):
                r1, p1 = pearsonr(lr_ae_supp_z[i, :], mean_supp_z[i, :])
                print('r/lrae: %.4f' % r1)
                corr_means_lr_ae[ilamb, i] = r1

                r2, p2 = pearsonr(lr_supp_z[i, :], mean_supp_z[i, :])
                print('r/lr: %.4f' % r2)
                corr_means_lr[ilamb, i] = r2

    # boxplot
    plt.figure()
    corrs = np.vstack((corr_means_lr[0, :], corr_means_lr_ae))
    plt.boxplot(corrs.T)
    plt.ylabel('correlation r')
    plt.title('Support Recovery: normal versus low-rank logistic regression\n'
              '%i components' % n_comp)
    tick_strs = [u'normal'] + [u'low-rank lambda=%.2f' % val for val in lambs]
    plt.xticks(np.arange(6) + 1, tick_strs, rotation=320)
    plt.ylim(0, 1.0)
    plt.yticks(np.linspace(0, 1., 11), np.linspace(0, 1., 11))
    plt.tight_layout()
    
    out_path = op.join(WRITE_DIR, 'supp_recov_comp=%i.png' % n_comp)
    plt.savefig(out_path)

    # barplot
    plt.figure()
    ind = np.arange(5)
    width = 1.
    colors = [(242., 62., 22.), #(7., 196., 255.),
        (7., 176., 242.), (7., 136., 217.), (7., 40., 164.), (1., 4., 64.)]
    my_colors = [(x/256, y/256, z/256) for x, y, z in colors]
    plt.bar(ind, np.mean(corrs, axis=1), yerr=np.std(corrs, axis=1),
            width=width, color=my_colors)
    plt.ylabel('correlation r (+/- SD)')
    plt.title('Support Recovery: normal versus low-rank logistic regression\n'
              '%i components' % n_comp)
    plt.xticks(ind + width / 2., tick_strs, rotation=320)
    plt.ylim(0, 1.0)
    plt.yticks(np.linspace(0, 1., 11), np.linspace(0, 1., 11))
    plt.tight_layout()
    out_path2 = out_path.replace('.png', '_bars.png')
    plt.savefig(out_path2)


