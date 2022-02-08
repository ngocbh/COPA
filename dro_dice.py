from dice_ml.explainer_interfaces.explainer_base import ExplainerBase
from dice_ml.explainer_interfaces.dice_pytorch import DicePyTorch
from dice_ml import Dice
from dice_ml import diverse_counterfactuals as exp
from validity import lower_validity_bound, upper_validity_bound
from correction import mahalanobis_correction
from utils.utils import compute_dpp, lp_dist, check_symmetry, sqrtm_psd
from autograd import value_and_grad
from epts.dice_wrapper import DicePyTorchWrapper

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import autograd.numpy as jnp
import numpy as np
import scipy


class DroDice(ExplainerBase):

    def __init__(self, data_interface, model_interface, method,
                 mean_weights, cov_weights, rho, epsilon, max_k=None, **kwargs):
        self.mean_weights = mean_weights
        self.cov_weights = cov_weights
        self.rho = rho
        self.epsilon = epsilon
        self.max_k = max_k

        self.history = []

        super(DroDice, self).__init__(data_interface, model_interface)

        self.dice = Dice(data_interface, model_interface, method, **kwargs)

    def _generate_counterfactuals(self, query_instance, total_CFs,  **kwargs):
        if isinstance(self.dice, DicePyTorch):
            exp = self.dice.generate_counterfactuals(query_instance,
                                                     total_CFs,
                                                     **kwargs)
        else:
            exp = self.dice._generate_counterfactuals(query_instance,
                                                      total_CFs,
                                                      **kwargs)

        # print("="*10, "Origin\n", exp.final_cfs_df)
        cfs = exp.final_cfs_df.drop(self.data_interface.outcome_name, axis=1)
        test_ins = exp.test_instance_df.drop(
            self.data_interface.outcome_name, axis=1)

        cfs = self.model.model.data_transformer.transform(cfs, tonumpy=True,
                                                          intercept_feature=True)
        test_ins = self.model.model.data_transformer.transform(test_ins, tonumpy=True,
                                                               intercept_feature=True)

        logs = {}

        # logs['on_correction_begin'] = self.on_correction_begin(
        # cfs, exp.final_cfs_df)

        cfs = self._correct(test_ins, cfs)

        cfs_df = self.model.model.data_transformer.inverse_transform(cfs,
                                                                     intercept_feature=True)

        cfs_pred = self.model.model.predict(cfs_df)
        cfs_df[self.data_interface.outcome_name] = cfs_pred

        # logs['on_correction_end'] = self.on_correction_end(cfs, cfs_df)

        self.history.append(logs)

        exp.final_cfs_df = cfs_df

        exp.final_cfs_df_sparse = cfs
        return exp

    def on_correction_begin(self, cfs, cfs_df, logs=None):
        logs = logs or {}
        logs['dpp'] = compute_dpp(cfs, method='inverse_dist', dist=lp_dist)
        lb, _ = lower_validity_bound(cfs, self.mean_weights,
                                     self.cov_weights, self.rho)

        logs['lower_validity_bound'] = lb
        return logs

    def on_correction_end(self, cfs, cfs_df, logs=None):
        logs = logs or {}
        logs['dpp'] = compute_dpp(cfs, method='inverse_dist', dist=lp_dist)
        lb, _ = lower_validity_bound(cfs, self.mean_weights,
                                     self.cov_weights, self.rho)

        logs['lower_validity_bound'] = lb
        return logs

    def _project(self, cfs):
        w = self.mean_weights
        for i in range(len(cfs)):
            cfs[i][0] = 1
            cfs[i][1:] = cfs[i][1:] - min(0, np.dot(w, cfs[i]) - self.epsilon) \
                * w[1:] / np.linalg.norm(w[1:]) ** 2
        return cfs

    def _correct(self, test_ins, cfs):
        # cfs = np.pad(cfs, ((0, 0), (1, 0)), constant_values=1)
        cfs = self._project(cfs)
        cfs = mahalanobis_correction(cfs, self.mean_weights, self.cov_weights,
                                     self.rho, self.epsilon, self.max_k)
        # cfs = cfs[:, 1:]
        return cfs


class DroDicePGDT(ExplainerBase):

    def __init__(self, data_interface, model_interface,
                 mean_weights, cov_weights, robust_weight=10.0,
                 diversity_weight=5.0, lambd=0.7, zeta=1, max_iter=500,
                 epsilon=0.1, learning_rate=0.005,
                 features_to_vary='all', verbose=False, **kwargs):
        self.max_iter = max_iter
        self.diversity_weight = diversity_weight
        self.robust_weight = robust_weight
        self.num_stable_iter = 0
        self.learning_rate = learning_rate
        self.max_stable_iter = 3
        self.loss_diff_threshold = 1e-7
        self.transformer = model_interface.model.data_transformer
        self.sigma_perturb_init = 1
        self.compute_value_and_grad = value_and_grad(self._compute_loss)
        self.mean_weights = mean_weights
        self.cov_weights = cov_weights
        self.epsilon = epsilon
        self.lambd = lambd
        self.zeta = zeta
        self.verbose = verbose

        super(DroDicePGDT, self).__init__(
            data_interface, model_interface, **kwargs)

        self.clf = model_interface.model
        self.features_to_vary = features_to_vary
        self.feat_to_vary_idxs = self.data_interface.get_indexes_of_features_to_vary(
            features_to_vary=features_to_vary)
        self.sqrtm_cov_weights = torch.tensor(sqrtm_psd(cov_weights)).double()
        self.mean_weights_tensor = torch.tensor(mean_weights).double()
        self.cov_weights_tensor = torch.tensor(cov_weights).double()
        self.minx, self.maxx, self.encoded_categorical_feature_indexes, self.encoded_continuous_feature_indexes, \
            self.cont_minx, self.cont_maxx, self.cont_precisions = self.data_interface.get_data_params_for_gradient_dice()

    def _cost_func(self, x, y):
        """_cost_func.
            cost function is L1 distance
        """
        return torch.norm(x-y, 2)

    def _compute_proximity_loss(self, test_ins, cfs):
        proximity = 0
        for i in range(len(cfs)):
            proximity += self._cost_func(test_ins, cfs[i])

        return proximity / len(cfs)

    def _compute_robust_loss(self, cfs):
        rs = []
        for i in range(len(cfs)):
            r = torch.dot(cfs[i], self.mean_weights_tensor) / \
                torch.norm(self.sqrtm_cov_weights @ cfs[i])
            rs.append(r)

        return -min(rs)

    def _compute_diversity_loss(self, cfs):
        diversity = 0
        num_cfs = len(cfs)
        for i in range(num_cfs):
            for j in range(i+1, num_cfs):
                diversity += cfs[i].T @ self.cov_weights_tensor @ cfs[j] / \
                    (torch.norm(self.sqrtm_cov_weights @ cfs[i]) *
                     torch.norm(self.sqrtm_cov_weights @ cfs[j]))
        return diversity / (num_cfs * (num_cfs - 1) / 2)

    def _compute_dpp_loss(self, cfs):
        num_cfs = len(cfs)
        det_entries = torch.ones((num_cfs, num_cfs))
        for i in range(num_cfs):
            for j in range(num_cfs):
                det_entries[(i, j)] = 1.0 / \
                    (1.0 + self._cost_func(cfs[i], cfs[j]))
                if i == j:
                    det_entries[(i, j)] += 0.0001

        diversity_loss = torch.det(det_entries)
        return -diversity_loss

    def _compute_loss(self, cfs):
        self.robust_loss = self._compute_robust_loss(cfs)
        self.proximity_loss = self._compute_proximity_loss(self.test_ins, cfs)
        # self.diversity_loss = self._compute_diversity_loss(cfs)
        self.diversity_loss = self._compute_dpp_loss(cfs)
        loss = self.proximity_loss + self.robust_weight * self.robust_loss + \
            self.diversity_weight * self.diversity_loss
        return loss

    def _get_output(self, cfs, w):
        with torch.no_grad():
            return torch.tensor([torch.dot(cf, w) for cf in cfs]).numpy()

    def print_cfs(self, cfs, grad=False):
        if grad:
            print(torch.stack([cf.grad.data for cf in cfs],
                  axis=0).detach().numpy())
        else:
            with torch.no_grad():
                print(torch.stack(cfs, axis=0).numpy())

    def _init_cfs(self, x_0, total_CFs):
        cfs = np.tile(x_0, (total_CFs, 1))
        cfs = cfs + np.random.randn(*cfs.shape) * self.sigma_perturb_init
        cfs[:, 0] = 1.0
        return cfs

    def _project(self, cfs):
        with torch.no_grad():
            w = self.mean_weights_tensor
            for i in range(len(cfs)):
                # cfs[i][0] = 1
                cfs[i][1:] = cfs[i][1:] - min(0, torch.dot(w, cfs[i]) - self.epsilon) \
                    * w[1:] / torch.norm(w[1:]) ** 2
        return cfs

    def _check_termination(self, loss_diff):
        # print(loss_diff, self.loss_diff_threshold, self.num_stable_iter)
        if loss_diff <= self.loss_diff_threshold:
            self.num_stable_iter += 1
            return (self.num_stable_iter >= self.max_stable_iter)
        else:
            self.num_stable_iter = 0
            return False

    def _generate_counterfactuals(self, query_instance, total_CFs,  **kwargs):
        test_ins = self.transformer.transform(
            query_instance, tonumpy=True, intercept_feature=True).squeeze()
        self.test_ins = torch.tensor(test_ins).double()
        initial_cfs = self._init_cfs(test_ins, total_CFs)
        num_cfs, d = initial_cfs.shape

        cfs = [torch.tensor(cf, requires_grad=True) for cf in initial_cfs]
        cfs = self._project(cfs)
        optim = torch.optim.Adam(cfs, 0.01)
        # for cf in cfs:
        # print(cf)

        loss_diff = 1.0
        prev_loss = 0
        self.num_stable_iter = 0
        # print(self.mean_weights)

        for num_iter in range(self.max_iter):
            i = 0
            # loss, l_grad = self.compute_value_and_grad(cfs)
            for cf in cfs:
                cf.grad = None

            loss = self._compute_loss(cfs)
            loss.backward()

            # freeze features other than feat_to_vary_idxs
            for ix in range(num_cfs):
                for jx in range(d):
                    if jx-1 not in self.feat_to_vary_idxs:
                        cfs[ix].grad[jx] = 0.0

            # for cf in cfs:
                # print(cf.grad.data)

            while True:
                with torch.no_grad():
                    cfs_prime = []
                    for cf in cfs:
                        cfs_prime.append(cf - self.lambd **
                                         i * self.zeta * cf.grad.data)

                    proj_cfs_prime = self._project(cfs_prime)

                    lhs = self._compute_loss(proj_cfs_prime)
                    # print(i, 1/(2 * self.lambd ** i * self.zeta))
                    rhs = loss - 1/(2 * self.lambd ** i * self.zeta) * \
                        torch.norm(torch.stack(
                            [cfs[i] - proj_cfs_prime[i] for i in range(num_cfs)])) ** 2

                    if lhs <= rhs:
                        cfs = proj_cfs_prime
                        for cf in cfs:
                            cf.requires_grad = True
                        loss_diff = loss - lhs
                        if self.verbose:
                            print("Iter %d: i = %d; old_loss: %f, new_loss: %f; new_loss_bound: %f" % (
                                num_iter, i, loss, lhs, rhs))
                            print("---- Robust loss: %f * %f; Proximity loss: %f * %f; Diversity loss: %f * %f" %
                                  (self.robust_weight, self.robust_loss,
                                   1, self.proximity_loss,
                                   self.diversity_weight, self.diversity_loss))
                        break
                    i += 1
                    if i >= 1000:
                        break

            # loss_diff = prev_loss - loss
            if self._check_termination(loss_diff):
                break
            prev_loss = loss

        cfs = np.array([cf.detach().numpy() for cf in cfs])

        cfs_df = self.transformer.inverse_transform(cfs,
                                                    intercept_feature=True)
        cfs_pred = self.clf.predict(cfs_df)
        cfs_df[self.data_interface.outcome_name] = cfs_pred

        test_ins_df = self.transformer.inverse_transform(np.expand_dims(test_ins, axis=0),
                                                         intercept_feature=True)
        test_pred = self.clf.predict(test_ins_df)
        test_ins_df[self.data_interface.outcome_name] = test_pred

        # logs['on_correction_end'] = self.on_correction_end(cfs, cfs_df)
        counterfactual_explanations = exp.CounterfactualExamples(
            data_interface=self.data_interface,
            final_cfs_df=cfs_df,
            test_instance_df=test_ins_df,
            final_cfs_df_sparse=cfs_df,
            posthoc_sparsity_param=None,
            desired_class="opposite")

        return counterfactual_explanations


class DroDicePGDAD(DroDicePGDT):
    def _generate_counterfactuals(self, query_instance, total_CFs, **kwargs):
        test_ins = self.transformer.transform(
            query_instance, tonumpy=True, intercept_feature=True).squeeze()
        # print(test_ins)
        # print(self.mean_weights)
        self.test_ins = torch.tensor(test_ins).double()
        initial_cfs = self._init_cfs(test_ins, total_CFs)
        num_cfs, d = initial_cfs.shape

        cfs = [torch.tensor(cf, requires_grad=True) for cf in initial_cfs]
        cfs = self._project(cfs)
        # self.print_cfs(cfs)
        # print(self._get_output(cfs, self.mean_weights_tensor))
        optim = torch.optim.Adam(cfs, self.learning_rate)

        loss_diff = 1.0
        prev_loss = 0.0
        self.num_stable_iter = 0

        for num_iter in range(self.max_iter):
            optim.zero_grad()

            loss_value = self._compute_loss(cfs)
            loss_value.backward()

            # freeze features other than feat_to_vary_idxs
            for ix in range(num_cfs):
                for jx in range(d):
                    if jx-1 not in self.feat_to_vary_idxs:
                        cfs[ix].grad[jx] = 0.0

            # print("before step")
            # self.print_cfs(cfs, grad=False)
            optim.step()

            # print("after step")
            cfs = self._project(cfs)
            # self.print_cfs(cfs, grad=False)
            # print("grad")
            # self.print_cfs(cfs, grad=True)

            if self.verbose:
                print("Iter %d: loss: %f" % (num_iter, loss_value.data.item()))

                print("---- Robust loss: %f * %f; Proximity loss: %f * %f; Diversity loss: %f * %f" %
                      (self.robust_weight, self.robust_loss,
                       1, self.proximity_loss,
                       self.diversity_weight, self.diversity_loss))

            # projection step
            for ix in range(num_cfs):
                for jx in range(1, d):
                    cfs[ix].data[jx] = torch.clamp(cfs[ix][jx],
                                                   min=self.minx[0][jx-1],
                                                   max=self.maxx[0][jx-1])

            # print(self._get_output(cfs, self.mean_weights_tensor))

            loss_diff = prev_loss - loss_value.data.item()
            if self._check_termination(loss_diff):
                break

            prev_loss = loss_value.data.item()

        # print(self._get_output(cfs, self.mean_weights_tensor))

        cfs = np.array([cf.detach().numpy() for cf in cfs])

        cfs_df = self.transformer.inverse_transform(cfs,
                                                    intercept_feature=True)
        cfs_pred = self.clf.predict(cfs_df)
        cfs_df[self.data_interface.outcome_name] = cfs_pred

        test_ins_df = self.transformer.inverse_transform(np.expand_dims(test_ins, axis=0),
                                                         intercept_feature=True)
        test_pred = self.clf.predict(test_ins_df)
        test_ins_df[self.data_interface.outcome_name] = test_pred

        # logs['on_correction_end'] = self.on_correction_end(cfs, cfs_df)
        counterfactual_explanations = exp.CounterfactualExamples(
            data_interface=self.data_interface,
            final_cfs_df=cfs_df,
            test_instance_df=test_ins_df,
            final_cfs_df_sparse=cfs,
            posthoc_sparsity_param=None,
            desired_class="opposite")

        return counterfactual_explanations
