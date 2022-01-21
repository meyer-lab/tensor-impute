# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from tensorly import partial_svd
import numpy as np

from .soft_impute import Solver

F32PREC = np.finfo(np.float32).eps

def masked_mae(X_true, X_pred, mask):
    masked_diff = X_true[mask] - X_pred[mask]
    return np.mean(np.abs(masked_diff))


class IterativeSVD(Solver):
    def __init__(
            self,
            rank,
            convergence_threshold=0.00001,
            max_iters=200,
            init_fill_method="zero",
            random_state=None,
            min_value=None,
            max_value=None,
            verbose=False):
        Solver.__init__(
            self,
            fill_method=init_fill_method,
            min_value=min_value,
            max_value=max_value)
        self.rank = rank
        self.max_iters = max_iters
        self.convergence_threshold = convergence_threshold
        self.verbose = verbose
        self.random_state = random_state

    def _converged(self, X_old, X_new, missing_mask):
        # check for convergence
        old_missing_values = X_old[missing_mask]
        new_missing_values = X_new[missing_mask]
        difference = old_missing_values - new_missing_values
        ssd = np.sum(difference ** 2)
        old_norm_squared = (old_missing_values ** 2).sum()
        # edge cases
        if old_norm_squared == 0 or \
                (old_norm_squared < F32PREC and ssd > F32PREC):
            return False
        else:
            return (ssd / old_norm_squared) < self.convergence_threshold

    def solve(self, X, missing_mask):
        # X = check_array(X, force_all_finite=False)

        observed_mask = ~missing_mask
        X_filled = X
        for i in range(self.max_iters):
            curr_rank = self.rank
            U, S, V = partial_svd(X_filled, curr_rank, random_state=self.random_state)
            X_reconstructed = U @ np.diag(S) @ V
            X_reconstructed = self.clip(X_reconstructed)
            mae = masked_mae(
                X_true=X,
                X_pred=X_reconstructed,
                mask=observed_mask)
            if self.verbose:
                print(
                    "[IterativeSVD] Iter %d: observed MAE=%0.6f" % (
                        i + 1, mae))
            converged = self._converged(
                X_old=X_filled,
                X_new=X_reconstructed,
                missing_mask=missing_mask)
            X_filled[missing_mask] = X_reconstructed[missing_mask]
            if converged:
                break
        return X_filled