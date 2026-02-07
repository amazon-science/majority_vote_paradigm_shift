from typing import Optional, Union, List
import cvxpy as cp
import numpy as np
import scipy.stats
import os
import warnings

class InterAnnotatorAgreementAPI:
    def __init__(self, annotations: np.array, optim_norm_p: int = 2, label_distribution: Optional[List[float]] = None,
                 tolerance: float = -1e-8):
        self._annotations = annotations
        self._optim_norm_p = optim_norm_p
        self._tolerance = tolerance
        self._compute_statistics()
        self._wrapper_compute_label_distribution(label_distribution)
        self.is_t_matrix_initialized = False  

    def _wrapper_compute_label_distribution(self, label_distribution: Optional[List[float]]) -> None:
        """
        :param label_distribution: it is a optional float list that sum up to 1, where the i-th element represents a probability for class i.
        :return: None
        """
        if label_distribution is None:
            label_distribution = self._compute_label_distribution(self._annotations)
        self._label_distribution = label_distribution
        assert (
                round(sum(self._label_distribution), 4) == 1
        ), "There is something wrong with your label distribution. It does not sum up to 1."

    def _to_one_hot(self, targets: np.array):
        """
        :param targets: it transforms a list of class
        :return: a numpy array with one-hot encoding
        """
        # target of the shame (dataset_size, num_annotators) to (dataset_size, num_annotators, num_classes)
        res = np.eye(self._num_classes)[targets.reshape(-1)]
        return res.reshape(list(targets.shape) + [self._num_classes])

    def _build_t_matrix(self, check_status: bool = True) -> None:
        assert not self.is_t_matrix_initialized, "T matrix is already optimized. You should construct a new object."
        self._d_matrix = self._D_estimator()
        self._m_matrix = self._M_estimator()

        self._optim_second_term = self._compute_optim_second_term(self._d_matrix, self._m_matrix)
        self._t_hat = self._optimize_T(self._optim_second_term, self._optim_norm_p, check_status)
        #assert (self._t_hat >= self._tolerance).all(), self._t_hat[self._t_hat < self._tolerance]
        assert np.equal(self._t_hat, self._t_hat.T).all(), "Error: T is not symmetric"
        self._fix_negative_values_t_matrix()
        self.is_t_matrix_initialized = True

    def _fix_negative_values_t_matrix(self):
        if not (self._t_hat >= 0).all():
            print("WARNING: WE MANUALLY REMOVED NEGATIVE NUMBERS FROM THE T MATRIX")
            self._t_hat[self._t_hat < 0] = 0

    def __repr__(self) -> str:
        return f"Num annotators: {self._num_annotators}, Dataset Size: {self._num_samples}, Num Classes: {self._num_classes}, Classes: {self._classes}, Class distribution: {self._label_distribution}"

    def _compute_statistics(self) -> None:
        self._num_samples= len(self._annotations)
        self._num_annotators = self._annotations.shape
        self._classes = sorted(list(set(np.hstack(self._annotations))))  # set of all the possible classes
        self._num_classes = len(self._classes)
        assert max(self._classes) == self._num_classes - 1, "Your dataset does not contains all the classes"

    def _compute_label_distribution(self, noisy_y: np.array) -> List[float]:
        results = []
        for item in range(len(noisy_y)):
            results.append([np.mean(noisy_y[item] == c) for c in self._classes])
        return np.mean(np.array(results), axis=0)

    def _D_estimator(self) -> np.array:
        """
        :return: a matrix of shape (num_classes, num_classes) with the label distribution in the diagonal.
        """
        return np.diag(self._label_distribution)

    def _M_estimator(self) -> np.array:
        hat_m = np.zeros((self.num_classes, self.num_classes))
        for i in range(self.num_samples):
            for ann_a in range(len(self._annotations[i])):
                for ann_b in range(ann_a + 1, len(self._annotations[i])):
                    class_a = self.annotations[i][ann_a]
                    class_b = self.annotations[i][ann_b]
                    hat_m[class_a, class_b] += 1
                    hat_m[class_b, class_a] += 1
        hat_m /= (len(self._annotations[i]) * (len(self._annotations[i]) - 1) * self.num_samples) + 1e-6
        return hat_m


    def _compute_optim_second_term(self, D: np.array, M_hat: np.array) -> np.array:
        assert D.shape == (self._num_classes, self._num_classes) and M_hat.shape == (
            self._num_classes,
            self._num_classes,
        )
        app = np.linalg.inv(D ** 0.5)
        app = np.dot(app, np.dot(M_hat, app))
        app, U = np.linalg.eig(app)
        Λ = np.diag(app).astype(complex)  #complex needed to avoid NAN
        inv_U = U.T
        optim_second_term = np.dot(U, np.dot(Λ ** 0.5, inv_U))
        return optim_second_term

    def _optimize_T(self, optim_second_term: np.array, norm_p: int, check_status: bool = True):
        """
        This function find the optimal T in the space of the symmetric,
        stochastic matrices with elements on the diagonal > 0.5.
        We don't need T, I put it just as a control
        """
        # useful functions https://www.cvxpy.org/tutorial/functions/index.html
        t_hat = cp.Variable((self._num_classes, self._num_classes), symmetric=True)
        # Create two constraints.
        constraints = [
            t_hat[self._classes, self._classes] >= 0.5,
            cp.sum(t_hat, axis=1) == 1,
            t_hat >= 0.00001,  # 1e-5 approximation tollerance is 1e-8
        ]
        # I have to put 0.5 + eps because strict inequalities are not allowed
        # Form objective.
        objective = cp.Minimize(cp.norm(t_hat - optim_second_term, norm_p))
        # Form and solve problem.
        problem = cp.Problem(objective, constraints)
        problem.solve()  # Returns the optimal value.
        if problem.status != 'optimal':
            print(problem.status)    
        if check_status:
            if problem.status != 'optimal':
                warnings.warn("Warning: Your T can't be optimized!")
        #assert problem.status == "optimal", "Error: Your T can't be optimized!"
        return t_hat.value
        
    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def classes(self) -> List[int]:
        return self._classes

    @property
    def annotations(self) -> np.array:
        return self._annotations

    @property
    def tolerance(self) -> int:
        return self.tolerance

    @property
    def num_annotators(self) -> int:
        return self._num_annotators

    @property
    def num_samples(self) -> int:
        return self._num_samples

    @property
    def t_hat(self):
        if not self.is_t_matrix_initialized:
            return None
        return self._t_hat

    @property
    def label_distribution(self) -> List[float]:
        return self._label_distribution