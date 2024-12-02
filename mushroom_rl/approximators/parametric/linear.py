import numpy as np

from mushroom_rl.core import Serializable


class LinearApproximator(Serializable):
    """
    This class implements a linear approximator.

    """
    def __init__(self, weights=None, input_shape=None, output_shape=(1,), phi=None,
                 **kwargs):
        """
        Constructor.

        Args:
             weights (np.ndarray): array of weights to initialize the weights of the approximator;
             input_shape (np.ndarray, None): the shape of the input of the model;
             output_shape (np.ndarray, (1,)): the shape of the output of the model;
             phi (object, None): features to extract from the state;
             **kwargs: other params of the approximator.

        """
        assert len(input_shape) == 1 and len(output_shape) == 1

        input_dim = input_shape[0]
        output_dim = output_shape[0]

        if weights is not None:
            self._w = weights.reshape((output_dim, -1))
        elif input_dim is not None:
            self._w = np.zeros((output_dim, input_dim))
        else:
            raise ValueError('You should specify the initial parameter vector'
                             ' or the input dimension')

        self._phi = phi
        self._add_save_attr(
            _w='numpy',
            _phi='pickle'
        )

    def fit(self, x, y, **fit_params):
        """
        Fit the model.

        Args:
            x (np.ndarray): input;
            y (np.ndarray): target;
            **fit_params: other parameters used by the fit method of the regressor.

        """
        phi = np.atleast_2d(self.phi(x))
        self._w = np.atleast_2d(np.linalg.pinv(phi).dot(y).T)

    def predict(self, x, **predict_params):
        """
        Predict.

        Args:
            x (np.ndarray): input;
            **predict_params: other parameters used by the predict method the regressor.

        Returns:
            The predictions of the model.

        """
        phi = np.atleast_2d(self.phi(x))

        prediction = np.matmul(phi, self._w.T)

        return prediction

    @property
    def weights_size(self):
        """
        Returns:
            The size of the array of weights.

        """
        return self._w.size

    def get_weights(self):
        """
        Getter.

        Returns:
            The set of weights of the approximator.

        """
        return self._w.flatten()

    def set_weights(self, w):
        """
        Setter.

        Args:
            w (np.ndarray): the set of weights to set.

        """
        self._w = w.reshape(self._w.shape)

    def phi(self, x):
        if self._phi is not None:
            return self._phi(x)
        else:
            return x

    def diff(self, state, action=None):
        """
        Compute the derivative of the output w.r.t. ``state``, and ``action`` if provided.

        Args:
            state (np.ndarray): the state;
            action (np.ndarray, None): the action.

        Returns:
            The derivative of the output w.r.t. ``state``, and ``action``
            if provided.

        """
        if len(self._w.shape) == 1 or self._w.shape[0] == 1:
            return self.phi(state)
        else:
            n_phi = self._w.shape[1]
            n_outs = self._w.shape[0]

            if action is None:
                shape = (n_phi * n_outs, n_outs)
                df = np.zeros(shape)
                start = 0
                for i in range(n_outs):
                    stop = start + n_phi
                    df[start:stop, i] = self.phi(state)
                    start = stop
            else:
                shape = (n_phi * n_outs)
                df = np.zeros(shape)
                start = action[0] * n_phi
                stop = start + n_phi
                df[start:stop] = self.phi(state)

            return df

