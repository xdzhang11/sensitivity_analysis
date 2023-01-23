import numpy as np
import openturns as ot
import pandas as pd

from .utils import q2_cv, DF_NAMES

# TODO: parallelize the model function

class Model(object):
    """A model that evaluate a given function.

    This class aims to gather the informations of a function and eventually 
    parallelize it.

    Parameters
    ----------
    model_func : callable,
        The model function.
    """
    def __init__(self, model_func, name='Custom'):
        self.model_func = model_func
        self.name = name

    def __call__(self, x):
        y = self._model_func(x)
        return y

    @property
    def model_func(self):
        """The model function.
        """
        return self._model_func

    @model_func.setter
    def model_func(self, func):
        if func is not None:
            assert callable(func), "The function should be callable"
        self._model_func = func


class ProbabilisticModel(Model):
    """A probabilistic model that evaluate a given function.

    This class aims to gather the informations of a function and its
    probabilistic input distribution.

    Parameters
    ----------
    model_func : callable
        The model function.

    input_distribution : ot.DistributionImplementation
        The probabilistic input distribution.

    first_sobol_indices : list, array or None, optional (default=None)
        The true first-order sobol indices.

    total_sobol_indices : list, array or None, optional (default=None)
        The true total sobol indices.

    shapley_indices : list, array or None, optional (default=None)
        The true shapley indices.
    """
    def __init__(self, 
                 model_func, 
                 input_distribution,
                 name='Custom',
                 first_sobol_indices=None,
                 total_sobol_indices=None,
                 shapley_indices=None):
        #super(ProbabilisticModel, self).__init__(
        #    model_func=model_func)
        Model.__init__(self, model_func=model_func, name=name)

        self.input_distribution = input_distribution
        self.first_sobol_indices = first_sobol_indices
        self.total_sobol_indices = total_sobol_indices
        self.shapley_indices = shapley_indices

    @property
    def input_distribution(self):
        """The OpenTURNS input distribution.
        """
        return self._input_distribution

    @input_distribution.setter
    def input_distribution(self, dist):
        if dist is not None:
            assert isinstance(dist, ot.DistributionImplementation), \
                "The distribution should be an OpenTURNS implementation."
            self._dim = dist.getDimension()
            self._margins = [dist.getMarginal(i) for i in range(self._dim)]
            self._copula = dist.getCopula()

        self._input_distribution = dist

    # TODO: make the computation of the property possible for functions of true
    # indices.
    @property
    def first_sobol_indices(self):
        """The true first-order sobol indices.
        """        
        return self._first_sobol_indices

    @first_sobol_indices.setter
    def first_sobol_indices(self, indices):
        if indices is not None:
            assert len(indices) == self._dim, \
                "Incorrect number of indices: %d!=%d" % (len(indices), self._dim)
        self._first_sobol_indices = indices

    @property
    def total_sobol_indices(self):
        """The true total sobol indices.
        """
        return self._total_sobol_indices

    @total_sobol_indices.setter
    def total_sobol_indices(self, indices):
        if indices is not None:
            assert len(indices) == self._dim, \
                "Incorrect number of indices: %d!=%d" % (len(indices), self._dim)
        self._total_sobol_indices = indices

    @property
    def shapley_indices(self):
        """The true shapley indices.
        """
        return self._shapley_indices

    @shapley_indices.setter
    def shapley_indices(self, indices):
        if indices is not None:
            assert len(indices) == self._dim, \
                "Incorrect number of indices: %d!=%d" % (len(indices), self._dim)
        self._shapley_indices = indices

    @property
    def margins(self):
        """The problem margins.
        """
        return self._margins

    @margins.setter
    def margins(self, margins):
        for marginal in margins:
            assert isinstance(marginal, ot.DistributionImplementation), \
                "The marginal should be an OpenTURNS implementation."
        assert len(margins) == self._dim, \
            "Incorrect dimension: %d!=%d" % (len(margins), self._dim)
        self._input_distribution = ot.ComposedDistribution(margins, self._copula)
        self._margins = margins

    @property
    def copula(self):
        """The input distribution copula.
        """
        return self._copula
    
    @copula.setter
    def copula(self, copula):
        assert isinstance(copula, (ot.CopulaImplementation, ot.DistributionImplementationPointer)), \
            "The copula should be an OpenTURNS implementation: {0}".format(type(copula))
        assert copula.getDimension() == self._dim, \
            "Incorrect dimension: %d!=%d" % (copula.getDimension(), self._dim)
        self._input_distribution = ot.ComposedDistribution(self._margins, copula)
        self._copula = copula

    @property
    def copula_parameters(self):
        """The copula parameters.
        """
        return self._copula_parameters

    @copula_parameters.setter
    def copula_parameters(self, params):
        copula = self._copula
        copula.setParameter(params)
        self.copula = copula

    @property
    def dim(self):
        """The problem dimension.
        """
        return self._dim

    def get_input_sample(self, n_sample, sampling='lhs'):
        """Generate a sample of the input distribution.

        Parameters
        ----------
        n_sample : int
            The number of observations.

        sampling : str, optional (default='lhs')
            The sampling type.

        Returns
        -------
        input_sample : array,
            A sample of the input distribution.
        """
        assert isinstance(n_sample, int), "Sampling number must be an integer"
        assert isinstance(sampling, str), "Sampling must be a string"
        
        dist = self._input_distribution
        input_sample = sample_dist(dist, n_sample, sampling)

        return input_sample

    @property
    def df_indices(self):
        """A dataframe of the true indices.
        
        Returns
        -------
        indices : dataframe
            The dataframe of the registered sensitivity indices.
        """
        d_indices = {}
        if self._first_sobol_indices is not None:
            d_indices['True first'] = self._first_sobol_indices
        if self._total_sobol_indices is not None:
            d_indices['True total'] = self._total_sobol_indices
        if self._shapley_indices is not None:
            d_indices['True shapley'] = self._shapley_indices

        if d_indices:
            d_indices[DF_NAMES['var']] = ['$X_{%d}$' % (i+1) for i in range(self._dim)]
            df = pd.DataFrame(d_indices)
            indices = pd.melt(df, id_vars=[DF_NAMES['var']], var_name=DF_NAMES['ind'], value_name=DF_NAMES['val'])
        else:
            indices = None
            print('There is no true indices.')
        return indices


class MetaModel(ProbabilisticModel):
    """Meta model class.

    The base class for meta models.

    Parameters
    ----------
    model : callable or None, optional (default=None)
        The true model function.

    input_distribution : ot.DistributionImplementation or None, optional (default=None)
        The probabilistic input distribution.
    """
    def __init__(self, model=None, input_distribution=None, name='Custom'):
        if isinstance(model, ProbabilisticModel):
            super(MetaModel, self).__init__(
                model_func=None,
                input_distribution=input_distribution,
                name=name,
                first_sobol_indices=model.first_sobol_indices,
                total_sobol_indices=model.total_sobol_indices,
                shapley_indices=model.shapley_indices
                )
        else:
            super(MetaModel, self).__init__(
                model_func=None,
                input_distribution=input_distribution,
                name=name)
        self.true_model = model

    def generate_sample(self, n_sample=50, sampling='lhs', sampling_type='uniform', alpha=0.999):
        """Generate the sample to build the model.

        Parameters
        ----------
        n_sample : int,
            The sampling size.
        sampling : str,
            The sampling method to use.
        """
        dist = change_distribution(self._input_distribution, sampling_type, alpha)
        input_sample = sample_dist(dist, n_sample, sampling)

        self.input_sample = np.asarray(input_sample)
        self.output_sample = self.true_model(input_sample)

    @property
    def input_sample(self):
        """The input sample to build the model.
        """
        return self._input_sample
    
    @input_sample.setter
    def input_sample(self, sample):
        n_sample, dim = sample.shape
        assert dim == self._dim, "Dimension should be the same as the input_distribution: %d != %d" % (dim, self._dim)
        self._n_sample = n_sample
        self._input_sample = sample

    @property
    def output_sample(self):
        """The output sample to build the model.
        """
        return self._output_sample
    
    @output_sample.setter
    def output_sample(self, sample):
        n_sample = sample.shape[0]
        assert n_sample == self._n_sample, "Samples should be the same sizes: %d != %d" % (n_sample, self._n_samples)
        self._output_sample = sample
        
    def compute_score_q2_cv(self, n_sample=100, sampling='lhs', sampling_type='classic', alpha=0.99):
        """Cross Validation estimation of Q2.
        """        
        dist = change_distribution(self._input_distribution, sampling_type, alpha)
        x = sample_dist(dist, n_sample, sampling)
        
        ytrue = self.true_model(x)
        ypred = self.predict(x)
        q2 = q2_cv(ytrue, ypred)
        self.score_q2_cv = q2
        return q2

    def __call__(self, X, n_estimators):
        y = self._model_func(X, n_estimators)
        return y
    
def get_margins(dist):
    """
    """
    dim = dist.getDimension()
    margins = []
    for i in range(dim):
        margins.append(dist.getMarginal(i))
    return margins

def sample_dist(dist, n_sample, sampling):
    """
    """
    if sampling == 'lhs':
        lhs = ot.LHSExperiment(dist, n_sample)
        input_sample = lhs.generate()
    elif sampling == 'monte-carlo':
        input_sample = dist.getSample(n_sample)
    else:
        raise ValueError('Unknow sampling: {0}'.format(sampling))
    return input_sample

def change_distribution(dist, sampling_type, alpha):
    """Slightly change the distribution for a custom sampling.
    
    Parameters
    ----------
    
    """
    dist = ot.ComposedDistribution(dist)
    if sampling_type == 'uniform':
        margins = get_margins(dist)
        new_margins = []
        for marginal in margins:
            up = marginal.computeQuantile(alpha)[0]
            down = marginal.computeQuantile(1. - alpha)[0]
            new_margins.append(ot.Uniform(down, up))
        dist = ot.ComposedDistribution(new_margins)
    elif sampling_type == 'independent':
        dim = dist.getDimension()
        dist.setCopula(ot.IndependentCopula(dim))
    elif sampling_type == 'classic':
        pass
    else:
        raise ValueError('Unknow sampling type')
        
    return dist