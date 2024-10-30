import numpy as np 
import scipy.stats as ss 
import matplotlib.pyplot as plt

def qq_plot(Y_ref, Y_alt, Q=99, B=None, residual=False, ax=None, figsize=None, xlabel="$\\hat{Y}$ quantiles", ylabel="$Y$ quantiles", linestyle="", markerstyle=".", linewidth=1, markersize=5, color="tab:orange", diag_color="tab:blue", diag_linestyle="--", diag_alpha=0.1):
    """
    Create a quantile-quantile (Q-Q) plot.

    Parameters
    ----------
    Y_ref : array-like
        One-dimensional array containing the reference sample (to be plotted on the x-axis)
    Y_alt : array-like
        One-dimensional array containing the alternate sample to be compared to Y_ref
    Q : int or None, optional
        Number of quantiles to compute (default=99; i.e. percentiles).
        If None, will attempt an unbinned Q-Q plot by plotting np.sort(Y_alt) vs. np.sort(Y_ref)
    B : int or None, optional
        Number of bootstrap realizations to use for computing the uncertainty envelope.
        If None (default), no envelope will be computed.
    residual : bool, optional
        Plot the residuals about the diagonal. Default is False.
    ax : matplotlib.axes.Axes, optional
        Existing axis to add the plot to
    figsize : two-tuple, optional
        If ax is None, will create a new figure of this size. Default is (6.4, 4.8) for a Q-Q residual plot, or (6.4, 6.4) for a Q-Q plot
    xlabel : string, optional
        Label to show on the reference (horizontal) axis. Default is "$\\hat{Y}$ quantiles"
    ylabel : string, optional
        Label to show on the alternate (vertical) axis. Default is "$Y$ quantiles".
    linestyle : string, optional
        Linestyle for the Q-Q plot. Default is "" (no line).
    markerstyle : string, optional
        Marker style to use for the quantiles. Default is "."
    linewidth : float or None, optional
        Linewidth for the Q-Q plot. Default is 1
    markersize : float, optional
        Markersize for the quantiles. Default is 5.
    color : string, optional
        Colour for the Q-Q plot. Default is "tab:orange"
    diag_color : string, optional
        Colour for the diagonal. Default is "tab:blue"
    diag_linestyle : string, optional
        Linestyle of the diagonal. Default is "--"
    diag_alpha : float, optional
        Alpha of the bootstrapped confidence region. Default is 0.1.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes of the plot
    """

    # do some checking of the inputs
    if isinstance(Y_ref, list):
        Y_ref = np.array(Y_ref)
    if isinstance(Y_alt, list):
        Y_alt = np.array(Y_alt)
    if Y_ref.ndim > 1 or Y_alt.ndim > 1:
        raise ValueError("Input arrays should be one-dimensional. Found ndim = {:d} and {:d} for Y_ref and Y_alt.".format(Y_ref.ndim, Y_alt.ndim))
    if Q is None and len(Y_ref) != len(Y_alt):
        raise ValueError("Unbinned Q-Q plot only implemented for len(Y_ref)==len(Y_alt). Found len(Y_ref)={:d} and len(Y_alt)={:d}. For an unbinned plot, consider downsampling one of the inputs.".format(len(Y_ref), len(Y_alt)))

    # set up axis if not provided
    if ax is None:
        fig = plt.figure(figsize=(6.4, 4.8 if residual else 6.4))
        ax = fig.gca()
    # unbinned version
    if Q is None:
        # limits
        Y_min = np.min([Y_ref, Y_alt])
        Y_max = np.max([Y_ref, Y_alt])
        # quantiles
        Y_ref_quantiles = np.sort(Y_ref)
        Y_alt_quantiles = np.sort(Y_alt)
    # binned version
    else:
        q = np.linspace(1.0/(Q+1), 1.0 - 1.0/(Q+1), Q)
        # quantiles
        Y_ref_quantiles = np.quantile(Y_ref, q)
        Y_alt_quantiles = np.quantile(Y_alt, q)
        # limits
        Y_min = np.min([Y_ref_quantiles, Y_alt_quantiles])
        Y_max = np.max([Y_ref_quantiles, Y_alt_quantiles])

    # what to plot
    if residual:
        diag_plot = [0, 0]
        Y_plot = Y_alt_quantiles - Y_ref_quantiles
    else:
        diag_plot = [Y_min, Y_max]
        Y_plot = Y_alt_quantiles
    # plot quantiles vs. quantiles
    ax.plot([Y_min, Y_max], diag_plot, color=diag_color, linestyle=diag_linestyle, linewidth=linewidth)
    ax.plot(Y_ref_quantiles, Y_plot, color=color, linestyle=linestyle, linewidth=linewidth, marker=markerstyle, markersize=markersize)

    # bootstrap
    if B is not None:
        # resample
        Y_b = np.random.choice(Y_ref, size=(B, len(Y_ref)))
        # compute quantiles of bootstrapped realisations
        if Q is None:
            Y_b_quantiles = np.sort(Y_b, axis=1)
        else:
            Y_b_quantiles = np.quantile(Y_b, q, axis=1).T
        # plot
        Y_b_std = np.std(Y_b_quantiles, axis=0)
        if residual:
            ax.fill_between(Y_ref_quantiles, Y_b_std, -Y_b_std, alpha=diag_alpha, color=diag_color, linewidth=0)
            ax.fill_between(Y_ref_quantiles, 2*Y_b_std, -2*Y_b_std, alpha=diag_alpha, color=diag_color, linewidth=0)
        else:
            ax.fill_between(Y_ref_quantiles, Y_ref_quantiles+Y_b_std, Y_ref_quantiles-Y_b_std, alpha=diag_alpha, color=diag_color, linewidth=0)
            ax.fill_between(Y_ref_quantiles, Y_ref_quantiles+2*Y_b_std, Y_ref_quantiles-2*Y_b_std, alpha=diag_alpha, color=diag_color, linewidth=0)
    
    # set labels and return axes
    ax.set_xlabel(xlabel)
    if residual:
        ax.set_ylabel(ylabel + " $-$ " + xlabel)
    else:
        ax.set_ylabel(ylabel)
    return ax

def pp_plot(Y_ref, Y_alt, Q=None, B=None, residual=False, ax=None, figsize=None, xlabel="$\\hat{Y}$ cumulative prob.", ylabel="$Y$ cumulative prob.", linestyle="", markerstyle=".", linewidth=1, markersize=5, color="tab:orange", diag_color="tab:blue", diag_linestyle="--", diag_alpha=0.1):
    """
    Create a probability-probability (P-P) plot.

    Parameters
    ----------
    Y_ref : array-like
        One-dimensional array containing the reference sample (to be plotted on the x-axis)
    Y_alt : array-like
        One-dimensional array containing the alternate sample to be compared to Y_ref
    Q : int or None, optional
        Number of quantiles to compute (default=None).
        If None, will use the unbinned Y_ref
    B : int or None, optional
        Number of bootstrap realizations to use for computing the uncertainty envelope.
        If None (default), no envelope will be computed.
    residual : bool, optional
        Plot the residuals about the diagonal. Default is False.
    ax : matplotlib.axes.Axes, optional
        Existing axis to add the plot to
    figsize : two-tuple, optional
        If ax is None, will create a new figure of this size. Default is (6.4, 4.8) for a P-P residual plot, or (6.4, 6.4) for a P-P plot
    xlabel : string, optional
        Label to show on the reference (horizontal) axis. Default is "$\\hat{Y}$ cumulative prob."
    ylabel : string, optional
        Label to show on the alternate (vertical) axis. Default is "$Y$ cumulative prob.".
    linestyle : string, optional
        Linestyle for the P-P plot. Default is "" (no line).
    markerstyle : string, optional
        Marker style to use for the quantiles. Default is "."
    linewidth : float or None, optional
        Linewidth for the P-P plot. Default is 1
    markersize : float, optional
        Markersize for the quantiles. Default is 5.
    color : string, optional
        Colour for the P-P plot. Default is "tab:orange"
    diag_color : string, optional
        Colour for the diagonal. Default is "tab:blue"
    diag_linestyle : string, optional
        Linestyle of the diagonal. Default is "--"
    diag_alpha : float, optional
        Alpha of the bootstrapped confidence region. Default is 0.1.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes of the plot
    """

    # do some checking of the inputs
    if isinstance(Y_ref, list):
        Y_ref = np.array(Y_ref)
    if isinstance(Y_alt, list):
        Y_alt = np.array(Y_alt)
    if Y_ref.ndim > 1 or Y_alt.ndim > 1:
        raise ValueError("Input arrays should be one-dimensional. Found ndim = {:d} and {:d} for Y_ref and Y_alt.".format(Y_ref.ndim, Y_alt.ndim))

    # set up axis if not provided
    if ax is None:
        fig = plt.figure(figsize=(6.4, 4.8 if residual else 6.4))
        ax = fig.gca()
        if not residual:
            ax.set_aspect("equal")

    # unbinned version
    if Q is None:
        # cumulative probabilities
        Q = len(Y_ref)
        q = np.linspace(1.0/(Q+1), 1.0 - 1.0/(Q+1), Q)
        # quantiles
        Y_ref_quantiles = np.sort(Y_ref)
    # binned version
    else:
        # cumulative probabilities
        q = np.linspace(1.0/(Q+1), 1.0 - 1.0/(Q+1), Q)
        # quantiles
        Y_ref_quantiles = np.quantile(Y_ref, q)
    q_alt = ss.percentileofscore(Y_alt, Y_ref_quantiles)/100.0

    # what to plot
    if residual:
        diag_plot = [0, 0]
        q_plot = q_alt - q
    else:
        diag_plot = [0, 1]
        q_plot = q_alt
    # plot quantiles vs. quantiles
    ax.plot([0, 1], diag_plot, color=diag_color, linestyle=diag_linestyle, linewidth=linewidth)
    ax.plot(q, q_plot, color=color, linestyle=linestyle, linewidth=linewidth, marker=markerstyle, markersize=markersize)

    # bootstrap
    if B is not None:
        # resample
        Y_b = np.random.choice(Y_ref, size=(B, len(Y_ref)))
        # compute cumulative probabilities for bootstrapped realisations
        q_b =np.array([ss.percentileofscore(Y, Y_ref_quantiles)/100.0 for Y in Y_b])
        q_std = np.std(q_b, axis=0)
        if residual:
            ax.fill_between(q, q_std, -q_std, alpha=diag_alpha, color=diag_color, linewidth=0)
            ax.fill_between(q, 2*q_std, -2*q_std, alpha=diag_alpha, color=diag_color, linewidth=0)
        else:
            ax.fill_between(q, q+q_std, q-q_std, alpha=diag_alpha, color=diag_color, linewidth=0)
            ax.fill_between(q, q+2*q_std, q-2*q_std, alpha=diag_alpha, color=diag_color, linewidth=0)
    
    # set labels and return axes
    ax.set_xlabel(xlabel)
    if residual:
        ax.set_ylabel(ylabel + " $-$ " + xlabel)
    else:
        ax.set_ylabel(ylabel)
    return ax

def pca(X, X_alt=None, method="cov"):
    """
    Perform a principal component analysis (PCA).

    Parameters
    ----------
    X : array-like
        Data matrix to perform PCA on (shape N, D)
    X_alt : array-like, optional
        Second data matrix to be projected along the principal axes of the first
    method : "cov" or "svd"
        Method used to perform the PCA. The data covariance is used if method=="cov". Else, a singular value decomposition is used. Default is "cov".

    Returns
    -------
    l : numpy.array
        Length D array containing variances explained by the principal components (eigenvalues)
    V : numpy.array
        Matrix (D,D), whose dth column is the dth principal component (eigenvectors)
    Y : array-like
        Projection of X
    Y_alt : numpy.array
        Projection of X_alt (if Y_alt provided)
    """

    if method == "cov":
        C = np.cov(X, rowvar=False)
        l, V = np.linalg.eig(C)
        V = V[:,np.argsort(l)[::-1]] # sort descending
        l = np.sort(l)[::-1]
    elif method == "svd":
        X_ = X - np.mean(X, axis=0) # mean center
        U, S, V_ = np.linalg.svd(X_)
        V = V_.T
        l = S**2/(len(X)-1)
    else:
        raise ValueError("Method must be one of 'cov' or 'svd'. Found {}".format(method))

    if X_alt is None:
        return l, V, X@V
    else:
        return l, V, X@V, X_alt@V

def wp_univariate(Y, Y_alt, p=2):
    """
    Univariate Wasserstein distance.

    Parameters
    ----------
    Y : array-like
        One dimensional array
    Y_alt : array-like
        Second array
    p : int, optional
        Norm to use. Default is p=2 -> L2 norm.
    """

    # check inputs
    if isinstance(Y, list):
        Y = np.array(Y)
    if isinstance(Y_alt, list):
        Y_alt = np.array(Y_alt)
    if Y.ndim > 1 or Y_alt.ndim > 1:
        raise ValueError("Input arrays should be one-dimensional. Found ndim = {:d} and {:d} for Y and Y_alt.".format(Y.ndim, Y_alt.ndim))

    # smallest sample size
    N = np.min([len(Y), len(Y_alt)])

    if p == np.inf: # infinity norm
        w = lambda a, b : np.max(np.fabs(a - b))
    else: # Lp norm
        w = lambda a, b : np.mean(np.fabs(a - b)**p)**(1/p)
    if len(Y_alt) == len(Y):
        W = w(np.sort(Y), np.sort(Y_alt))
    elif len(Y_alt) > len(Y):
        W = w(np.sort(Y), np.random.choice(Y_alt, len(Y), replace=False))
    else:
        W = w(np.sort(Y_alt), np.random.choice(Y, len(Y_alt), replace=False))
    return W
