import seaborn as sns
from sklearn.neighbors import KDTree
from scipy.spatial.distance import pdist, squareform

sns.set_style("whitegrid") # pretty plots

def variogram_at_lag(XYs, y, lag, bw):
    '''
    Return semivariance values for defined lags.
    
    Parameters
    ----------
    XYs : Geoseries series
        Series containing X and Y coordinates.
    y : array or list
        Array containing response variables.
    lag : integer
        Distance lag to obtain semivariances.
    bw : integer or float
        Bandwidth, plus and minus to calculate semivariance. 
        
    Returns
    -------
    semivariances : Array of floats
        Array of semivariances for each lag
    '''
    
    XYs = np.array(list(map(lambda x : (x.x, x.y), XYs)))
    y = np.asarray(y)

    paired_distances = pdist(XYs)
    pd_m = squareform(paired_distances)
    n = pd_m.shape[0]

    ssvs = []
    for i in range(n):
        for j in range(i+1, n):
            if ( pd_m[i,j] >= lag-bw) and ( pd_m[i,j] <= lag+bw):
                ssvs.append( (y[i] - y[j])**2 )
    semivariance = np.sum(ssvs) / (2.0 * len(ssvs) )
    
    return semivariance

def plot_variogram(XYs, y, lags, bw, **kwargs):
    
    semivariances = [variogram_at_lag(XYs, y, lag, bw) for lag in lags] 
    
    figsize = kwargs.pop('figsize', (6,4))
    
    fig, ax = plt.subplots(1, figsize = figsize)
    ax.plot(lags, semivariances, '.-')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['left'].set_edgecolor('black')
    ax.spines['bottom'].set_edgecolor('black')
    
    ax.set_ylabel('Semivariance')
    ax.set_xlabel('Lags')
    
    return fig, ax


def aoa(test_data, training_data, model=None, variables='all', thres=0.95):
    '''
    Meyer and Pebesma (2020) Area of Applicability https://arxiv.org/pdf/2005.07939.pdf
    
    '''
    
    if len(training_data) <= 1:
        raise Exception('At least two training instances need to be specified.') 
    
    # Scale data and weight if applicable
    training_data = (training_data - np.mean(training_data)) / np.std(training_data)
    test_data = (test_data - np.mean(test_data)) / np.std(test_data)

    tree = KDTree(training_data, metric='euclidean') 
    mindist, _ = tree.query(test_data, k=1, return_distance=True)

    paired_distances = pdist(training_data)
    train_dist = squareform(paired_distances)

    train_dist_mean = train_dist.mean(axis=1)
    train_dist_avgmean = np.mean(train_dist_mean)


    mindist = mindist / train_dist_avgmean

    # Set diagonal nan to exclude from average
    np.fill_diagonal(train_dist, np.nan)

    # Define threshold for AOA
    train_dist_min = np.nanmin(train_dist, axis=1)
    aoa_train_stats = np.percentile(train_dist_min / train_dist_avgmean, 
                                    q = np.array([0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]))
    thres = np.percentile(train_dist_min / train_dist_avgmean, q = thres)

    # We choose the AOA as the area where the DI does not exceed the threshold
    out = mindist.reshape(-1)
    masked_result = np.repeat(1, len(mindist))
    masked_result[out > thres] = 0
    
    return out, masked_result