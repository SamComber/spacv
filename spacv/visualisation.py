import seaborn as sns
from sklearn.neighbors import BallTree
from scipy.spatial.distance import pdist, squareform

sns.set_style("whitegrid") # pretty plots

# ADD: create nicer plots, they're horrible

def variogram_at_lag(XYs, y, lag, bw):
    """
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
    """
    
    XYs = geometry_to_2d(XYs.geometry)
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

def aoa(new_data, 
        training_data, 
        model=None, 
        thres=0.95,
        fold_indices=None
       ):
    """
    Area of Applicability (AOA) measure for spatial prediction models from
    Meyer and Pebesma (2020) https://arxiv.org/abs/2005.07939
    
    """
    if len(training_data) <= 1:
        raise Exception('At least two training instances need to be specified.')
                    
    # Scale data 
    training_data = (training_data - np.mean(training_data)) / np.std(training_data)
    new_data = (new_data - np.mean(new_data)) / np.std(new_data)

    # Calculate nearest training instance to test data, return Euclidean distances
    tree = BallTree(training_data, metric='euclidean') 
    mindist, _ = tree.query(new_data, k=1, return_distance=True)

    # Build matrix of pairwise distances 
    paired_distances = pdist(training_data)
    train_dist = squareform(paired_distances)
    np.fill_diagonal(train_dist, np.nan)
    
    # Remove data points that are within the same fold
    if fold_indices:            
        # Get number of training instances in each fold
        instances_in_folds = [len(fold) for fold in fold_indices]
        instance_fold_id = np.repeat(np.arange(0, len(fold_indices)), instances_in_folds)

        # Create mapping between training instance and fold ID
        fold_indices = np.concatenate(fold_indices)
        folds = np.vstack((fold_indices, instance_fold_id)).T

        # Mask training points in same fold for DI measure calculation
        for i, row in enumerate(train_dist):
            mask = folds[:,0] == folds[:,0][i]
            train_dist[i, mask] = np.nan

    # Scale distance to nearest training point by average distance across training data
    train_dist_mean = np.nanmean(train_dist, axis=1)
    train_dist_avgmean = np.mean(train_dist_mean)
    mindist /= train_dist_avgmean    

    # Define threshold for AOA
    train_dist_min = np.nanmin(train_dist, axis=1)
    aoa_train_stats = np.quantile(train_dist_min / train_dist_avgmean, 
                                    q = np.array([0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]))
    thres = np.quantile(train_dist_min / train_dist_avgmean, q = thres)
    
    # We choose the AOA as the area where the DI does not exceed the threshold
    DIs = mindist.reshape(-1)
    masked_result = np.repeat(1, len(mindist))
    masked_result[out > thres] = 0
    
    return DIs, masked_result

def plot_aoa(new_data, training_data, columns, figsize, **kwargs):
    
    # Pop geometry for use later in plotting
    new_data = new_data.copy()
    new_data_geometry = new_data.pop('geometry')
    
    # Subset to variables
    new_data_aoa = new_data[columns]
    training_data_aoa = training_data[columns]
    
    DIs, masked_result = aoa(new_data_aoa, 
                             training_data_aoa, 
                             fold_indices=fold_indices)
    
    new_data.loc[:, 'DI'] = DIs
    new_data.loc[:, 'AOA'] = masked_result
    new_data.loc[:, 'geometry'] = new_data_geometry
    
    new_data = gpd.GeoDataFrame(new_data, geometry=new_data['geometry'])
        
    f,ax = plt.subplots(1, 2, figsize=figsize)
    
    new_data.plot(ax=ax[0], column='DI', legend=True, cmap='viridis', legend_kwds={'shrink':.7})
    training_data.plot(ax=ax[0], alpha=.1)
    
    new_data.plot(ax=ax[1], column='AOA', categorical=True, legend=True)
    training_data.plot(ax=ax[1], alpha=.1)

    ax[0].set_title('Dissimilarity index (DI)')
    ax[1].set_title('AOA');

    return f, ax