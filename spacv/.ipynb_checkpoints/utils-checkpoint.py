def variogram_at_lag(XYs, lag, bw):
    '''
    Return semivariance values are defined lags.
    
    Parameters
    ----------
    XYs : Geoseries series
        Series containing X and Y coordinates.
    lag : array or list
        Array containing distance lags to obtain semivariances.
    bw : integer or float
        Bandwidth, plus and minus to calculate semivariance. 
        
    Returns
    -------
    semivariances : Array of floats
        Array of semivariances for each lag
    '''
    
    paired_distances = pdist(XYs.iloc[:, :2])
    pd_m = squareform(paired_distances)
    n = pd_m.shape[0]

    ssvs = []
    for i in range(n):
        for j in range(i+1, n):
            if ( pd_m[i,j] >= lag-bw) and ( pd_m[i,j] <= lag+bw):
                ssvs.append( (XYs.iloc[i,2] - XYs.iloc[j,2])**2 )
    
    semivariances = np.sum(ssvs) / (2.0 * len(ssvs) )
    
    return semivariances