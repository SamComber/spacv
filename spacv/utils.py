def variogram_at_lag(XYs, lag, bw):
    
    paired_distances = pdist(XYs.iloc[:, :2])
    pd_m = squareform(paired_distances)
    n = pd_m.shape[0]

    ssvs = []
    for i in range(n):
        for j in range(i+1, n):
            if ( pd_m[i,j] >= lag-bw) and ( pd_m[i,j] <= lag+bw):
                ssvs.append( (XYs.iloc[i,2] - XYs.iloc[j,2])**2 )

    return np.sum(ssvs) / (2.0 * len(ssvs) )