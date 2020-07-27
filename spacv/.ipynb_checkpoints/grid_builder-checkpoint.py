import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from sklearn.neighbors import BallTree

from .utils import convert_geodataframe, geometry_to_2d, convert_numpy

def construct_blocks(XYs, tiles_x, tiles_y, method='unique', **kwargs):
    """
    Build grid over study area with user-defined number of tiles.
    
    Parameters
    ----------
    XYs : Geoseries series
        Series containing X and Y coordinates.
    tiles_x : integer
        Integer declaring number of tiles along X axis.
    tiles_y : integer
        Integer declaring number of tiles along Y axis.
    method : string
        String identifying grid assignment method.
    n_groups : integer
        Integer declaring number of randomized block groups.
    direction : string
        String stating direction of systematic diagonal pattern.
        
    Returns
    -------
    grid : GeoDataFrame Dataframe
        GeoDataFrame with square grids as shapely polygons.
    
    """
    # Construct grid of square polygons of defined size
    grid = construct_grid(XYs, tiles_x, tiles_y)
    
    # Set grid assignment method
    if method == 'unique':
        grid['grid_id'] = grid.index
    elif method == 'systematic':
        grid['grid_id'] = assign_systematic(grid, tiles_x, tiles_y, kwargs.get('direction'))
    elif method == 'random':
        grid['grid_id'] = assign_randomized(grid, kwargs.get('n_groups'))
    elif method == 'optimized_random':
        grid['grid_id'] = assign_optimized_random(grid, XYs, kwargs.get('data'), 
                                                             kwargs.get('n_groups'),
                                                             kwargs.get('n_sims'), 
                                                             kwargs.get('distance_metric'))
    else:
        raise ValueError('Method not recognised. Choose between: unique, systematic, random or optimized_random.')
    
    return grid
    
def construct_grid(XYs, tiles_x, tiles_y):
    """
    Build grid over study area with user-defined number of tiles.
    
    Parameters
    ----------
    XYs : Geoseries series
        Series containing X and Y coordinates.
    tiles_x : integer
        Integer declaring number of tiles along X axis.
    tiles_y : integer
        Integer declaring number of tiles along Y axis.
        
    Returns
    -------
    grid : GeoDataFrame Dataframe
        GeoDataFrame with square grids as shapely polygons.
    
    """
    minx, miny, maxx, maxy = XYs.total_bounds

    # Set length and height of tiles
    dx = (maxx - minx) / tiles_x
    dy = (maxy - miny) / tiles_y

    # Build grid of polygons
    polys = []
    for tile_y in range(0, tiles_y):
        for tile_x in range(0, tiles_x):
            polys.append(
                Polygon([
                    # bottom-left
                    (minx + (tile_x * dx), miny + ( tile_y * dy ) ), 
                    # bottom-right
                    (minx + ((tile_x + 1) * dx), miny + ( tile_y * dy  ) ), 
                    # top-right
                    (minx + ( (tile_x + 1) * dx), miny  + ( (tile_y + 1) * dy) ), 
                    #top-left
                    (minx + (tile_x * dx), miny + (tile_y + 1) * dy )
            ]))
    grid = gpd.GeoDataFrame({'geometry':polys})
            
    return grid    

def assign_systematic(grid, tiles_x, tiles_y, direction='diagonal'):
    """
    Set grid pattern as systematic by assigning grid IDs along diagonals, normal and anti-diagonal.
    """
    # Reshape length of grid to matrix
    sys_matrix = np.arange(0, tiles_x * tiles_y) \
                   .reshape(tiles_y, tiles_x)
    length, width = sys_matrix.shape

    # Set systematic pattern, diagonal or anti-diagonal
    if direction == 'diagonal':
        diags = [sys_matrix.diagonal(i) 
                     for i in range(width-1, -length,-1)]
    elif direction == 'anti':
        diags = [sys_matrix[::-1,:].diagonal(i) 
                     for i in range(-length+1, width)]
    else:
        raise ValueError('Direction of systematic pattern not recognised. Choose between: diagonal or anti.')

    # Construct lookup between diagonal element indices and grid dataframe
    systematic_lookup = dict([
                            tuple([element, key]) 
                                for key, diag in enumerate(diags) 
                                    for element in diag
                        ])
    grid_id = grid.index.map(systematic_lookup)

    return grid_id


def assign_randomized(grid, n_groups=5):
    """
    Set grid pattern as randomized by randomly assigning grid IDs.
    """
    # Determine number of randomized groups
    n_random_grps = np.arange(0, n_groups)
    n_grids = grid.shape[0]
    
    # Allocate random group id to each grid row
    grid_id = np.random.choice(n_random_grps, size=n_grids, replace=True)
    
    return grid_id

def assign_optimized_random(grid, XYs, data, n_groups=5, n_sims=10, distance_metric='euclidean'):
    """
    Set grid pattern as optimized random by taking grid IDs that minimize dissimilarity between folds.
    """
    if data is None:
        raise ValueError(
            'Data parameter must be supplied to spacv.HBLOCK() to compute fold dissimilarity.'
        )
    
    data = convert_numpy(data)
    
    # Build dictionary of grid IDs with paired SSR for dissimilarity 
    optimized_grid = {}
    for sim in range(n_sims):
        grid_id = assign_randomized(grid, n_groups, distance_metric)
        grid['grid_id'] = grid_id
        folds = assign_pt_to_grid(XYs, grid)

        # Scale for SSR calculation
        X = (data - data.mean(axis=0)) / data.std(axis=0)
        Xbar = X.mean(axis=0)
        X_grid_means = np.array([ X[v].mean(axis=0) 
                                     for k, v in folds.groupby('grid_id').groups.items()])

        # Calculate dissimilarity between folds and mean values across all data 
        sse = sum(
            sum((X_grid_means - Xbar)**2)
        )

        optimized_grid.update( {sim : {'sse': sse, 'grid_id': grid_id}} )

    # Take the optimized grid as one that minimises dissimilarity between folds
    minimised_obj = min(optimized_grid, key = lambda x : optimized_grid[x]['sse'])
    grid_id = optimized_grid[minimised_obj]['grid_id']

    return grid_id


def assign_pt_to_grid(XYs, grid, distance_metric='euclidean'):
    """
    Spatial join pts to grids. Reassign border points to nearest grid based on centroid distance. 
    """
    # Sjoin pts to grid polygons
    XYs = convert_geodataframe(XYs)
    
    # Equate spatial reference systems if defined 
    if not grid.crs == XYs.crs:
        grid.crs = XYs.crs        
    XYs = gpd.sjoin(XYs, grid, how='left' , op='within')

    # In rare cases, points will sit at the border separating two grids
    if XYs['grid_id'].isna().any():
        # Find border pts and assign to nearest grid centroid
        grid_centroid = grid.geometry.centroid
        grid_centroid = geometry_to_2d(grid_centroid)
        border_pt_index = XYs['grid_id'].isna()
        border_pts = XYs[border_pt_index].geometry
        border_pts = geometry_to_2d(border_pts)        
        tree = BallTree(grid_centroid, metric=distance_metric) 
        grid_id  = tree.query(border_pts, k=1, return_distance=False)
        
        # Update border pt grid IDs
        XYs.loc[border_pt_index, 'grid_id'] = grid_id
        XYs = XYs.drop(columns=['index_right'])

    return XYs