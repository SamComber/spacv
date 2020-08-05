import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, asPolygon
from matplotlib.collections import PolyCollection
from sklearn.neighbors import BallTree
from .utils import convert_geodataframe, geometry_to_2d, convert_numpy

__all__ = [
    "construct_blocks",
    "construct_square_grid",
    "construct_hex_grid"
]

def construct_blocks(XYs, tiles_x, tiles_y, method='unique', shape='square', 
                     direction='diagonal', data=None, n_groups=5, n_sims=10, 
                     distance_metric='euclidean', random_state=None):
    """
    Build a grid over study area with user-defined number of tiles. This
    function exposes a number of configurable parameters to design the shape
    of grid, but also the fold assignment technique. 
    
    Parameters
    ----------
    XYs : Geoseries series
        Series containing X and Y coordinates.
    tiles_x : integer
        Integer declaring number of tiles along X axis.
    tiles_y : integer
        Integer declaring number of tiles along Y axis.
    method : string
        String identifying grid assignment method. Options are
        unique, systematic, random and optimized_random.
    n_groups : integer
        Integer declaring number of randomized block groups.
    direction : string
        String stating direction of systematic diagonal pattern.
    random_state : int, RandomState instance or None, optional, default=None
        If int, random_state is the seed used by the random number generator.
        If None, the random number generator is the RandomState instance used
        by `np.random`.
        
    Returns
    -------
    grid : GeoDataFrame Dataframe
        GeoDataFrame with square grids as shapely polygons.

    """
    # Construct grid of polygons of defined size and shape
    grid = construct_grid(XYs, tiles_x, tiles_y, shape)
    
    # Set grid assignment method
    if method == 'unique':
        grid['grid_id'] = grid.index
    elif method == 'systematic':
        if shape != 'square':
            raise Exception("systematic grid assignment method does not work for irregular grids.")
        grid['grid_id'] = assign_systematic(grid, tiles_x, tiles_y, direction)
    elif method == 'random':
        grid['grid_id'] = assign_randomized(grid, n_groups, random_state)
    elif method == 'optimized_random':
        grid['grid_id'] = assign_optimized_random(grid, XYs, data, 
                                                             n_groups,
                                                             n_sims, 
                                                             distance_metric)
    else:
        raise ValueError("Method not recognised. Choose between: unique, systematic, random or optimized_random.")
    return grid
    
def construct_grid(XYs, tiles_x, tiles_y, shape):
    """
    Choose grid shape to build across bounds of study area.
    """
    if shape == 'square':
        return construct_square_grid(XYs, tiles_x, tiles_y)
    if shape == 'hex':
        return construct_hex_grid(XYs, tiles_x, tiles_y)
    

def construct_square_grid(XYs, tiles_x, tiles_y):
    """
    Build square grid over study area with user-defined number of tiles.
    
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

    rows = np.array(list(np.arange(0, tiles_x))*tiles_y)
    columns = np.repeat(np.arange(0, tiles_y), tiles_x)
    
    dx = (maxx - minx) / tiles_x
    dy = (maxy - miny) / tiles_y

    bottom_left = np.add(minx, np.multiply(rows, dx)), np.add(miny, np.multiply(columns, dy))
    bottom_right = np.add(minx, np.multiply(rows+1, dx)), np.add(miny, np.multiply(columns, dy))
    top_right = np.add(minx, np.multiply(rows+1, dx)), np.add(miny, np.multiply(columns+1, dy))
    top_left = np.add(minx, np.multiply(rows, dx)), np.add(miny, np.multiply(columns+1, dy))

    polys = np.vstack([bottom_left, bottom_right, top_right, top_left]).reshape(4,2,-1)
    polys = [asPolygon(polys[:,:,i]) for i in range(tiles_x*tiles_y)]

    grid = gpd.GeoDataFrame({'geometry':polys})

    return grid

def construct_hex_grid(XYs, tiles_x, tiles_y):
    """
    Build hexagon grid over study area with user-defined number of tiles.
    
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
        GeoDataFrame with hexagon grids as shapely polygons.
    
    """
    minx, miny, maxx, maxy = XYs.total_bounds
    padding = 1.e-9 * (maxx - minx)
    minx -= padding
    maxx += padding
    dx = (maxx - minx) / tiles_x
    dy = (maxy - miny) / tiles_y

    tiles_x1 = tiles_x + 1
    tiles_y1 = tiles_y + 1
    n = tiles_x1 * tiles_y1 + tiles_x * tiles_y

    offsets = np.zeros((n, 2), float)
    offsets[:tiles_x1 * tiles_y1, 0] = np.repeat(np.arange(tiles_x1), tiles_y1)
    offsets[:tiles_x1 * tiles_y1, 1] = np.tile(np.arange(tiles_y1), tiles_x1)
    offsets[tiles_x1 * tiles_y1:, 0] = np.repeat(np.arange(tiles_x) + 0.5, tiles_y)
    offsets[tiles_x1 * tiles_y1:, 1] = np.tile(np.arange(tiles_y), tiles_x) + 0.5
    
    offsets[:, 0] *= dx
    offsets[:, 1] *= dy
    offsets[:, 0] += minx
    offsets[:, 1] += miny

    polygon = [dx, dy / 3] * np.array(
        [[.5, -.5], [.5, .5], [0., 1.], [-.5, .5], [-.5, -.5], [0., -1.]])

    collection = PolyCollection([polygon],offsets=offsets)
    hex_polys = collection.get_paths()[0].vertices
    hex_array = []
    for xs,ys in collection.get_offsets():
        hex_x = np.add(hex_polys[:,0],  xs)
        hex_y = np.add(hex_polys[:,1],  ys)
        hex_array.append(Polygon(np.vstack([hex_x, hex_y]).T))

    hex_grid = gpd.GeoDataFrame({'geometry':hex_array})
    
    return hex_grid

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
        raise ValueError("Direction of systematic pattern not recognised. Choose between: diagonal or anti.")

    # Construct lookup between diagonal element indices and grid dataframe
    systematic_lookup = dict([
                            tuple([element, key]) 
                                for key, diag in enumerate(diags) 
                                    for element in diag
                        ])
    grid_id = grid.index.map(systematic_lookup)

    return grid_id


def assign_randomized(grid, n_groups=5, random_state=None):
    """
    Set grid pattern as randomized by randomly assigning grid IDs.
    """
    np.random.seed(random_state)
    
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
            "Data must be supplied to spacv.HBLOCK() for computing fold"
            " dissimilarity when using optimized_random method."
        )
    data = convert_numpy(data)
    # Build dictionary of grid IDs with paired SSR for dissimilarity 
    optimized_grid = {}
    for sim in range(n_sims):
        grid_id = assign_randomized(grid, n_groups)
        grid['grid_id'] = grid_id
        folds = assign_pt_to_grid(XYs, grid, distance_metric)

        # Scale for SSR calculation
        X = (data - data.mean(axis=0)) / data.std(axis=0)
        Xbar = X.mean(axis=0)
        X_grid_means = np.array([ X[v].mean(axis=0) 
                                     for _, v in folds.groupby('grid_id').groups.items()])
        # Calculate dissimilarity between folds and mean values across all data 
        sse = sum(
            sum((X_grid_means - Xbar)**2)
        )
        optimized_grid.update( {sim : {'sse': sse, 'grid_id': grid_id}} )

    # Take the optimized grid as one that minimises dissimilarity between folds
    minimised_obj = min(optimized_grid, key = lambda x : optimized_grid[x]['sse'])
    grid_id = optimized_grid[minimised_obj]['grid_id']
    return grid_id


def assign_pt_to_grid(XYs, grid, distance_metric='euclidean', random_state=None):
    """
    Spatial join pts to grids. Reassign border points to nearest grid based on centroid distance. 
    """
    np.random.seed(random_state)
    XYs = convert_geodataframe(XYs)   
    # Equate spatial reference systems if defined 
    if not grid.crs == XYs.crs:
        grid.crs = XYs.crs        
    XYs = gpd.sjoin(XYs, grid, how='left' , op='within')[['geometry', 'grid_id']]
        
    # In rare cases, points will sit at the border separating two grids
    if XYs['grid_id'].isna().any():
        # Find border pts and assign to nearest grid centroid
        grid_centroid = grid.geometry.centroid
        grid_centroid = geometry_to_2d(grid_centroid)
        border_pt_index = XYs['grid_id'].isna()
        border_pts = XYs[border_pt_index].geometry
        border_pts = geometry_to_2d(border_pts)      
        
        # Update border pt grid IDs
        tree = BallTree(grid_centroid, metric=distance_metric) 
        grid_id  = tree.query(border_pts, k=1, return_distance=False).flatten()
        grid_id = grid.loc[grid_id, 'grid_id'].values
        XYs.loc[border_pt_index, 'grid_id'] = grid_id
    return XYs