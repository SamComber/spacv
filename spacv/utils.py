import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd

def geometry_to_2d(geometry):
    return np.array(list(map(lambda x : (x.x, x.y), geometry)))

def convert_geoseries(XYs):
    if isinstance(XYs, gpd.GeoDataFrame):
        XYs = XYs.geometry
    if isinstance(XYs, np.ndarray):
        XYs= gpd.GeoSeries( [Point(xy) for xy in XYs] )
    return XYs

def convert_geodataframe(XYs):
    if isinstance(XYs, gpd.GeoSeries):
        XYs = gpd.GeoDataFrame({'geometry':XYs})
    if isinstance(XYs, np.ndarray):
        XYs = gpd.GeoDataFrame({'geometry': [Point(xy) for xy in XYs]} )
    return XYs

def convert_numpy(X):
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        return X.values
        