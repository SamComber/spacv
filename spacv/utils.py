import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon, LineString, MultiPolygon

def geometry_to_2d(geometry):
    return np.array(list(map(lambda x : (x.x, x.y), geometry)))

def convert_geoseries(XYs):
    if isinstance(XYs, gpd.GeoDataFrame):
        XYs = XYs.geometry
    if isinstance(XYs, np.ndarray):
        XYs = gpd.GeoSeries( gpd.points_from_xy(XYs) )
    
    if isinstance(XYs, gpd.GeoDataFrame):
        if any(XYs.geom_type == 'Polygon') or any(XYs.geom_type == 'MultiPolygon'):
            XYs = XYs.geometry.centroid        
    return XYs

def convert_geodataframe(XYs):
    if isinstance(XYs, gpd.GeoSeries):
        XYs = gpd.GeoDataFrame({'geometry':XYs})
    if isinstance(XYs, np.ndarray):
        XYs = gpd.GeoDataFrame({'geometry': gpd.points_from_xy(XYs) })
    if isinstance(XYs, (Point, Polygon, LineString, MultiPolygon)):
        XYs = gpd.GeoDataFrame({'geometry': [XYs] })        
    return XYs

def convert_numpy(X):
    if isinstance(X, (pd.DataFrame, pd.Series, gpd.GeoDataFrame)):
        return X.values
    else:
        return X
    
def load_custom_polygon(custom_poly):
    if isinstance(custom_poly, (gpd.GeoDataFrame, gpd.GeoSeries)):
        return custom_poly.reset_index(drop=True)
    if isinstance(custom_poly, str):
        return gpd.read_file(custom_poly).reset_index(drop=True)
    