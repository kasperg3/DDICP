
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import zipfile
import requests
from io import BytesIO
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
import rpy2.robjects.packages as rpackages
import tempfile
from shapely.geometry import Point, Polygon
from shapely import wkt
rpy2.robjects.numpy2ri.activate()
zip_url = 'https://tylermw.com/data/dem_01.tif.zip'
response = requests.get(zip_url)
with zipfile.ZipFile(BytesIO(response.content)) as thezip:
    thezip.extractall('/home/kang/workspace/EnvironmentalModelling/')
zip_url = '/home/kang/workspace/EnvironmentalModelling/dem_01.tif'
with rio.open(zip_url) as f:
    z = f.read(1)
    
    # Example Shapely objects
    point = Point(1, 1)
    polygon = Polygon([(0, 0), (1, 1), (1, 0)])

    # Convert Shapely objects to WKT (Well-Known Text)
    point_wkt = point.wkt
    polygon_wkt = polygon.wkt

    # Parse WKT in R
    ro.globalenv['point_wkt'] = point_wkt
    ro.globalenv['polygon_wkt'] = polygon_wkt
    # Example NumPy array
    np_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Convert NumPy array to R matrix
    r_matrix = ro.r.matrix(np_array, nrow=np_array.shape[0], ncol=np_array.shape[1])
    ro.globalenv['r_matrix'] = r_matrix

    # Print the R matrix
    ro.r('print(r_matrix)')

    ro.r('''
    library(sf)
    point <- st_as_sfc(point_wkt)
    polygon <- st_as_sfc(polygon_wkt)
    print(point)
    print(polygon)
    ''')
    
def rayshade(z, img_path=None, zscale=10, fov=0, theta=135, zoom=0.75, phi=45, windowsize=(1000, 1000)):
    
    # Output path.
    if not img_path:
        img_path = tempfile.NamedTemporaryFile(suffix='.png').name
    
    # Import needed packages.
    rayshader = rpackages.importr('rayshader')
    
    # Convert array to matrix.
    z = np.asarray(z)
    rows, cols = z.shape
    z_mat = ro.r.matrix(z, nrow=rows, ncol=cols)
    ro.globalenv['elmat'] = z_mat
    
    # Save python state to r.
    ro.globalenv['img_path'] = img_path
    ro.globalenv['zscale'] = zscale
    ro.globalenv['fov'] = fov
    ro.globalenv['theta'] = theta
    ro.globalenv['zoom'] = zoom
    ro.globalenv['phi'] = phi
    ro.globalenv['windowsize'] = ro.IntVector(windowsize)
    
    # Do the render.
    ro.r('''
        elmat %>%
          sphere_shade(texture = "desert") %>%
          add_water(detect_water(elmat), color = "desert") %>%
          add_shadow(ray_shade(elmat, zscale = zscale), 0.5) %>%
          add_shadow(ambient_shade(elmat), 0) %>%
          plot_3d(elmat, zscale = zscale, fov = fov, theta = theta, zoom = zoom, phi = phi, windowsize = windowsize)
    ''')
    
    # Return path.
    return img_path
img_path = rayshade(z)

plt.imshow(z)
plt.show()
