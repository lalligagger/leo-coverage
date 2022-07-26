import numpy as np
from skyfield.api import load, wgs84, Distance
from skyfield.toposlib import ITRSPosition
from skyfield.framelib import itrs
from scipy.spatial.transform import Rotation

import shapely
import branca
import folium
import fiona
from shapely.geometry import Polygon
import geopandas as gpd
import pandas as pd
from stactools.core.utils import antimeridian
from rich import print
from rich.progress import track

gpd.options.use_pygeos = True

def gen_sats(sat_nos=[48915]):
    """
    Skyfield satellite lookup from Celestrack, based on catalog ID.
    Landsat 8 & 9 defaults.
    """
    sats = []
    for n in sat_nos:
        url = "https://celestrak.org/NORAD/elements/gp.php?CATNR={}".format(n)
        tle_filename = "tle-CATNR-{}.txt".format(n)
        sat = load.tle_file(url, filename=tle_filename)
        sats.append(sat)
    print("Satellite(s) Loaded from TLE:")
    for sat in sats:
        print(sat)
    return sats


def single_sat(sat_no):
    """
    Skyfield satellite lookup from Celestrack.
    """
    url = "https://celestrak.org/NORAD/elements/gp.php?CATNR={}".format(sat_no)
    tle_filename = "tle-CATNR-{}.txt".format(sat_no)
    sat = load.tle_file(url, filename=tle_filename)
    print("Satellite Loaded from TLE:")
    print(sat)
    return sat


def gen_times(start_yr=2022, start_mo=6, start_day=15, days=1, step_min=1):
    """
    Generate skyfield timespan over desired range.
    """
    ts = load.timescale()
    # times = ts.utc(start_yr, start_mo, start_day, 0, range(0, 60 * 24 * days, step_min))
    times = ts.utc(
        start_yr, start_mo, start_day, 0, np.arange(0, 60 * 24 * days, step_min)
    )

    print(
        "Propogation time: \n {} \nto \n {}".format(
            str(times[0].utc_datetime()), str(times[-1].utc_datetime())
        )
    )

    return times


def camera_model(
    name="instrument",
    fl=178,
    pitch=0.025,
    h_pix=1850,
    v_pix=1800,
):
    """
    Takes in instrument parameters and calculates the azimuth offset to generate azimuth angles to top corners, and the half-diagonal FOV in angle space.
    For v2, we use the horizontal and vertical FOVs instead, which are divided by 2 and applied in the test notebook as (az, el) offsets in LVLH...
    The complete function needs to be integrated somewhere below.
    Defaults are TIRS.
    """
    hfov_deg = np.degrees(h_pix * pitch / fl)
    vfov_deg = np.degrees(v_pix * pitch / fl)
    instrument = {
        "name": name,
        "fl": fl,
        "pitch": pitch,
        "h_pix": h_pix,
        "v_pix": v_pix,
        "hfov_deg": hfov_deg,  # update to atan
        "vfov_deg": vfov_deg,  # update to atan
        "half_diag_deg": np.degrees(
            (pitch / fl) * np.sqrt(h_pix**2 + v_pix**2) / 2
        ),
        "az1": np.degrees(np.arctan2(h_pix, v_pix)),
        "az2": 360 - np.degrees(np.arctan2(h_pix, v_pix)),
        "corners": {
            "c1": {"X": -hfov_deg / 2, "Y": vfov_deg / 2},
            "c2": {"X": hfov_deg / 2, "Y": vfov_deg / 2},
            "c3": {"X": hfov_deg / 2, "Y": -vfov_deg / 2},
            "c4": {"X": -hfov_deg / 2, "Y": -vfov_deg / 2},
        },
    }
    return instrument


def los_to_earth(position, pointing):
    """Find the intersection of a pointing vector with the Earth
    Finds the intersection of a pointing vector u and starting point s with the WGS-84 geoid
    Source: https://stephenhartzell.medium.com/satellite-line-of-sight-intersection-with-earth-d786b4a6a9b6
    Args:
        position (np.array): length 3 array defining the starting point location(s) in meters
        pointing (np.array): length 3 array defining the pointing vector(s) (must be a unit vector)
    Returns:
        np.array: length 3 defining the point(s) of intersection with the surface of the Earth in meters
    """

    a = 6378.137
    b = 6378.137
    c = 6356.752314245
    x, y, z = position
    u, v, w = pointing

    value = (
        -(a**2) * b**2 * w * z - a**2 * c**2 * v * y - b**2 * c**2 * u * x
    )
    radical = (
        a**2 * b**2 * w**2
        + a**2 * c**2 * v**2
        - a**2 * v**2 * z**2
        + 2 * a**2 * v * w * y * z
        - a**2 * w**2 * y**2
        + b**2 * c**2 * u**2
        - b**2 * u**2 * z**2
        + 2 * b**2 * u * w * x * z
        - b**2 * w**2 * x**2
        - c**2 * u**2 * y**2
        + 2 * c**2 * u * v * x * y
        - c**2 * v**2 * x**2
    )
    magnitude = (
        a**2 * b**2 * w**2 + a**2 * c**2 * v**2 + b**2 * c**2 * u**2
    )

    if radical < 0:
        raise ValueError("The Line-of-Sight vector does not point toward the Earth")
    d = (value - a * b * c * np.sqrt(radical)) / magnitude

    if d < 0:
        raise ValueError("The Line-of-Sight vector does not point toward the Earth")

    return np.array(
        [
            x + d * u,
            y + d * v,
            z + d * w,
        ]
    )


def get_los(sat, time):
    """
    No longer used. Keep as convenience function?
    """

    geo = sat.at(time)
    xyz_dist_rates = geo.frame_xyz_and_velocity(itrs)
    xyz_dist = xyz_dist_rates[0]
    nadir = -xyz_dist.km / xyz_dist.length().km

    xyz_vel = xyz_dist_rates[1]

    los_xyz = los_to_earth(xyz_dist.km, nadir)  # input is meters

    los = Distance(km=los_xyz)
    los_itrs = ITRSPosition(los)
    los_itrs.at(time).frame_xyz(itrs).km

    los_lat, los_lon = wgs84.latlon_of(los_itrs.at(time))
    d = np.sqrt(np.sum(np.square(xyz_dist.km - los_xyz)))

    return los_lat, los_lon, d


def get_lvlh_pointing(sat, time):
    """
    This function defines the unit vectors which make up the satellites LVLH (local vertical, local horizontal) frame.

    It uses the instantaenous position and bearing to calculate the unit vectors using cross-products.

    from: https://ai-solutions.com/_freeflyeruniversityguide/attitude_reference_frames.htm
    """
    geo = sat.at(time)
    xyz_dist_rates = geo.frame_xyz_and_velocity(itrs)
    xyz_dist = xyz_dist_rates[0]
    local_vertical = -xyz_dist.km / xyz_dist.length().km

    xyz_vel = xyz_dist_rates[1]
    bearing = xyz_vel.km_per_s / np.linalg.norm(xyz_vel.km_per_s)

    neg_orb_normal = -np.cross(local_vertical, bearing)
    local_horizontal = np.cross(neg_orb_normal, local_vertical)
    lvlh = {"X": local_horizontal, "Y": neg_orb_normal, "Z": local_vertical}

    return lvlh#, local_vertical


def get_inst_fov(sat, time, inst):
    """
    This function takes in the camera model with satellite object and time, to calculate the instantaneous FOV.
    
    Can only take a single time value as input, which must be a skyfield timescale object.
    """
    # lvlh, pointing = get_lvlh_pointing(sat, time)
    lvlh = get_lvlh_pointing(sat, time)
    xyz_dist_rates = sat.at(time).frame_xyz_and_velocity(itrs)
    xyz_dist = xyz_dist_rates[0]
    z_rate = xyz_dist_rates[1]

    # Empty dict to populate with lat/ lons, mapped from cs_dict in angle space
    cs_lla_dict = {
        "c1": {"lat": None, "lon": None},
        "c2": {"lat": None, "lon": None},
        "c3": {"lat": None, "lon": None},
        "c4": {"lat": None, "lon": None},
    }

    # For each corner in FOV...
    for c in inst["corners"]:
        # Generate X and Y rotation vectors
        rot_X_deg = inst["corners"][c]["X"]
        rot_X_rad = np.radians(rot_X_deg)
        rot_X_ax = lvlh["X"]

        rot_Y_deg = inst["corners"][c]["Y"]
        rot_Y_rad = np.radians(rot_Y_deg)
        rot_Y_ax = lvlh["Y"]

        # Rotations with scipy:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html

        # Rotation about X
        rot_X_vec = rot_X_rad * rot_X_ax
        rot_X = Rotation.from_rotvec(rot_X_vec)

        # Rotation about Y 
        rot_Y_vec = rot_Y_rad * rot_Y_ax
        rot_Y = Rotation.from_rotvec(rot_Y_vec)

        # Calculate final LOS
        rot = rot_X * rot_Y
        # rot = Rotation.from_rotvec([rot_X_vec, rot_Y_vec])
        los_XY = rot.apply(lvlh["Z"])

        # Calculate Earth intercept of LOS, create ITRS position object
        los_xyz = los_to_earth(xyz_dist.km, los_XY)
        los_itrs = ITRSPosition(Distance(km=los_xyz))

        # Convert intercept lat/ lon from ITRS frame
        los_lat, los_lon = wgs84.latlon_of(los_itrs.at(time))
        cs_lla_dict[c]["lat"] = los_lat.degrees
        cs_lla_dict[c]["lon"] = los_lon.degrees

    return cs_lla_dict


def forecast_fovs(sat, times, inst):
    """
    This function handles FOV forecasting in batches (over multiple times). It takes in satellite, times as datetime array, and camera model.

    The datetime is converted to skyfield timescale and passed to a temporary function which can be used with df.apply().
    """
    ts = load.timescale()

    df = gpd.GeoDataFrame({'datetime': times.utc_datetime()})
    df["satellite"] = sat.name
    df["id"] = np.abs(sat.target)
    df["time"] = times.utc_strftime()

    def _get_inst_fov(time):
        ts_time = ts.from_datetime(time)
        # return Polygon(get_inst_fov(sat, ts_time, inst))
        cs_lla_dict = get_inst_fov(sat, ts_time, inst)
        return Polygon(
        [
            (cs_lla_dict["c1"]["lon"], cs_lla_dict["c1"]["lat"]),
            (cs_lla_dict["c2"]["lon"], cs_lla_dict["c2"]["lat"]),
            (cs_lla_dict["c3"]["lon"], cs_lla_dict["c3"]["lat"]),
            (cs_lla_dict["c4"]["lon"], cs_lla_dict["c4"]["lat"]),
            (cs_lla_dict["c1"]["lon"], cs_lla_dict["c1"]["lat"]),
        ]
    )

    fov_df = df.set_geometry(df['datetime'].apply(_get_inst_fov), crs="EPSG:4326")

    fov_df["lonspan"] = fov_df.bounds['maxx'] - fov_df.bounds['minx']
    mask = fov_df["lonspan"] > 20

    fov_df.loc[mask, "geometry"] = fov_df.loc[mask, "geometry"].apply(antimeridian.split)
    # fov_df.loc[mask, "geometry"] = None
    
    fov_df = fov_df.drop('lonspan', axis=1)

    return fov_df



def create_grid(bounds, xcell_size, ycell_size):
    """
    Create a grid of input bound and x/y cell sizes (all in lat/lon degrees), for revisit calculations.
    """
    (xmin, ymin, xmax, ymax) = bounds

    # Create grid of points with regular spacing in degrees
    # projection of the grid
    crs = "EPSG:4326"

    xcells = np.arange(xmin, xmax + xcell_size, xcell_size)
    ycells = np.arange(ymin, ymax + ycell_size, ycell_size)
    grid_shape = (len(xcells), len(ycells))

    # create the grid points in a loop
    grid_points = []
    for x0 in xcells:
        for y0 in ycells:
            grid_points.append(shapely.geometry.Point(x0, y0))

    grid = gpd.GeoDataFrame(grid_points, columns=["geometry"], crs=crs)

    return grid, grid_shape


def calculate_revisits(fov_df, aoi, grid_x=0.1, grid_y=0.1):
    """
    Calculates revisits based on a geodataframe of FOVs, geojson AOI, and user-defined grid spacing (lat/lon).

    Computation is done using gdf.sjoin and the return is a dataframe with row for each grid point.
    """
    # 1) Create a grid of equally spaced points
    grid, grid_shape = create_grid(aoi.total_bounds, grid_x, grid_y)

    # 2) Add "n_visits" column to grid using sjoin/ dissolve
    shapes = gpd.GeoDataFrame(fov_df.geometry)
    merged = gpd.sjoin(shapes, grid, how="left", predicate="intersects")
    merged[
        "n_visits"
    ] = 0  # this will be replaced with nan or positive int where n_visits > 0
    dissolve = merged.dissolve(
        by="index_right", aggfunc="count"
    )  # no difference in count vs. sum here?
    grid.loc[dissolve.index, "n_visits"] = dissolve.n_visits.values

    return grid, grid_shape


## Plotting Revisit Map
def revisit_map(grid, grid_shape, grid_x, grid_y):
    ## Form 2D array of n_visits based on grid shape
    img = np.rot90(grid.n_visits.values.reshape(grid_shape))

    ## Create colormap and apply to img
    ## TODO: Make this our geoTIFF item for STAC catalog
    colormap = branca.colormap.step.viridis.scale(1, grid.n_visits.max())
    # colormap = branca.colormap.step.viridis.scale(1, 2)

    def colorfunc(x):
        if np.isnan(x):
            return (0, 0, 0, 0)
        else:
            return colormap.rgba_bytes_tuple(x)

    # Apply cmap to img array and rearrange for RGBA
    cmap = np.vectorize(colorfunc)
    rgba_img = np.array(cmap(img))
    rgba_img = np.moveaxis(rgba_img, 0, 2)

    # Update image corner bounds based on cell size
    xmin, ymin, xmax, ymax = grid.total_bounds
    xmin = xmin - grid_x / 2
    ymin = ymin - grid_y / 2
    xmax = xmax + grid_x / 2
    ymax = ymax + grid_y / 2

    m = folium.Map()
    m.fit_bounds([[ymin, xmin], [ymax, xmax]])
    m.add_child(
        folium.raster_layers.ImageOverlay(
            rgba_img,
            opacity=0.4,
            mercator_project=True,  # crs="EPSG:4326",
            bounds=[[ymin, xmin], [ymax, xmax]],
        )
    )
    colormap.add_to(m)
    # m.save("./tmp/revisits_map.html")
    return m


if __name__ == "__main__":
    from landsat import Scene, Instrument, Platform
    from datetime import datetime, timezone, timedelta

    start_dt = datetime.fromisoformat(Scene.start_utc)
    num_days = 1
    xcell_size = ycell_size = 0.1

    tles = gen_sats(
        # sat_nos=[Platform.norad_id] # How to best handle multiple platforms? (TLE vs. SPG4 model too)
        sat_nos=[39084, 49260]
    )

    inst = camera_model(
        name=Instrument.name,
        fl=Instrument.focal_length_mm,
        pitch=Instrument.pitch_um * 1e-3,
        h_pix=Instrument.rows,
        v_pix=Instrument.cols,
    )

    times = gen_times(
        start_yr=start_dt.year,
        start_mo=start_dt.month,
        start_day=start_dt.day,
        days=num_days,
        step_min=Instrument.img_period,
    )

    ## Batch FOV generation over N satellites
    gdfs = []

    for tle in track(tles, description="Processing..."):
        # for tle in tles:
        sat = tle[0]
        fov_df = forecast_fovs(sat, times, inst)
        gdfs.append(fov_df)
    fov_df = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs="epsg:4326")

    ## Create cmap for unique satellites and create color column
    sat_ids = list(fov_df["id"].unique()).sort()
    cmap = branca.colormap.StepColormap(
        ["red", "blue"], sat_ids, vmin=139084, vmax=149260
    )
    fov_df["color"] = fov_df["id"].apply(cmap)

    ## Save to geojson based on sat name
    for satname in fov_df.satellite.unique():
        with fiona.Env(OSR_WKT_FORMAT="WKT2_2018"):
            fov_df[fov_df.satellite == satname].to_file(
                "./tmp/{}_fovs.geojson".format(satname.replace(" ", "_"))
            )

    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    aoi = gpd.read_file(
        "./aois/eastern_us.geojson"
    ).geometry  # ...so use AOI for subsection of US

    ## Filter fov_df by aoi
    xmin, ymin, xmax, ymax = aoi.total_bounds
    fov_df = fov_df.cx[xmin:xmax, ymin:ymax]

    ## Coverage data analysis for single satellite/ batch of satellites
    grid, grid_shape = calculate_revisits(
        fov_df, aoi, grid_x=xcell_size, grid_y=ycell_size
    )
    with fiona.Env(OSR_WKT_FORMAT="WKT2_2018"):
        grid.to_file("./tmp/all_revisits.geojson")
    print(grid.n_visits.fillna(0).describe())

    ## Plotting FOVs

    ## Make a folium map
    m = fov_df.drop('datetime', axis=1).explore(color="color", tooltip=["satellite", "time"])

    ## Add WRS2
    # wrs2 = gpd.read_file('./WRS2_descending_0/WRS2_descending.shp')
    # wrs2 = wrs2.cx[xmin: xmax, ymin: ymax]
    # folium.GeoJson(data=wrs2["geometry"], overlay=False).add_to(m)

    m.save("./tmp/fovs_map.html")

    ## Plotting Revisits

    m = revisit_map(grid, grid_shape, grid_x=xcell_size, grid_y=ycell_size)
    m.save("./tmp/revisit_map.html")
