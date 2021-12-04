import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
from pymap3d.los import lookAtSpheroid
from shapely.geometry import Polygon
from skyfield.api import load, wgs84


def gen_sats(sat_nos=[39084, 49260]):
    """
    Skyfield satellite lookup from Celestrack, based on catalog ID.
    """
    sats = []
    for n in sat_nos:
        url = "https://celestrak.com/satcat/tle.php?CATNR={}".format(n)
        tle_filename = "tle-CATNR-{}.txt".format(n)
        sat = load.tle_file(url, filename=tle_filename)
        sats.append(sat)

    print("Satellite(s) Loaded from TLE:")
    for sat in sats:
        print(sat)

    return sats


def gen_times(start_yr=2021, start_mo=11, start_day=20, days=1, step_min=1):
    """
    Generate skyfield timespan over desired range.
    """
    ts = load.timescale()
    times = ts.utc(start_yr, start_mo, start_day, 0, range(0, 60 * 24 * days, step_min))

    print(
        "Propogation time: \n {} \nto \n {}".format(
            str(times[0].utc_datetime()), str(times[-1].utc_datetime())
        )
    )

    return times


def prop_orbits(satellites, times):
    """
    Using SGP4 propagator in skyfield, calculate lat/lon/alt (LLA) of sub-satellite point (SSP), and whether or not the satellite is sunlit.
    """
    # Load ephemeris (change to input variable?)
    eph = load("de421.bsp")

    dfs = []

    for n in range(len(satellites)):
        sat = satellites[n][0]
        print("Adding {}".format(sat.name))
        geos = sat.at(times)
        lat, lon = wgs84.latlon_of(geos)
        df = pd.DataFrame({"lat": lat.degrees, "lon": lon.degrees})
        df["altitude"] = wgs84.height_of(geos).km
        df["satellite"] = sat.name
        df["time"] = times.utc_datetime()
        df["sunlit"] = sat.at(times).is_sunlit(eph)
        dfs.append(df)

    df_out = pd.concat(dfs, ignore_index=True)  # .reset_index.drop("index", axis=1)#
    df_out = df_out.reindex(
        ["time", "satellite", "lat", "lon", "altitude", "bearing", "sunlit"], axis=1
    )
    return df_out


def add_bearing(df_in):
    """Calculate bearing based on nth and n+1th lat/long SSP.
    TODO:
    - Calculate bearing from instantaneous velocity?
    """
    dfs = []

    for sat in df_in.satellite.unique():
        # df = pd.DataFrame()
        df = df_in[df_in.satellite == sat].copy()
        df["bearing"] = np.nan
        for n in df[:-1].index:
            lat1 = np.radians(df.loc[n].lat)
            lat2 = np.radians(df.loc[n + 1].lat)
            lon1 = np.radians(df.loc[n].lon)
            lon2 = np.radians(df.loc[n + 1].lon)

            dLon = lon2 - lon1

            X = np.cos(lat2) * np.sin(dLon)
            Y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dLon)

            brng = np.degrees(np.arctan2(X, Y))
            if brng < 0:
                brng += 360
            df.at[n, "bearing"] = brng
            # df.loc[n, "bearing"] = brng

        dfs.append(df)
    df_out = pd.concat(dfs)

    return df_out


def gen_instrument(
    name="instrument", fl=178, pitch=0.025, h_pix=1850, v_pix=1800, mm=True
):
    """
    Takes in instrument parameters and calculates the azimuth offset to generate azimuth angles to top corners, and the half-diagonal FOV in angle space.
    These are used in gen_los_offsets to create the full 4-corner instrument frustrum in angle space.

    Defaults are TIRS.
    """
    instrument = {
        "name": name,
        "fl": fl,
        "pitch": pitch,
        "h_pix": h_pix,
        "v_pix": v_pix,
        "mm": mm,
        "half_diag_deg": np.degrees(
            (pitch / fl) * np.sqrt(h_pix ** 2 + v_pix ** 2) / 2
        ),
        "az1": np.degrees(np.arctan2(h_pix, v_pix)),
        "az2": 360 - np.degrees(np.arctan2(h_pix, v_pix)),
    }
    return instrument


def los_intercept(lat0, lon0, h0, az, tilt, ell=None, deg=True):
    """
    Generates LOS intercepts based on LLA and individual az, tilt angles from velocity angles and nadir, respectively.
    Follows this article: https://medium.com/@stephenhartzell/satellite-line-of-sight-intersection-with-earth-d786b4a6a9b6
    Would be better off doing in ECEF coordinates directly using skyfield? Just need to compute unit vector and convert to LLA after the fact.
    TODO:
    - Remove dependency
    - Skyfield branch?
    """
    lat, lon, d = lookAtSpheroid(lat0, lon0, h0, az, tilt, ell=None, deg=True)

    return lat, lon, d


def gen_los_offsets(df_in, inst):
    """
    Calculates instrument "frustrum" in angle space, which is applied in psuedo LVLH frame? (Azimuth / tilt offsets from satellite bearing)
    Returns returns a df that should be unmodified, and a geodf that has polygons of FOV intercepts.
    Currently works at yaw=0/ nadir pointing only. These would be added as offsets to all frustrum corners.
    TODO:
    - Clean up corner point calculation (use zip or something)
    - Fix the Polygon issue at 180th meridian, currently handled by dropping area outliers
    - Add off-nadir/ yaw pointing (azimuth) angular offsets
    """
    dfs = []
    gdfs = []
    # c_cols = [
    #     "c1_lat",
    #     "c1_lon",
    #     "c2_lat",
    #     "c2_lon",
    #     "c3_lat",
    #     "c3_lon",
    #     "c4_lat",
    #     "c4_lon",
    # ]

    for sat in df_in.satellite.unique():
        gdf = gpd.GeoDataFrame(crs="epsg:4326", columns=["geometry"])
        df = df_in[df_in.satellite == sat].copy()
        # df.loc[:, c_cols] = [np.nan] * len(c_cols)

        for n in df.index:
            lat0 = df.loc[n]["lat"]
            lon0 = df.loc[n]["lon"]
            h0 = df.loc[n]["altitude"] * 1000
            az0 = df.loc[n]["bearing"]

            az = az0 + inst["az1"]
            if az < 0:
                az += 360
            tilt = inst["half_diag_deg"]
            c1_lat, c1_lon, _ = los_intercept(lat0, lon0, h0, az, tilt)

            az = az0 + inst["az2"]
            if az < 0:
                az += 360
            tilt = inst["half_diag_deg"]
            c2_lat, c2_lon, _ = los_intercept(lat0, lon0, h0, az, tilt)

            az = az0 + inst["az1"]
            if az < 0:
                az += 360
            tilt = -inst["half_diag_deg"]
            c3_lat, c3_lon, _ = los_intercept(lat0, lon0, h0, az, tilt)

            az = az0 + inst["az2"]
            if az < 0:
                az += 360
            tilt = -inst["half_diag_deg"]
            c4_lat, c4_lon, _ = los_intercept(lat0, lon0, h0, az, tilt)

            # df.loc[n, c_cols] = [
            #     c1_lat,
            #     c1_lon,
            #     c2_lat,
            #     c2_lon,
            #     c3_lat,
            #     c3_lon,
            #     c4_lat,
            #     c4_lon,
            # ]

            gdf.loc[n, "geometry"] = Polygon(
                [(c1_lon, c1_lat), (c2_lon, c2_lat), (c3_lon, c3_lat), (c4_lon, c4_lat)]
            )

        gdf["satellite"] = sat
        gdfs.append(gdf)
        dfs.append(df)
    gdf_out = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs="epsg:4326")
    df_out = pd.concat(dfs)
    return df_out, gdf_out


# if __name__ == "__main__":
