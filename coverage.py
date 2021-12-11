import numpy as np
from skyfield.api import load, wgs84, Distance
from skyfield.toposlib import ITRSPosition
from skyfield.framelib import itrs
from scipy.spatial.transform import Rotation


def gen_sats(sat_nos=[39084, 49260]):
    """
    Skyfield satellite lookup from Celestrack, based on catalog ID.
    Landsat 8 & 9 defaults.
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


def gen_instrument(
    name="instrument", fl=178, pitch=0.025, h_pix=1850, v_pix=1800, mm=True
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
        "mm": mm,
        "hfov_deg": hfov_deg,  # update to atan
        "vfov_deg": vfov_deg,  # update to atan
        "half_diag_deg": np.degrees(
            (pitch / fl) * np.sqrt(h_pix ** 2 + v_pix ** 2) / 2
        ),
        "az1": np.degrees(np.arctan2(h_pix, v_pix)),
        "az2": 360 - np.degrees(np.arctan2(h_pix, v_pix)),
        "corners" : {"c1": {"X" : -hfov_deg/2, "Y": vfov_deg/2}, 
                "c2": {"X" : hfov_deg/2, "Y": vfov_deg/2},
                "c3": {"X" : hfov_deg/2, "Y": -vfov_deg/2},
                "c4": {"X" : -hfov_deg/2, "Y": -vfov_deg/2}
            }
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
    x = position[0]
    y = position[1]
    z = position[2]
    u = pointing[0]
    v = pointing[1]
    w = pointing[2]

    value = (
        -(a ** 2) * b ** 2 * w * z - a ** 2 * c ** 2 * v * y - b ** 2 * c ** 2 * u * x
    )
    radical = (
        a ** 2 * b ** 2 * w ** 2
        + a ** 2 * c ** 2 * v ** 2
        - a ** 2 * v ** 2 * z ** 2
        + 2 * a ** 2 * v * w * y * z
        - a ** 2 * w ** 2 * y ** 2
        + b ** 2 * c ** 2 * u ** 2
        - b ** 2 * u ** 2 * z ** 2
        + 2 * b ** 2 * u * w * x * z
        - b ** 2 * w ** 2 * x ** 2
        - c ** 2 * u ** 2 * y ** 2
        + 2 * c ** 2 * u * v * x * y
        - c ** 2 * v ** 2 * x ** 2
    )
    magnitude = (
        a ** 2 * b ** 2 * w ** 2 + a ** 2 * c ** 2 * v ** 2 + b ** 2 * c ** 2 * u ** 2
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
    geo = sat.at(time)
    xyz_dist_rates = geo.frame_xyz_and_velocity(itrs)
    xyz_dist = xyz_dist_rates[0]
    pointing = -xyz_dist.km / xyz_dist.length().km

    xyz_vel = xyz_dist_rates[1]
    # bearing = xyz_vel.km_per_s / np.linalg.norm(xyz_vel.km_per_s)

    los_xyz = los_to_earth(xyz_dist.km, pointing)  # input is meters

    los = Distance(km=los_xyz)
    los_itrs = ITRSPosition(los)
    los_itrs.at(time).frame_xyz(itrs).km

    los_lat, los_lon = wgs84.latlon_of(los_itrs.at(time))
    d = np.sqrt(np.sum(np.square(xyz_dist.km - los_xyz)))

    return los_lat, los_lon, d


def get_lvlh_pointing(sat, time):
    geo = sat.at(time)
    xyz_dist_rates = geo.frame_xyz_and_velocity(itrs)
    xyz_dist = xyz_dist_rates[0]
    pointing = -xyz_dist.km / xyz_dist.length().km

    xyz_vel = xyz_dist_rates[1]
    bearing = xyz_vel.km_per_s / np.linalg.norm(xyz_vel.km_per_s)

    neg_orb_normal = -np.cross(pointing, bearing)
    x_axis = np.cross(neg_orb_normal, pointing)
    lvlh = {"X": x_axis, "Y": neg_orb_normal, "Z": pointing}

    # cross = np.cross(pointing, bearing)
    # cross = cross / np.linalg.norm(cross)
    # lvlh = {"X": bearing, "Y": cross, "Z": pointing}

    los_xyz = los_to_earth(xyz_dist.km, pointing)
    los = Distance(km=los_xyz)
    los_itrs = ITRSPosition(los)
    los_itrs.at(time).frame_xyz(itrs).km

    return lvlh, pointing

def get_inst_fov(sat, time, inst):

    lvlh, pointing = get_lvlh_pointing(sat, time)
    xyz_dist_rates = sat.at(time).frame_xyz_and_velocity(itrs)
    xyz_dist = xyz_dist_rates[0]
    z_rate = xyz_dist_rates[1]
    # direction.append(z_rate.km_per_s[2])

    # Empty dict to populate with lat/ lons, mapped from cs_dict in angle space
    cs_lla_dict = {"c1": {"lat" : None, "lon": None}, 
        "c2": {"lat" : None, "lon": None},
        "c3": {"lat" : None, "lon": None},
        "c4": {"lat" : None, "lon": None}
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
        # Rotate about X
        rot_X_vec = rot_X_rad * rot_X_ax
        rot_X = Rotation.from_rotvec(rot_X_vec)
        los_X = rot_X.apply(pointing)

        # Rotate about Y for final LOS
        rot_Y_vec = rot_Y_rad * rot_Y_ax
        rot_Y = Rotation.from_rotvec(rot_Y_vec)
        los_XY = rot_Y.apply(los_X)

        # Get Earth intercept of LOS, create ITRS position object
        los_xyz = los_to_earth(xyz_dist.km, los_XY)
        los = Distance(km=los_xyz)
        los_itrs = ITRSPosition(los)
        los_itrs.at(time).frame_xyz(itrs).km

        # Calculate intercept lat/ lon from ITRS frame
        los_lat, los_lon = wgs84.latlon_of(los_itrs.at(time))
        cs_lla_dict[c]["lat"] = los_lat.degrees
        cs_lla_dict[c]["lon"] = los_lon.degrees

    return cs_lla_dict


if __name__ == "__main__":
    tles = gen_sats(sat_nos=[48915])
    sat = tles[0][0]
    times = gen_times(start_yr=2021, start_mo=11, start_day=27, days=1, step_min=1)
    los_lat, los_lon, d = get_los(sat, times[0])
    geo = sat.at(times[0])
    lat, lon = wgs84.latlon_of(geo)
    print(lat)
    print(los_lat)

    print(lon)
    print(los_lon)

    print(d)
