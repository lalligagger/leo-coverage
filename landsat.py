from skyfield.api import load, EarthSatellite
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
# import shapely
from coverage import * 

ts = load.timescale()
now_utc = datetime.now(timezone.utc)
tom_utc = now_utc + timedelta(days=1, hours=2)

now_ts = ts.from_datetime(now_utc)
tom_ts = ts.from_datetime(tom_utc)

@dataclass
class Scene:
    scene_ref: str = "WRS-2"
    aoi_dir: str = "../aois/hytran_test/"
    out_dir: str = "../output/hytran_test/"

    start_utc: str = str(now_ts.utc_strftime())
    end_utc: str = str(tom_ts.utc_strftime())

    # Simulation parameters (are these always used?)
    wav_min_um: float = 10.0
    wav_max_um: float = 13.0
    wav_step_um: float = 0.1

@dataclass
class Instrument:
    name: str = "TIRS"

    # System primary wavelength
    wav: float = 12.0

    # First-order optical properties
    f_no: float = 1.2
    focal_length_mm: float = 178
    transmission: float = 0.8

    # Detector properties
    pitch_um: float = 25.0
    cols: int = 1850
    rows: int = 1800
    framerate: float = 1/22

@dataclass
class Platform:
    name: str = "Landsat 9"

    # Simulation clock (how often pos, attitude, thermal are updated)
    sim_clock_hz: float = 100.0

    # Satellite NORAD identifier for TLE lookup
    norad_id: int = 49260
    # satellite: list = single_sat(norad_id)[0]

    # Position info
    coord_frame: str = "ITRS"
    x_pos_km: float = 0.0
    y_pos_km: float = 0.0
    z_pos_km: float = 0.0