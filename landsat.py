from skyfield.api import load, EarthSatellite
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
import pprint

pp = pprint.PrettyPrinter(indent=4)
# from coverage import *

ts = load.timescale()
now_utc = datetime.now(timezone.utc)
tom_utc = now_utc + timedelta(days=1, hours=0.5)


@dataclass
class Scene:
    scene_ref: str = "WRS-2"
    aoi_dir: str = "./aois/"
    out_dir: str = "./tmp/"

    start_utc: str = str(now_utc)
    end_utc: str = str(tom_utc)

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
    # f_no: float = 1.2
    focal_length_mm: float = 178
    # transmission: float = 0.8

    # Detector properties
    pitch_um: float = 25.0
    cols: int = 1850
    rows: int = 1800
    img_period: float = 22 / 60


@dataclass
class Platform:
    name: str = "LANDSAT 9"

    # Satellite NORAD identifier for TLE lookup
    norad_id: int = 49260
    # satellite: list = single_sat(norad_id)[0]

    # Simulation clock (how often pos, attitude, thermal are updated)
    # sim_clock_hz: float = 100.0

    # Position info
    # coord_frame: str = "ITRS"
    # x_pos_km: float = 0.0
    # y_pos_km: float = 0.0
    # z_pos_km: float = 0.0


if __name__ == "__main__":
    landsat9 = Instrument()
    pp.pprint(asdict(landsat9))
