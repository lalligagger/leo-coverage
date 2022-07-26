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

@dataclass
class Instrument:
    name: str = "TIRS"
    focal_length_mm: float = 178
    pitch_um: float = 25.0
    cols: int = 1850
    rows: int = 1800
    img_period: float = 22 / 60

@dataclass
class Platform:
    name: str = "LANDSAT 9"
    norad_id: int = 49260

if __name__ == "__main__":
    landsat9 = Instrument()
    pp.pprint(asdict(landsat9))
