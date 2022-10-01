import wget
from pathlib import Path
from tqdm import tqdm
import argparse
import pickle
import cdflib

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, help="Choose wi mag or swe to download")
args = parser.parse_args()
download = False
convert = True

if args.data == "wi":
    base_url = "https://cdaweb.gsfc.nasa.gov/pub/data/wind/mfi/mfi_h2/2022/"
    base_name = "wi_h2_mfi_2022"
    version = "_v04"

elif args.data == "mag":
    base_url = "https://cdaweb.gsfc.nasa.gov/pub/data/dscovr/h0/mag/2022/"
    base_name = "dscovr_h0_mag_2022"
    version = "_v01"

elif args.data == "swe":
    base_url = "https://cdaweb.gsfc.nasa.gov/pub/data/wind/swe/swe_h1/2022/"
    base_name = "wi_h1_swe_2022"
    version = "_v01"

for i in tqdm(range(1, 10)):
    if i < 10:
        stri = "0" + str(i)
    else:
        stri = str(i)
    for j in range(1, 32):
        if j < 10:
            strj = "0" + str(j)
        else:
            strj = str(j)
        filename = Path(base_name + stri + strj + version)
        if download:
            final_url = base_url + base_name + stri + strj + version
            if not filename.exists():
                try:
                    filename = wget.download(final_url + ".cdf")
                except:
                    -1
        if convert:
            if Path(str(filename) + ".cdf").exists():
                with open(Path(str(filename) + ".pickle"), "wb") as handle:
                    pickle.dump(
                        cdflib.cdf_to_xarray(str(filename) + ".cdf"),
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
