import wget
from pathlib import Path
import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("data", type="str", help="Choose wi mag or swe")
args = parser.parse_args()

if args.data == "wi":
    base_url = "https://cdaweb.gsfc.nasa.gov/pub/data/wind/mfi/mfi_h2/2022/"
    base_name = "wi_h2_mfi_2022"
    version = "_v04.cdf"

elif args.data == "mag":
    base_url = "https://cdaweb.gsfc.nasa.gov/pub/data/dscovr/h0/mag/2022/"
    base_name = "dscovr_h0_mag_2022"
    version = "_v01.cdf"

elif args.data == "swe":
    base_url = "https://cdaweb.gsfc.nasa.gov/pub/data/wind/swe/swe_h1/2022/"
    base_name = "wi_h1_swe_2022"
    version = "_v01.cdf"

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
        final_url = base_url + base_name + stri + strj + version
        if filename.exists():
            try:
                filename = wget.download(final_url)
            except:
                -1
