# %%

import io
import logging

# import pickle
import zipfile

# import numpy as np
import pandas as pd
import requests

# from pathlib import Path
# from zipfile import ZipFile


# from scipy import signal
# from tqdm import tqdm

# from physped.core.piecewise_potential import PiecewisePotential

log = logging.getLogger(__name__)

# %%

# link = "https://data.4tu.nl/ndownloader/items/b8e30f8c-3931-4604-842a-77c7fb8ac3fc/versions/1"
# bytestring = requests.get(link, timeout=10)
# with zipfile.ZipFile(io.BytesIO(bytestring.content), "r") as outerzip:
#     with zipfile.ZipFile(outerzip.open("data.zip")) as innerzip:
#         with innerzip.open("left-to-right.ssv") as paths_ltr:
#             paths_ltr = paths_ltr.read().decode("utf-8")
#         with innerzip.open("right-to-left.ssv") as paths_rtl:
#             paths_rtl = paths_rtl.read().decode("utf-8")


# %%

print("Hello world!")

# %%

link = "https://data.4tu.nl/file/7d78a5e3-6142-49fe-be03-e4c707322863/40ea5cd9-95dc-4e3c-8760-7f4dd543eae7"
bytestring = requests.get(link, timeout=10)

with zipfile.ZipFile(io.BytesIO(bytestring.content), "r") as zipped_file:
    # with zipfile.ZipFile(outerzip.open("data.zip")) as innerzip:
    with zipped_file.open("Amsterdam Zuid - platform 3-4 - set1.csv") as paths:
        paths = paths.read().decode("utf-8")

# %%

df = pd.read_csv(io.StringIO(paths), sep=",")

# %%

df.head()

# %%
