from pathlib import Path

import data_pipeline
import pandas as pd
from data_pipeline.cleanup import clean_data
from data_pipeline.config import load_config
from data_pipeline.data import prepare_data
from engineer_fundamentals import engineer_fundamentals

package_dir = Path(data_pipeline.__file__).resolve().parent
resource_path = package_dir / "resources"

config = load_config(resource_path)

raw_data = prepare_data(
    config["data"],
    resource_path,
    force_redownload=False,
)

cleaned_data = clean_data(raw_data)
raw_fundamentals = pd.read_pickle(resource_path / "raw_fundamentals.pkl")

fundamental_data = engineer_fundamentals(
    raw=raw_fundamentals,
    market_data=cleaned_data,
)

print(fundamental_data.shape)
print(fundamental_data.columns)