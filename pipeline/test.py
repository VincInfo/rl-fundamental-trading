from pathlib import Path

import data_pipeline
from data_pipeline.cleanup import clean_data
from data_pipeline.config import load_config
from data_pipeline.data import prepare_data
from data_pipeline.fundamentals import enrich_data_with_fundamentals

package_dir = Path(data_pipeline.__file__).resolve().parent
resource_path = package_dir / "resources"

config = load_config(resource_path)

raw_data = prepare_data(
    config["data"],
    resource_path,
    force_redownload=False,
)

cleaned_data = clean_data(raw_data)

fundamental_data = enrich_data_with_fundamentals(
    data=cleaned_data,
    tickers=config["data"]["tickers"],
    config=config["fundamentals"],
    resource_path=resource_path,
)

print(fundamental_data.shape)
print(fundamental_data.columns)