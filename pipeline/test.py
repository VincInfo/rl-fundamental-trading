from data_pipeline import DataSplit, DataVariant, get_data

data = get_data(DataVariant.WITH_FUNDAMENTALS)
train_data = data[DataSplit.TRAIN]
validation_data = data[DataSplit.VALIDATION]
test_data = data[DataSplit.TEST]

print(f"train: {train_data.shape}")
print(f"validation: {validation_data.shape}")
print(f"test: {test_data.shape}")