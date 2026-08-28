from data_pipeline import DataSplit, DataVariant, get_data


def test_shared_pipeline_returns_fundamental_splits():
    data = get_data(DataVariant.WITH_FUNDAMENTALS)

    assert set(data) == set(DataSplit)
    for split in DataSplit:
        frame = data[split]
        assert not frame.empty
        assert len(frame.columns) == 320
