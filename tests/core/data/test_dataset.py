from pathlib import Path


def test_dataset_class():
    from policyengine_core.data.dataset import Dataset
    from policyengine_core.periods import period

    tmp_folder_path = Path(__file__).parent / "tmp"
    tmp_folder_path.mkdir(exist_ok=True)

    class TestDataset(Dataset):
        name = "test_dataset"
        label = "Test dataset"
        file_path = tmp_folder_path / "test_dataset.h5"
        data_format = Dataset.TIME_PERIOD_ARRAYS

        def generate(self) -> None:
            input_period = period("2022-01")
            self.save_dataset({"salary": {input_period: [100, 200, 300]}})

    test_dataset = TestDataset()
    test_dataset.remove()
    test_dataset.generate()
    assert test_dataset.exists
    test_dataset.remove()
    assert not test_dataset.exists
