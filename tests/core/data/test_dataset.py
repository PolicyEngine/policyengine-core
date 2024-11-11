from pathlib import Path
from tempfile import NamedTemporaryFile


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


def test_atomic_write():
    from policyengine_core.data.dataset import atomic_write

    with NamedTemporaryFile(mode="w") as file:
        file.write("Hello, world\n")
        file.flush()
        # Open the file before overwriting
        with open(file.name, "r") as file_original:

            atomic_write(Path(file.name), "NOPE\n".encode())

            # Open file descriptor still points to the old node
            assert file_original.readline() == "Hello, world\n"
            # But if I open it again it has the new content
            with open(file.name, "r") as file_updated:
                assert file_updated.readline() == "NOPE\n"
