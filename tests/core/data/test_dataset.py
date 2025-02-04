from pathlib import Path
from tempfile import NamedTemporaryFile
import sys
import threading
from policyengine_core.tools.win_file_manager import WindowsAtomicFileManager
import tempfile
from pathlib import Path
import uuid


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
    if sys.platform != "win32":
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


def test_atomic_write_windows():
    if sys.platform == "win32":
        temp_dir = Path(tempfile.gettempdir())
        temp_files = [
            temp_dir / f"tempfile_{uuid.uuid4().hex}.tmp" for _ in range(5)
        ]

        managers = [WindowsAtomicFileManager(path) for path in temp_files]

        contents_list = [
            [f"Content_{i}_{j}".encode() for j in range(5)] for i in range(5)
        ]

        check_results = [[] for _ in range(5)]

        threads = []
        for i in range(5):
            thread = threading.Thread(
                target=file_task,
                args=(managers[i], contents_list[i], check_results[i]),
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        for i, results in enumerate(check_results):
            for expected, actual in results:
                assert (
                    expected == actual
                ), f"Mismatch in file {i}: expected {expected}, got {actual}"

        for temp_file in temp_files:
            if temp_file.exists():
                temp_file.unlink()


def file_task(manager, contents, check_results):
    for content in contents:
        manager.write(content)
        actual_content = manager.read().decode()
        expected_content = content.decode()
        check_results.append((expected_content, actual_content))
