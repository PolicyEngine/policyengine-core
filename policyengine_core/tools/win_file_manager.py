import os
import tempfile
from pathlib import Path
from threading import Lock


class WindowsAtomicFileManager:
    """
    https://stackoverflow.com/a/2368286
    - Each instance manages a specific logical file by name.
    - Files are written atomically using temporary files.
    - Thread-safe operations are ensured using a `Lock`.
    - Temporary files are created in the system's temporary directory.
    - Replace the target file with the temporary file
    - For any fallback, Cleanup temporary file if replacement fails
    """

    def __init__(self, file: Path):
        self.logical_name = file.name
        self.target_path = file
        self.lock = Lock()

    def write(self, content: bytes):
        with self.lock:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self.target_path.parent.absolute().as_posix(),
                delete=False,
            ) as temp_file:
                temp_file.write(content)
                temp_file.flush()
                os.fsync(temp_file.fileno())
                temp_file_path = Path(temp_file.name)

            try:
                os.replace(temp_file_path, self.target_path)
            except Exception as e:
                temp_file_path.unlink(missing_ok=True)
                raise e

    def read(self) -> str:
        with self.lock:
            if not self.target_path.exists():
                raise FileNotFoundError(f"{self.target_path} does not exist.")
            with open(self.target_path, "rb") as file:
                return file.read()
