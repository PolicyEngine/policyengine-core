from policyengine_core.__version__ import __version__


def fetch_version():
    try:
        return __version__
    except Exception as e:
        print(f"Error fetching version: {e}")
        return None


if __name__ == "__main__":
    print(fetch_version())