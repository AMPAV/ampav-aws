"""Package version helpers."""

from importlib.metadata import PackageNotFoundError, version


DISTRIBUTION_NAME = "ampav-aws"


def _package_version(distribution_name: str, *, fallback: str = "0+unknown") -> str:
    try:
        return version(distribution_name)
    except PackageNotFoundError:
        return fallback


__version__ = _package_version(DISTRIBUTION_NAME)
