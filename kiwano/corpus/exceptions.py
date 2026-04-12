"""
Specialized exceptions for clear error-handling policies.
"""


class DatasetDownloaderError(Exception):
    """Base class for all dataset downloader errors."""


class DownloadError(DatasetDownloaderError):
    """Network, HTTP, or timeout failure during download."""


class ChecksumError(DatasetDownloaderError):
    """Computed checksum does not match the expected value."""


class AssemblyError(DatasetDownloaderError):
    """Failure during multi-part archive reconstruction."""


class ExtractionError(DatasetDownloaderError):
    """Failure during archive extraction."""


class DatasetSpecError(DatasetDownloaderError):
    """Dataset specification is invalid or incomplete."""
