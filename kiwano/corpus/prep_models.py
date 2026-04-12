"""
Data models for the corpus preparation phase.

Preparation follows the same principle as downloading: each corpus is
described by a PrepareSpec, the engine (PreparePipeline) executes it.

File processor contract (per-corpus callable):
    def process_file(
        segment: Path,
        out_data: Path,
        sampling_frequency: int,
        vad: bool,
    ) -> Tuple[str, str, str, Path]:
        '''Return (name, spkid, duration, output_path).'''

For corpora with multiple output lists (e.g. CN-Celeb train/dev/test),
use `multi_processor` which returns a dict {list_name: row_tuple}.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# Standard file_processor signature
FileProcessorFn = Callable[
    [Path, Path, int, bool],       # segment, out_data, sampling_freq, vad
    Tuple[str, str, str, Path],    # name, spkid, duration, output_path
]


@dataclass
class TrialsSpec:
    """
    Describes conversion of a source trials file to the target format.

    The converter_fn receives (in_data, out_data) and writes the output files.
    """

    converter_fn: Callable[[Path, Path], None]


@dataclass
class PrepareSpec:
    """
    Complete description of how to prepare a corpus.

    Parameters
    ----------
    file_processor :
        Corpus-specific callable — builds name/spkid, extracts duration,
        runs FFmpeg/VAD if needed. Signature: FileProcessorFn.
    file_extensions :
        Extensions to search for (e.g. [".wav"], [".m4a"], [".flac"]).
    search_root :
        Subdirectory relative to in_data where files are searched
        (e.g. "wav", "dev/aac"). None = search in_data directly.
    match_or :
        Inclusive path filters (e.g. ["mediumroom", "smallroom"]).
        A file is kept if its path contains AT LEAST one of these tokens.
    output_lists :
        Names of the output list files. One element for most corpora;
        CN-Celeb has three: ["listeTrain", "listeDev", "listeTest"].
    multi_processor :
        Variant of file_processor for multi-list corpora.
        Signature: (segment, out_data, sf, vad) -> Dict[list_name, row_tuple].
        Mutually exclusive with file_processor.
    trials :
        Trial file conversion spec (optional).
    cleanup_fn :
        Called after preparation to remove archives.
        Signature: (in_data: Path, delete: bool) -> None.
    """

    file_processor: Optional[FileProcessorFn] = None
    multi_processor: Optional[Callable] = None
    file_extensions: List[str] = field(default_factory=lambda: [".wav"])
    search_root: Optional[str] = None
    match_or: List[str] = field(default_factory=list)
    output_lists: List[str] = field(default_factory=lambda: ["liste"])
    trials: Optional[TrialsSpec] = None
    cleanup_fn: Optional[Callable[[Path, bool], None]] = None

    def __post_init__(self):
        if self.file_processor is None and self.multi_processor is None:
            raise ValueError(
                    "PrepareSpec requires either file_processor or multi_processor."
                    )
        if self.file_processor is not None and self.multi_processor is not None:
            raise ValueError(
                    "file_processor and multi_processor are mutually exclusive."
                    )
