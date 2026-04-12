"""
Declarative data models for describing a corpus.

Principle: each corpus is a *configuration*, not a script.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .prep_models import PrepareSpec

from .exceptions import DatasetSpecError

# ---------------------------------------------------------------------------
# Primitive types
# ---------------------------------------------------------------------------
HashAlgo = Literal["md5", "sha256"]
ResourceKind = Literal["part", "metadata", "file"]
AssembleMethod = Literal["none", "concat_parts", "copy_glob", "concat_clean"]
ExtractMethod = Literal["none", "zip", "tar", "tgz"]
DownloaderBackend = Literal["http", "github", "youtube"]

@dataclass
class ChecksumSpec:
    algo: HashAlgo
    value: str


@dataclass
class ResourceSpec:
    """Describes a single downloadable resource."""

    url: str
    checksum: Optional[ChecksumSpec] = None
    kind: ResourceKind = "file"
    optional: bool = False
    backend: DownloaderBackend = "http"
    # Clé dans le JSON de manifeste pour les corpus à structure dynamique (ex. ArabCeleb)
    #manifest_key: Optional[str] = None

    @property
    def filename(self) -> str:
        return self.url.split("/")[-1]


@dataclass
class AssembleSpec:
    """Describes how to reconstruct an archive from split parts."""

    method: AssembleMethod = "none"
    output_name: Optional[str] = None   # e.g. "vox2_aac.zip"
    pattern: Optional[str] = None       # e.g. "vox2_dev_aac_part*"
    remove_parts: bool = False


@dataclass
class ExtractSpec:
    """Describes how to extract an archive."""

    method: ExtractMethod = "none"
    archive_name: Optional[str] = None  # None = inferred from the first resource
    remove_after: bool = False


@dataclass
class ResourceGroup:
    """
    Coherent group of resources sharing the same assemble/extract cycle.

    CN-Celeb is the typical case: two independent groups within one corpus,
    each with its own assembly and extraction.
    """

    resources: List[ResourceSpec] = field(default_factory=list)
    assemble: AssembleSpec = field(default_factory=AssembleSpec)
    extract: ExtractSpec = field(default_factory=ExtractSpec)


@dataclass
class DatasetSpec:
    """
    Complete description of a corpus.

    Adding a corpus = instantiating this dataclass.
    The engine needs nothing else.

    Two modes:
      - Simple (most corpora): set `resources`, `assemble`, `extract` directly.
        The engine creates an implicit ResourceGroup.
      - Multi-group (CN-Celeb, etc.): set `groups`. The `resources`, `assemble`,
        `extract` fields are then ignored.
    """

    name: str
    description: str

    # Simple mode
    resources: List[ResourceSpec] = field(default_factory=list)
    assemble: AssembleSpec = field(default_factory=AssembleSpec)
    extract: ExtractSpec = field(default_factory=ExtractSpec)

    # Multi-group mode (optional)
    groups: List[ResourceGroup] = field(default_factory=list)

    max_retries: int = 3

    # Optional hook for truly corpus-specific post-processing
    postprocess_fn: Optional[Callable] = None
    postprocess_kwargs: Dict[str, Any] = field(default_factory=dict)
    prepare_spec: Optional[PrepareSpec] = None


    @property
    def can_download(self) -> bool:
        return bool(self.resources or self.groups)

    @property
    def can_prepare(self) -> bool:
        return self.prepare_spec is not None

    def validate_for_download(self) -> None:
        if not self.resources and not self.groups:
            raise DatasetSpecError(
                f"[{self.name}] Cannot download: no resources defined."
            )

    def validate_for_prepare(self) -> None:
        if self.prepare_spec is None:
            raise DatasetSpecError(
                f"[{self.name}] Cannot prepare: no prepare_spec defined."
            )

    def resolved_groups(self) -> List[ResourceGroup]:
        """Return a list of ResourceGroup regardless of mode."""
        if self.groups:
            return self.groups
        return [ResourceGroup(
            resources=self.resources,
            assemble=self.assemble,
            extract=self.extract,
        )]
