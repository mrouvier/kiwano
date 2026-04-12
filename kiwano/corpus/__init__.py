"""
Dataset registry.

Adding a corpus = instantiate a DatasetSpec here and register it in REGISTRY.
The engine needs nothing else.
"""

from pathlib import Path

from .corpus_processors import (
    prepare_butreverb,
    prepare_cn_celeb,
    prepare_esc50,
    prepare_musan,
    prepare_rirs_noises,
    prepare_voxceleb1,
    prepare_voxceleb2,
)
from .models import (
    AssembleSpec,
    ChecksumSpec,
    DatasetSpec,
    ExtractSpec,
    ResourceGroup,
    ResourceSpec,
)

# ── Post-processing hooks ────────────────────────────────────────


def _extract_all_tarballs(target_dir: Path, **kwargs) -> None:
    """Extract each Hi-MIA tar.gz archive independently."""
    from kiwano.utils import extract_tar

    for resource in hi_mia.resources:
        archive = target_dir / resource.filename
        if archive.exists():
            extract_tar(archive, target_dir)


def _download_arab_celeb_videos(target_dir: Path, num_jobs: int = 1, **kwargs) -> None:
    """Download ArabCeleb YouTube videos from the JSON manifest."""
    from ..downloaders.youtube import download_from_manifest

    manifest = target_dir / "utterance_info.json"
    download_from_manifest(manifest, target_dir, num_jobs=num_jobs)


# ── VoxCeleb 1 ────────────────────────────────────────────────────
voxceleb1 = DatasetSpec(
    name="voxceleb1",
    description="Large-scale celebrity speech corpus from YouTube for speaker verification for speaker training and evaluation.",
    resources=[
        ResourceSpec(
            "http://kiwano-toolkit.com/files/voxceleb1/vox1a/vox1_dev_wav_partaa",
            ChecksumSpec("md5", "e395d020928bc15670b570a21695ed96"),
            kind="part",
        ),
        ResourceSpec(
            "http://kiwano-toolkit.com/files/voxceleb1/vox1a/vox1_dev_wav_partab",
            ChecksumSpec("md5", "bbfaaccefab65d82b21903e81a8a8020"),
            kind="part",
        ),
        ResourceSpec(
            "http://kiwano-toolkit.com/files/voxceleb1/vox1a/vox1_dev_wav_partac",
            ChecksumSpec("md5", "017d579a2a96a077f40042ec33e51512"),
            kind="part",
        ),
        ResourceSpec(
            "http://kiwano-toolkit.com/files/voxceleb1/vox1a/vox1_dev_wav_partad",
            ChecksumSpec("md5", "7bb1e9f70fddc7a678fa998ea8b3ba19"),
            kind="part",
        ),
        ResourceSpec(
            "http://kiwano-toolkit.com/files/voxceleb1/vox1a/vox1_test_wav.zip",
            ChecksumSpec("md5", "185fdc63c3c739954633d50379a3d102"),
            kind="file",
        ),
        ResourceSpec(
            "http://kiwano-toolkit.com/files/voxceleb1/meta/veri_test2.txt",
            ChecksumSpec("md5", "b73110731c9223c1461fe49cb48dddfc"),
            kind="metadata",
        ),
        ResourceSpec(
            "http://kiwano-toolkit.com/files/voxceleb1/meta/list_test_hard2.txt",
            ChecksumSpec("md5", "857790e09d579a68eb2e339a090343c8"),
            kind="metadata",
        ),
        ResourceSpec(
            "http://kiwano-toolkit.com/files/voxceleb1/meta/list_test_all2.txt",
            ChecksumSpec("md5", "a53e059deb562ffcfc092bf5d90d9f3a"),
            kind="metadata",
        ),
        ResourceSpec(
            "https://www.openslr.org/resources/49/vox1_meta.csv",
            ChecksumSpec("md5", "f5067cae5157ca27904c4b0a926e2661"),
            kind="metadata",
        ),
    ],
    assemble=AssembleSpec(
        method="copy_glob", output_name="vox1_dev_wav.zip", pattern="vox1_dev_wav_part*"
    ),
    extract=ExtractSpec(method="zip", archive_name="vox1_dev_wav.zip"),
    max_retries=3,
    prepare_spec=prepare_voxceleb1,
)


# ── VoxCeleb 2 ────────────────────────────────────────────────────


voxceleb2 = DatasetSpec(
    name="voxceleb2",
    description="Expanded celebrity speech corpus from YouTube with more speakers and utterances.",
    resources=[
        ResourceSpec(
            "http://kiwano-toolkit.com/files/voxceleb2/vox2_dev_aac_partaa",
            ChecksumSpec("md5", "da070494c573e5c0564b1d11c3b20577"),
            kind="part",
        ),
        ResourceSpec(
            "http://kiwano-toolkit.com/files/voxceleb2/vox2_dev_aac_partab",
            ChecksumSpec("md5", "17fe6dab2b32b48abaf1676429cdd06f"),
            kind="part",
        ),
        ResourceSpec(
            "http://kiwano-toolkit.com/files/voxceleb2/vox2_dev_aac_partac",
            ChecksumSpec("md5", "1de58e086c5edf63625af1cb6d831528"),
            kind="part",
        ),
        ResourceSpec(
            "http://kiwano-toolkit.com/files/voxceleb2/vox2_dev_aac_partad",
            ChecksumSpec("md5", "5a043eb03e15c5a918ee6a52aad477f9"),
            kind="part",
        ),
        ResourceSpec(
            "http://kiwano-toolkit.com/files/voxceleb2/vox2_dev_aac_partae",
            ChecksumSpec("md5", "cea401b624983e2d0b2a87fb5d59aa60"),
            kind="part",
        ),
        ResourceSpec(
            "http://kiwano-toolkit.com/files/voxceleb2/vox2_dev_aac_partaf",
            ChecksumSpec("md5", "fc886d9ba90ab88e7880ee98effd6ae9"),
            kind="part",
        ),
        ResourceSpec(
            "http://kiwano-toolkit.com/files/voxceleb2/vox2_dev_aac_partag",
            ChecksumSpec("md5", "d160ecc3f6ee3eed54d55349531cb42e"),
            kind="part",
        ),
        ResourceSpec(
            "http://kiwano-toolkit.com/files/voxceleb2/vox2_dev_aac_partah",
            ChecksumSpec("md5", "6b84a81b9af72a9d9eecbb3b1f602e65"),
            kind="part",
        ),
        ResourceSpec(
            "https://www.openslr.org/resources/49/vox2_meta.csv",
            ChecksumSpec("md5", "6090d767f8334733dfe4c6578fa725c2"),
            kind="metadata",
        ),
    ],
    assemble=AssembleSpec(
        method="copy_glob", output_name="vox2_aac.zip", pattern="vox2_dev_aac_part*"
    ),
    extract=ExtractSpec(method="zip", archive_name="vox2_aac.zip"),
    max_retries=3,
    prepare_spec=prepare_voxceleb2,
)

# ── BUT ReverbDB ──────────────────────────────────────────────────

but_reverb = DatasetSpec(
    name="but_reverb",
    description="Real room impulse responses for reverberation and far-field speech experiments for speech data augmentation.",
    resources=[
        ResourceSpec(
            "http://merlin.fit.vutbr.cz/ReverbDB/BUT_ReverbDB_rel_19_06_RIR-Only.tgz",
            ChecksumSpec("md5", "3ad91777f6cadec81fb4c07dfbef6f19"),
        ),
    ],
    assemble=AssembleSpec(method="none"),
    extract=ExtractSpec(method="tgz"),
    max_retries=3,
    prepare_spec=prepare_butreverb,
)

# ── Hi-MIA ────────────────────────────────────────────────────────
# Multiple independent tar.gz archives, no concatenation needed.
# Extraction of each archive is handled via postprocess_fn because
# ExtractSpec only covers a single main archive.

hi_mia = DatasetSpec(
    name="hi_mia",
    description="Mandarin wake-word and speaker corpus for far-field voice biometrics for speaker training and evaluation.",
    resources=[
        ResourceSpec(
            "http://kiwano-toolkit.com/files/himia/train.tar.gz",
            ChecksumSpec("md5", "95aeffa182a7e9da668785babc64fc8b"),
        ),
        # ResourceSpec("https://openslr.elda.org/resources/85/dev.tar.gz",     ChecksumSpec("md5", "f870cb7771412b173b93baac0cdf0f13")),
        # ResourceSpec("https://openslr.elda.org/resources/85/test.tar.gz", ChecksumSpec("md5", "f49396aaea461cfd70bcaa906f53cc31")),
    ],
    assemble=AssembleSpec(method="none"),
    extract=ExtractSpec(method="none"),  # handled by postprocess_fn
    max_retries=3,
)


# ── ArabCeleb ─────────────────────────────────────────────────────
# JSON manifest from GitHub + YouTube videos downloaded in parallel.
# The only declared resource is the JSON; videos are handled by postprocess_fn.

arab_celeb = DatasetSpec(
    name="arab_celeb",
    description="Arabic celebrity speech corpus collected from YouTube videos for speaker training and evaluation.",
    resources=[
        ResourceSpec(
            "https://github.com/CeLuigi/ArabCeleb/blob/main/utterance_info.json",
            ChecksumSpec("md5", "d22b74358ace281d866f00dad302d53f"),
            backend="github",
        ),
    ],
    assemble=AssembleSpec(method="none"),
    extract=ExtractSpec(method="none"),
    postprocess_kwargs={"num_jobs": 1},
    max_retries=3,
)

# ── CN-Celeb ──────────────────────────────────────────────────────
# Hybrid: two independent groups within one corpus.
#   Group 1 — CN-Celeb1: single tar.gz, direct extraction.
#   Group 2 — CN-Celeb2: 3 parts (.tar.gzaa/ab/ac), Python concatenation
#

cn_celeb = DatasetSpec(
    name="cn_celeb",
    description="Chinese celebrity speech corpus for speaker recognition in real-world conditions for speaker training and evaluation.",
    groups=[
        ResourceGroup(
            resources=[
                ResourceSpec(
                    "https://www.openslr.org/resources/82/cn-celeb_v2.tar.gz",
                    ChecksumSpec("md5", "7ab1b214028a7439e26608b2d5a0336c"),
                ),
            ],
            assemble=AssembleSpec(method="none"),
            extract=ExtractSpec(method="tgz"),
        ),
        ResourceGroup(
            resources=[
                ResourceSpec(
                    "https://www.openslr.org/resources/82/cn-celeb2_v2.tar.gzaa",
                    ChecksumSpec("md5", "4cdf738cff565ce35ad34274848659c3"),
                    kind="part",
                ),
                ResourceSpec(
                    "https://www.openslr.org/resources/82/cn-celeb2_v2.tar.gzab",
                    ChecksumSpec("md5", "4ff59a7009b79ef7043498ad3882b81b"),
                    kind="part",
                ),
                ResourceSpec(
                    "https://www.openslr.org/resources/82/cn-celeb2_v2.tar.gzac",
                    ChecksumSpec("md5", "3aa8e6e7f7ec4382f9926b1b31498228"),
                    kind="part",
                ),
            ],
            assemble=AssembleSpec(
                method="concat_clean",
                output_name="cn-celeb2.tar.gz",
                pattern="cn-celeb2_v2.tar.gza*",
            ),
            extract=ExtractSpec(method="tgz", archive_name="cn-celeb2.tar.gz"),
        ),
    ],
    max_retries=3,
    prepare_spec=prepare_cn_celeb,
)

# ── MUSAN ─────────────────────────────────────────────────────────

musan = DatasetSpec(
    name="musan",
    description="Music, speech, and noise corpus for data augmentation and robustness testing for speech data augmentation.",
    resources=[
        ResourceSpec(
            "https://www.openslr.org/resources/17/musan.tar.gz",
            ChecksumSpec("md5", "0c472d4fc0c5141eca47ad1ffeb2a7df"),
        ),
    ],
    assemble=AssembleSpec(method="none"),
    extract=ExtractSpec(method="tgz"),
    max_retries=3,
    prepare_spec=prepare_musan,
)

# ── RIRS Noises ───────────────────────────────────────────────────

rirs_noises = DatasetSpec(
    name="rirs_noises",
    description="Simulated and real room impulse responses with additive noise recordings for speech data augmentation.",
    resources=[
        ResourceSpec(
            "https://openslr.elda.org/resources/28/rirs_noises.zip",
            ChecksumSpec("md5", "e6f48e257286e05de56413b4779d8ffb"),
        ),
    ],
    assemble=AssembleSpec(method="none"),
    extract=ExtractSpec(method="zip"),
    max_retries=3,
    prepare_spec=prepare_rirs_noises,
)

# ── ESC-50 ────────────────────────────────────────────────────────

esc50 = DatasetSpec(
    name="esc50",
    description="Environmental sound dataset with 50 balanced classes for speech data augmentation.",
    resources=[
        ResourceSpec(
            "https://github.com/karoldvl/ESC-50/archive/master.zip",
            ChecksumSpec("md5", "629e8e9ebc1592bcceb06da6bec40275"),
        ),
    ],
    assemble=AssembleSpec(method="none"),
    extract=ExtractSpec(method="zip"),
    max_retries=3,
    prepare_spec=prepare_esc50,
)


hi_mia.postprocess_fn = _extract_all_tarballs
arab_celeb.postprocess_fn = _download_arab_celeb_videos


# ── Central registry ──────────────────────────────────────────────

REGISTRY: dict[str, DatasetSpec] = {
    "voxceleb1": voxceleb1,
    "voxceleb2": voxceleb2,
    "but_reverb": but_reverb,
    "musan": musan,
    "rirs_noises": rirs_noises,
    "cn_celeb": cn_celeb,
    "esc50": esc50,
    "hi_mia": hi_mia,
    "arab_celeb": arab_celeb,
}
