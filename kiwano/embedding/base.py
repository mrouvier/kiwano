import io
import struct
import subprocess
import sys
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

import torch

MAGIC = b"SPKEMB01"  # 8-byte magic, identifies our format
VERSION = 1  # bump if format changes

DTYPE_TO_CODE = {
    torch.float32: b"f",
    torch.float16: b"h",
    torch.float64: b"d",
}
CODE_TO_DTYPE = {v: k for k, v in DTYPE_TO_CODE.items()}

DTYPE_ITEMSIZE = {
    torch.float32: 4,
    torch.float16: 2,
    torch.float64: 8,
}


def _read_exact(f, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise EOFError("Unexpected EOF")
    return b


class SpeakerEmbeddingWriter:
    """
    Writer binaire ou texte, basé sur un fichier ou un flux existant.

    - binary=True  : format binaire avec MAGIC / VERSION
    - binary=False : texte "utt_id v1 v2 ... vD" (float32)
    """

    def __init__(
        self,
        target: Union[str, Path, "io.IOBase"],
        *,
        binary: bool = True,
        append: bool = False,
    ):
        self.binary = binary
        self._own_file = False

        if isinstance(target, (str, Path)):
            self._own_file = True
            if self.binary:
                mode = "ab" if append else "wb"
                self.f = open(target, mode)
            else:
                mode = "a" if append else "w"
                self.f = open(target, mode, encoding="utf-8")
        else:
            self.f = target

        if self.binary:
            self.f.write(MAGIC)
            self.f.write(struct.pack("<I", VERSION))  # little-endian uint32

    def write(self, utt_id: str, embedding: torch.Tensor):
        if not isinstance(utt_id, str):
            raise TypeError("utt_id must be str")
        if not isinstance(embedding, torch.Tensor):
            raise TypeError("embedding must be a torch.Tensor")

        dt = embedding.dtype
        if dt not in DTYPE_TO_CODE:
            embedding = embedding.to(torch.float32)
            dt = torch.float32

        if not embedding.is_contiguous():
            embedding = embedding.contiguous()

        if embedding.dim() != 1:
            embedding = embedding.reshape(-1)

        if self.binary:
            dtype_char = DTYPE_TO_CODE[dt]
            utt_bytes = utt_id.encode("utf-8")
            if len(utt_bytes) > (1 << 31):
                raise ValueError("utt_id too long")

            # Format: utt_len(4) + utt_bytes + dtype(1) + dim(4) + raw_bytes
            self.f.write(struct.pack("<I", len(utt_bytes)))
            self.f.write(utt_bytes)
            self.f.write(dtype_char)
            self.f.write(struct.pack("<I", embedding.shape[0]))
            emb_bytes = embedding.detach().cpu().numpy().tobytes(order="C")
            self.f.write(emb_bytes)
        else:
            # Text: "utt_id v1 v2 ... vD"
            emb = embedding.detach().cpu().to(torch.float32).view(-1)
            values_str = " ".join(f"{float(x):.8g}" for x in emb)
            self.f.write(f"{utt_id} {values_str}\n")

    def close(self):
        if self._own_file:
            self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class SpeakerEmbeddingReader:
    """
    Reader binaire ou texte, basé sur un fichier ou un flux existant.

    - binary=True  : lit le format binaire (MAGIC / VERSION / etc.)
    - binary=False : lit le format texte "utt_id v1 v2 ... vD"
    """

    def __init__(
        self,
        source: Union[str, Path, "io.IOBase"],
        *,
        binary: bool = True,
        max_utt_len: int = 1 << 20,
    ):
        self.binary = binary
        self.max_utt_len = max_utt_len
        self._own_file = False

        if isinstance(source, (str, Path)):
            self._own_file = True
            if self.binary:
                self.f = open(source, "rb")
            else:
                self.f = open(source, "r", encoding="utf-8")
        else:
            self.f = source

        if self.binary:
            self._consume_segment_header(initial=True)

    # ---------- Binaire ----------

    def _consume_segment_header(self, initial: bool = False):
        hdr = self.f.read(8)
        if not hdr:
            if initial:
                raise EOFError("Empty file")
            return False
        if hdr != MAGIC:
            if initial:
                raise ValueError(f"Wrong magic: {hdr!r}, expected {MAGIC!r}")
            raise ValueError("Stream desync: expected MAGIC")
        (version,) = struct.unpack("<I", _read_exact(self.f, 4))
        if version != VERSION:
            raise ValueError(f"Unsupported version {version}")
        return True

    def _iter_binary(self) -> Iterator[Tuple[str, torch.Tensor]]:
        f = self.f
        while True:
            len_or_magic = f.read(4)
            if not len_or_magic:
                break  # EOF

            if len_or_magic == MAGIC[:4]:
                rest = _read_exact(f, 4)
                if rest == MAGIC[4:]:
                    (version,) = struct.unpack("<I", _read_exact(f, 4))
                    if version != VERSION:
                        raise ValueError(f"Unsupported version {version}")
                    continue
                else:
                    f.seek(-4, 1)
                    (utt_len,) = struct.unpack("<I", len_or_magic)
            else:
                (utt_len,) = struct.unpack("<I", len_or_magic)

            if utt_len > self.max_utt_len:
                raise ValueError(f"Unreasonable utt_id length: {utt_len}")

            utt_bytes = _read_exact(f, utt_len)
            utt_id = utt_bytes.decode("utf-8")

            dtype_char = _read_exact(f, 1)
            dtype_type = CODE_TO_DTYPE.get(dtype_char)
            if dtype_type is None:
                raise ValueError(f"Unknown dtype code {dtype_char!r}")

            (dim,) = struct.unpack("<I", _read_exact(self.f, 4))
            if dim <= 0:
                raise ValueError(f"Invalid embedding dim: {dim}")

            itemsize = DTYPE_ITEMSIZE[dtype_type]
            nbytes = itemsize * dim
            emb_bytes = _read_exact(f, nbytes)

            buf = bytearray(emb_bytes)
            embedding = torch.frombuffer(buf, dtype=dtype_type, count=dim)
            embedding = embedding.clone()

            yield utt_id, embedding

    # ---------- Texte ----------

    def _iter_text(self) -> Iterator[Tuple[str, torch.Tensor]]:
        for line in self.f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            utt_id = parts[0]
            if len(utt_id.encode("utf-8")) > self.max_utt_len:
                raise ValueError(f"Unreasonable utt_id length: {len(utt_id)}")

            if len(parts) == 1:
                raise ValueError(f"No embedding values for utt_id {utt_id!r}")

            try:
                values = [float(x) for x in parts[1:]]
            except ValueError as e:
                raise ValueError(
                    f"Failed to parse embedding for {utt_id!r}: {e}"
                ) from e

            emb = torch.tensor(values, dtype=torch.float32)
            yield utt_id, emb

    # ---------- API public ----------

    def __iter__(self) -> Iterator[Tuple[str, torch.Tensor]]:
        if self.binary:
            yield from self._iter_binary()
        else:
            yield from self._iter_text()

    def close(self):
        if self._own_file:
            self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------
# Specs pkl:...
# ---------------------------------------------------------------------


def parse_pkl_spec(spec: str):
    """
    Retourne (binary: bool, kind: str, target: str)

    kind ∈ {"file", "stdio", "cmd"}.
    - "file" : target est un chemin
    - "stdio": target == "-"
    - "cmd"  : target est une commande (entrée uniquement), finit par '|'
    """
    if not spec.startswith("pkl"):
        raise ValueError(f"Unsupported spec (must start with 'pkl'): {spec!r}")

    rest = spec[3:]
    text_mode = False
    if rest.startswith(",t"):
        text_mode = True
        rest = rest[2:]

    if not rest.startswith(":"):
        raise ValueError(f"Bad spec (missing ':'): {spec!r}")

    target = rest[1:].strip()
    binary = not text_mode

    if target.endswith("|"):
        cmd = target[:-1].strip()
        if not cmd:
            raise ValueError(f"Empty command in spec: {spec!r}")
        return binary, "cmd", cmd

    if target == "-":
        return binary, "stdio", target
    else:
        return binary, "file", target


def open_output_writer(spec: str) -> SpeakerEmbeddingWriter:
    """
    Crée un writer vers fichier ou stdout.
    """
    binary, kind, target = parse_pkl_spec(spec)

    if kind == "cmd":
        raise ValueError("Command specs are not supported for output")
    elif kind == "stdio":
        sink = sys.stdout.buffer if binary else sys.stdout
    elif kind == "file":
        sink = target  # path
    else:
        raise ValueError(f"Unknown kind {kind!r}")

    writer = SpeakerEmbeddingWriter(sink, binary=binary, append=False)
    return writer


def open_input_reader(spec: str):
    """
    Retourne (reader, proc) où proc est un éventuel Popen
    (si on lit depuis une commande).
    """
    binary, kind, target = parse_pkl_spec(spec)
    proc: Optional[subprocess.Popen] = None

    if kind == "file":
        source = target
    elif kind == "stdio":
        source = sys.stdin.buffer if binary else sys.stdin
    elif kind == "cmd":
        if binary:
            proc = subprocess.Popen(target, shell=True, stdout=subprocess.PIPE)
        else:
            proc = subprocess.Popen(
                target,
                shell=True,
                stdout=subprocess.PIPE,
                text=True,
                encoding="utf-8",
            )
        if proc.stdout is None:
            raise RuntimeError("Failed to capture stdout of command")
        source = proc.stdout
    else:
        raise ValueError(f"Unknown kind {kind!r}")

    reader = SpeakerEmbeddingReader(source, binary=binary)
    return reader, proc


# ---------------------------------------------------------------------
# Chargement des embeddings
# ---------------------------------------------------------------------


def load_embeddings(spec: str) -> dict:
    reader, proc = open_input_reader(spec)
    emb_dict = {}
    try:
        for utt_id, emb in reader:
            emb_dict[utt_id] = emb.to(torch.float32)
    finally:
        reader.close()
        if proc is not None:
            proc.wait()
    return emb_dict
