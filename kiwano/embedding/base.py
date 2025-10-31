import pickle
import subprocess
import sys

import sympy
import torch


class EmbeddingSet:
    """
    Minimal mapping-like container for named speaker embeddings.

    Behaves like a dict `{key: torch.Tensor}` with lightweight iteration and
    containment. Intended as an interchange object for piping serialized
    embeddings via stdin/stdout using custom `pkl:`/`pkl,t:` URL-like schemes.

    Examples
    --------
    Create and index:

    >>> E = EmbeddingSet()
    >>> import torch
    >>> E["utt1"] = torch.randn(192)
    >>> "utt1" in E
    True
    >>> E["utt1"].shape[0]
    192

    Iterate keys:

    >>> for k in E:
    ...     _ = E[k]  # torch.Tensor

    Count entries:

    >>> E.len()
    1
    """

    def __init__(self):
        self.h = {}

    def __getitem__(self, name: str):
        return self.h[name]

    def __setitem__(self, name: str, tensor: torch.Tensor):
        self.h[name] = tensor

    def __iter__(self):
        return iter(self.h)

    def __next__(self):
        if not hasattr(self, "_iter"):
            self._iter = iter(self.h)
        return next(self._h)

    def __contains__(self, name):
        return name in self.h

    def len(self):
        return len(self.h)


def write_pkl(arg: str, arr: EmbeddingSet):
    """
    Serialize an `EmbeddingSet` to pickle or text according to a URL-like spec.

    Formats
    -------
    - Pickle stream:
        * `pkl:-`    → write a single pickle to **stdout** (binary).
        * `pkl:<fp>` → write a single pickle to file `<fp>` (binary).
    - Text (space-separated, human-readable):
        * `pkl,t:-`    → write to **stdout** as lines:
                         `<key> <v0> <v1> ...`
        * `pkl,t:<fp>` → write to text file `<fp>` with same format.

    Parameters
    ----------
    arg : str
        Output target spec as above.
    arr : EmbeddingSet
        Container to serialize.

    Examples
    --------
    Write pickle to file:

    >>> E = EmbeddingSet()
    >>> E["utt1"] = torch.arange(3, dtype=torch.float)
    >>> write_pkl("pkl:/tmp/emb.pkl", E)  # doctest: +SKIP

    Write text to stdout:

    >>> write_pkl("pkl,t:-", E)  # doctest: +ELLIPSIS
    utt1 0.0 1.0 2.0
    """
    arg = arg.strip()

    if arg[0:4] == "pkl:":
        if arg[4] == "-":
            pickle.dump(arr, sys.stdout.buffer)
        else:
            file = open(arg[4:], "wb")
            pickle.dump(arr, file)
            file.close

    if arg[0:6] == "pkl,t:":
        if arg[6] == "-":
            for v in arr:
                print(v + " " + " ".join(map(str, arr[v].numpy())))
        else:
            file = open(arg[6:], "w")
            for v in arr:
                file.write(v + " " + " ".join(map(str, arr[v].numpy())) + "\n")
            file.close


def read_pkl(arg: str):
    """
    Deserialize an `EmbeddingSet` from pickle or text, including piped input.

    Formats
    -------
    - Pickle input:
        * `pkl:-`     → read a pickle from **stdin** (binary).
                        Supports concatenated pickles delimited by `b"usb."`:
                        multiple pickles are merged into a single `EmbeddingSet`.
        * `pkl:<fp>`  → read from file `<fp>` (binary).
        * `pkl:<cmd>|`→ execute `<cmd>` in shell, read binary stdout as pickle.
                        Also supports the same `b"usb."` multi-pickle merge.
    - Text input:
        * `pkl,t:<fp>`  → read text file with lines:
                          `<key> <v0> <v1> ...`
        * `pkl,t:<cmd>|`→ execute `<cmd>`; interpret stdout as a **pickle**
                          (note: this branch assumes pickled output).

    Parameters
    ----------
    arg : str
        Input spec as above.

    Returns
    -------
    EmbeddingSet
        The deserialized container.

    Examples
    --------
    Read from pickle file:

    >>> # write_pkl("pkl:/tmp/emb.pkl", E)
    >>> emb = read_pkl("pkl:/tmp/emb.pkl")  # doctest: +SKIP
    >>> isinstance(emb, EmbeddingSet)
    True

    Read text embeddings:

    >>> # with open("/tmp/emb.txt","w") as f: f.write("utt1 0.0 1.0 2.0\\n")
    >>> emb = read_pkl("pkl,t:/tmp/emb.txt")  # doctest: +SKIP
    >>> torch.is_tensor(emb["utt1"])
    True

    Pipe a command producing a pickle:

    >>> # read_pkl("pkl:cat /tmp/emb.pkl|")  # doctest: +SKIP
    """
    arg = arg.strip()

    if arg[0:4] == "pkl:":
        if arg[4] == "-":
            delimiter = b"usb."
            output = sys.stdin.buffer.read()
            my_list = [x + delimiter for x in output.split(delimiter) if x]
            if len(my_list) == 1:
                arr = pickle.loads(output)
                return arr
            else:
                emb = EmbeddingSet()
                for x in my_list:
                    tmp = pickle.loads(x)
                    for k in tmp:
                        emb[k] = tmp[k]
                return emb

            # arr = pickle.load(  sys.stdin.buffer )
            # return arr

        elif arg[-1] == "|":
            cmd = arg[4:-1]
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
            output, error = proc.communicate()

            delimiter = b"usb."
            my_list = [x + delimiter for x in output.split(delimiter) if x]
            if len(my_list) == 1:
                arr = pickle.loads(output)
                return arr
            else:
                emb = EmbeddingSet()
                for x in my_list:
                    tmp = pickle.loads(x)
                    for k in tmp:
                        emb[k] = tmp[k]
                return emb

        else:
            file = open(arg[4:], "rb")
            arr = pickle.load(file)
            file.close()
            return arr

    if arg[0:6] == "pkl,t:":
        if arg[-1] == "|":
            cmd = arg[6:-1]
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
            output, error = proc.communicate()
            arr = pickle.loads(output)
            return arr
        else:
            f = open(arg[6:], "r")
            # arr = pickle.load( file )
            emb = EmbeddingSet()
            for line in f:
                parts = line.split()

                key = parts[0]
                values = list(map(float, parts[1:]))

                tensor_values = torch.tensor(values)

                emb[key] = tensor_values

            f.close()
            return emb
