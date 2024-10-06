import torch
import sympy
import pickle
import subprocess
import sys


class EmbeddingSet():
    def __init__(self):
        self.h = {}

    def __getitem__(self, name: str):
        return self.h[ name ]

    def __setitem__(self, name: str, tensor: torch.Tensor):
        self.h[ name ] = tensor

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
                print(v+" "+" ".join(map(str, arr[v].numpy())))
        else:
            file = open(arg[6:], "w")
            for v in arr:
                file.write(v+" "+" ".join(map(str, arr[v].numpy()))+"\n")
            file.close



def read_pkl(arg: str):
    arg = arg.strip()

    if arg[0:4] == "pkl:":
        if arg[4] == "-":
            delimiter = b'usb.'
            output = sys.stdin.buffer.read()
            my_list = [x+delimiter for x in output.split(delimiter) if x]
            if len(my_list) == 1:
                arr = pickle.loads( output )
                return arr
            else:
                emb = EmbeddingSet()
                for x in my_list:
                    tmp = pickle.loads( x )
                    for k in tmp:
                        emb[ k ] = tmp[k]
                return emb

            #arr = pickle.load(  sys.stdin.buffer )
            #return arr

        elif arg[-1] == "|":
            cmd = arg[4:-1]
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
            output, error = proc.communicate()

            delimiter = b'usb.'
            my_list = [x+delimiter for x in output.split(delimiter) if x]
            if len(my_list) == 1:
                arr = pickle.loads( output )
                return arr
            else:
                emb = EmbeddingSet()
                for x in my_list:
                    tmp = pickle.loads( x )
                    for k in tmp:
                        emb[ k ] = tmp[k]
                return emb

        else:
            file = open(arg[4:], "rb")
            arr = pickle.load( file )
            file.close()
            return arr

    if arg[0:6] == "pkl,t:":
        if arg[-1] == "|":
            cmd = arg[6:-1]
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
            output, error = proc.communicate()
            arr = pickle.loads(  output )
            return arr
        else:
            f = open(arg[6:], "r")
            #arr = pickle.load( file )
            emb = EmbeddingSet()
            for line in f:
                parts = line.split()

                key = parts[0]
                values = list(map(float, parts[1:]))

                tensor_values = torch.tensor(values)

                emb[ key ] = tensor_values

            f.close()
            return emb


