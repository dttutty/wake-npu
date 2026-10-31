"""Read CTranslate2 v6 model variables using its documented serializer layout.

The layout follows OpenNMT/CTranslate2's ModelSpec._serialize. This reader is
intentionally limited to v6 and validates offsets before exposing tensor data.
"""

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np


DTYPES = ("float32", "int8", "int16", "int32", "float16", "bfloat16")
ITEM_SIZES = (4, 1, 2, 4, 2, 2)


@dataclass(frozen=True)
class Variable:
    name: str
    shape: tuple[int, ...]
    dtype_id: int
    offset: int
    nbytes: int

    @property
    def dtype(self):
        return DTYPES[self.dtype_id]


class ModelWeights:
    def __init__(self, path):
        self.path = Path(path)
        self.variables = {}
        self.aliases = {}
        size = self.path.stat().st_size

        with self.path.open("rb") as stream:
            def read(fmt):
                length = struct.calcsize(fmt)
                data = stream.read(length)
                if len(data) != length:
                    raise ValueError(f"Truncated model at offset {stream.tell()}")
                return struct.unpack("<" + fmt, data)[0]

            def read_string():
                length = read("H")
                if length < 1 or length > 4096:
                    raise ValueError(f"Invalid string length {length}")
                data = stream.read(length)
                if len(data) != length or data[-1] != 0:
                    raise ValueError("Truncated or unterminated model string")
                return data[:-1].decode("utf-8")

            version = read("I")
            if version != 6:
                raise ValueError(f"Expected CTranslate2 binary v6, found v{version}")
            self.model_type = read_string()
            self.revision = read("I")
            count = read("I")
            if count > 100000:
                raise ValueError(f"Implausible variable count {count}")

            for _ in range(count):
                name = read_string()
                ndim = read("B")
                if ndim > 8:
                    raise ValueError(f"Implausible rank {ndim} for {name}")
                shape = tuple(read("I") for _ in range(ndim))
                dtype_id = read("B")
                if dtype_id >= len(DTYPES):
                    raise ValueError(f"Unknown dtype id {dtype_id} for {name}")
                nbytes = read("I")
                expected = int(np.prod(shape)) * ITEM_SIZES[dtype_id]
                if nbytes != expected or stream.tell() + nbytes > size:
                    raise ValueError(f"Invalid tensor byte count for {name}")
                if name in self.variables:
                    raise ValueError(f"Duplicate variable {name}")
                self.variables[name] = Variable(name, shape, dtype_id, stream.tell(), nbytes)
                stream.seek(nbytes, 1)

            alias_count = read("I")
            if alias_count > 100000:
                raise ValueError(f"Implausible alias count {alias_count}")
            for _ in range(alias_count):
                alias, target = read_string(), read_string()
                if alias in self.aliases:
                    raise ValueError(f"Duplicate alias {alias}")
                self.aliases[alias] = target
            if stream.tell() != size:
                raise ValueError(f"Trailing bytes after model: {size - stream.tell()}")
            if any(target not in self.variables for target in self.aliases.values()):
                raise ValueError("Alias points to an unknown variable")

    def array(self, name):
        variable = self.variables[self.aliases.get(name, name)]
        dtype = np.dtype("uint16" if variable.dtype == "bfloat16" else variable.dtype)
        with self.path.open("rb") as stream:
            stream.seek(variable.offset)
            data = stream.read(variable.nbytes)
        array = np.frombuffer(data, dtype=dtype).reshape(variable.shape)
        if variable.dtype == "bfloat16":
            from ml_dtypes import bfloat16
            return array.view(bfloat16)
        return array


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    model = ModelWeights(args.model)
    print(f"{model.model_type} revision {model.revision}, {len(model.variables)} variables")
    for variable in model.variables.values():
        print(f"{variable.name:65s} {str(variable.shape):20s} {variable.dtype}")
    for alias, target in model.aliases.items():
        print(f"{alias} -> {target}")
