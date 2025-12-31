# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import lru_cache
import inspect
import pickle
from collections.abc import Sequence
from types import FunctionType
from typing import Any, Callable, TypeAlias, cast

import cloudpickle
import msgspec
import numpy as np
import torch
import zmq
from msgspec import msgpack

CUSTOM_TYPE_PICKLE = 1
CUSTOM_TYPE_CLOUDPICKLE = 2
CUSTOM_TYPE_RAW_VIEW = 3
CUSTOM_TYPE_NUMPY_ARRAY = 4
CUSTOM_TYPE_TORCH_TENSOR = 5

bytestr: TypeAlias = bytes | bytearray | memoryview | zmq.Frame


class MsgpackEncoder:
    """Encoder with custom torch tensor and numpy array serialization.

    Note that unlike vanilla `msgspec` Encoders, this interface is generally
    not thread-safe when encoding tensors / numpy arrays.

    By default, arrays below 256B are serialized inline Larger will get sent
    via dedicated messages. Note that this is a per-tensor limit.
    """

    def __init__(self, size_threshold: int | None = None):
        if size_threshold is None:
            size_threshold = 256
        self.encoder = msgpack.Encoder(enc_hook=self.enc_hook)
        # This is used as a local stash of buffers that we can then access from
        # our custom `msgspec` hook, `enc_hook`. We don't have a way to
        # pass custom data to the hook otherwise.
        self.aux_buffers: list[bytestr] | None = None
        self.size_threshold = size_threshold

    def encode(self, obj: Any) -> Sequence[bytestr]:
        try:
            self.aux_buffers = bufs = cast(list[bytestr], [b""])
            bufs[0] = self.encoder.encode(obj)
            # This `bufs` list allows us to collect direct pointers to backing
            # buffers of tensors and np arrays, and return them along with the
            # top-level encoded buffer instead of copying their data into the
            # new buffer.
            return bufs
        finally:
            self.aux_buffers = None

    def encode_into(self, obj: Any, buf: bytearray) -> Sequence[bytestr]:
        try:
            self.aux_buffers = [buf]
            bufs = self.aux_buffers
            self.encoder.encode_into(obj, buf)
            return bufs
        finally:
            self.aux_buffers = None

    def enc_hook(self, obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return msgpack.Ext(CUSTOM_TYPE_TORCH_TENSOR, self._dump_tensor(obj))
        # Fall back to pickle for object or void kind ndarrays.
        if isinstance(obj, np.ndarray) and obj.dtype.kind not in ("O", "V"):
            return msgpack.Ext(CUSTOM_TYPE_NUMPY_ARRAY, self._dump_ndarray(obj))

        if isinstance(obj, FunctionType):
            # `pickle` is generally faster than cloudpickle, but can have
            # problems serializing methods.
            return msgpack.Ext(CUSTOM_TYPE_CLOUDPICKLE, cloudpickle.dumps(obj))

        return msgpack.Ext(
            CUSTOM_TYPE_PICKLE, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        )

    def _dump_ndarray(self, obj: np.ndarray):
        assert self.aux_buffers is not None
        # If the array is non-contiguous, we need to copy it first
        arr_data = obj.data if obj.flags.c_contiguous else obj.tobytes()
        if not obj.shape or obj.nbytes < self.size_threshold:
            # Encode small arrays and scalars inline. Using this extension type
            # ensures we can avoid copying when decoding.
            data = msgpack.Ext(CUSTOM_TYPE_RAW_VIEW, arr_data)
        else:
            # Otherwise encode index of backing buffer to avoid copy.
            data = len(self.aux_buffers)
            self.aux_buffers.append(arr_data)

        # We serialize the ndarray as a tuple of native types.
        # The data is either inlined if small, or an index into a list of
        # backing buffers that we've stashed in `aux_buffers`.
        dtype = self.encoder.encode(obj.dtype.str)
        shape = self.encoder.encode(obj.shape)
        data = self.encoder.encode(data)
        return b"".join(
            [
                len(dtype).to_bytes(4, "little"),
                dtype,
                len(shape).to_bytes(4, "little"),
                shape,
                len(data).to_bytes(4, "little"),
                data,
            ]
        )

    def _dump_tensor(self, obj: torch.Tensor):
        assert self.aux_buffers is not None
        # view the tensor as a contiguous 1D array of bytes
        arr = obj.flatten().contiguous().view(torch.uint8).numpy()
        if obj.nbytes < self.size_threshold:
            # Smaller tensors are encoded inline, just like ndarrays.
            data = msgpack.Ext(CUSTOM_TYPE_RAW_VIEW, arr.data)
        else:
            # Otherwise encode index of backing buffer to avoid copy.
            data = len(self.aux_buffers)
            self.aux_buffers.append(arr.data)
        dtype = self.encoder.encode(str(obj.dtype).removeprefix("torch."))
        shape = self.encoder.encode(obj.shape)
        data = self.encoder.encode(data)
        return b"".join(
            [
                len(dtype).to_bytes(4, "little"),
                dtype,
                len(shape).to_bytes(4, "little"),
                shape,
                len(data).to_bytes(4, "little"),
                data,
            ]
        )


class MsgpackDecoder:
    """Decoder with custom torch tensor and numpy array serialization.

    Note that unlike vanilla `msgspec` Decoders, this interface is generally
    not thread-safe when encoding tensors / numpy arrays.
    """

    def __init__(self, t: Any | None = None):
        args = () if t is None else (t,)
        self.decoder = msgpack.Decoder(*args, ext_hook=self.ext_hook)
        self.aux_buffers: Sequence[bytestr] = ()

    def decode(self, bufs: bytestr | Sequence[bytestr]) -> Any:
        if isinstance(bufs, bytestr):
            return self.decoder.decode(bufs)  # type: ignore

        self.aux_buffers = bufs
        try:
            return self.decoder.decode(bufs[0])  # type: ignore
        finally:
            self.aux_buffers = ()

    def _load_ndarray(self, arr: bytes) -> np.ndarray:
        dtype_size = int.from_bytes(arr[0:4], "little")
        dtype = self.decoder.decode(arr[4 : 4 + dtype_size])
        shape_start = 4 + dtype_size
        shape_size = int.from_bytes(arr[shape_start : shape_start + 4], "little")
        shape = self.decoder.decode(arr[shape_start + 4 : shape_start + 4 + shape_size])
        data_start = shape_start + 4 + shape_size
        data_size = int.from_bytes(arr[data_start : data_start + 4], "little")
        data = self.decoder.decode(arr[data_start + 4 : data_start + 4 + data_size])

        # zero-copy decode. We assume the ndarray will not be kept around,
        # as it now locks the whole received message buffer in memory.
        buffer = self.aux_buffers[data] if isinstance(data, int) else data
        return np.frombuffer(buffer, dtype=dtype).reshape(shape)

    def _load_tensor(self, arr: Any) -> torch.Tensor:
        dtype_size = int.from_bytes(arr[0:4], "little")
        dtype = self.decoder.decode(arr[4 : 4 + dtype_size])
        shape_start = 4 + dtype_size
        shape_size = int.from_bytes(arr[shape_start : shape_start + 4], "little")
        shape = self.decoder.decode(arr[shape_start + 4 : shape_start + 4 + shape_size])
        data_start = shape_start + 4 + shape_size
        data_size = int.from_bytes(arr[data_start : data_start + 4], "little")
        data = self.decoder.decode(arr[data_start + 4 : data_start + 4 + data_size])

        # Copy from inline representation, to decouple the memory storage
        # of the message from the original buffer. And also make Torch
        # not complain about a readonly memoryview.
        buffer = self.aux_buffers[data] if isinstance(data, int) else bytearray(data)
        torch_dtype = getattr(torch, dtype)
        assert isinstance(torch_dtype, torch.dtype)
        if not buffer:  # torch.frombuffer doesn't like empty buffers
            assert 0 in shape
            return torch.empty(shape, dtype=torch_dtype)
        # Create uint8 array
        arr = torch.frombuffer(buffer, dtype=torch.uint8)
        # Convert back to proper shape & type
        return arr.view(torch_dtype).view(shape)

    def ext_hook(self, code: int, data: memoryview) -> Any:
        if code == CUSTOM_TYPE_RAW_VIEW:
            return data
        if code == CUSTOM_TYPE_NUMPY_ARRAY:
            return self._load_ndarray(data)
        if code == CUSTOM_TYPE_TORCH_TENSOR:
            return self._load_tensor(data)

        if code == CUSTOM_TYPE_PICKLE:
            return pickle.loads(data)
        if code == CUSTOM_TYPE_CLOUDPICKLE:
            return cloudpickle.loads(data)

        raise NotImplementedError(f"Extension type code {code} is not supported")
