"""Run an ONNX graph with a given libonnxruntime.so through the ONNX Runtime C API (ctypes).

Why: sherpa-onnx 2.0.1 (NexaRoom's runtime) bundles ONNX Runtime 1.28.2 as
sherpa_onnx/lib/libonnxruntime.so. No Python or npm package of 1.28.x exists
here (npm has 1.27.0 and 1.29.0 only), so to feed IDENTICAL input features to
the deployed runtime version we call that shared library directly.

Only the few C-API entries needed for CPU inference are used, by their index
in the OrtApi function table (onnxruntime_c_api.h; the table is append-only,
so the indices are the same in every 1.x release):
  2 GetErrorMessage, 3 CreateEnv, 7 CreateSession, 9 Run, 10 CreateSessionOptions,
  24 SetIntraOpNumThreads, 25 SetInterOpNumThreads,
  49 CreateTensorWithDataAsOrtValue, 69 CreateCpuMemoryInfo.
The output tensor is pre-allocated over a NumPy buffer, so ORT allocates no
output memory. The wrapper is validated in experiments/20_runtime_isolation.py:
with the 1.31.1 library shipped in the onnxruntime Python wheel it reproduces
the Python package's outputs (results/runtime_isolation.json, "ctypes_validation").
"""

from __future__ import annotations

import ctypes as C
from pathlib import Path

import numpy as np

SHERPA_ORT = Path("/home/ali/Desktop/Articles/.venv/lib/python3.12/site-packages/sherpa_onnx/lib/libonnxruntime.so")
PY_ORT = Path("/home/ali/Desktop/Articles/.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.31.1")

_API_VERSION = 16  # any version <= the library's ORT_API_VERSION returns the same table
_VP = C.c_void_p


class _ApiBase(C.Structure):
    _fields_ = [("GetApi", C.CFUNCTYPE(_VP, C.c_uint32)), ("GetVersionString", C.CFUNCTYPE(C.c_char_p))]


class OrtC:
    """Minimal CPU session over one libonnxruntime.so."""

    _libs: dict = {}

    def __init__(self, lib_path: str | Path, model_path: str | Path, input_name: str, output_name: str,
                 out_dim: int, threads: int = 1, out_shape: tuple | None = None) -> None:
        lib_path = str(lib_path)
        if lib_path not in OrtC._libs:
            lib = C.CDLL(lib_path, mode=C.RTLD_LOCAL)
            lib.OrtGetApiBase.restype = C.POINTER(_ApiBase)
            base = lib.OrtGetApiBase().contents
            api = C.cast(base.GetApi(_API_VERSION), C.POINTER(_VP))
            OrtC._libs[lib_path] = (lib, base, api)
        self.lib, base, self.api = OrtC._libs[lib_path]
        self.version = base.GetVersionString().decode()
        f = self._fn
        self._err = f(2, C.c_char_p, _VP)
        create_env = f(3, _VP, C.c_int, C.c_char_p, C.POINTER(_VP))
        create_opts = f(10, _VP, C.POINTER(_VP))
        intra = f(24, _VP, _VP, C.c_int)
        inter = f(25, _VP, _VP, C.c_int)
        create_sess = f(7, _VP, _VP, C.c_char_p, _VP, C.POINTER(_VP))
        mem = f(69, _VP, C.c_int, C.c_int, C.POINTER(_VP))
        self._tensor = f(49, _VP, _VP, _VP, C.c_size_t, C.POINTER(C.c_int64), C.c_size_t, C.c_int, C.POINTER(_VP))
        self._run = f(9, _VP, _VP, _VP, C.POINTER(C.c_char_p), C.POINTER(_VP), C.c_size_t,
                      C.POINTER(C.c_char_p), C.c_size_t, C.POINTER(_VP))
        self.env, self.opts, self.sess, self.mem = _VP(), _VP(), _VP(), _VP()
        self._ok(create_env(2, b"ortc", C.byref(self.env)))
        self._ok(create_opts(C.byref(self.opts)))
        self._ok(intra(self.opts, threads))
        self._ok(inter(self.opts, 1))
        self._ok(create_sess(self.env, str(model_path).encode(), self.opts, C.byref(self.sess)))
        self._ok(mem(1, 0, C.byref(self.mem)))  # OrtArenaAllocator, OrtMemTypeDefault
        self.in_names_py = [input_name] if isinstance(input_name, str) else list(input_name)
        self.in_names = (C.c_char_p * len(self.in_names_py))(*[n.encode() for n in self.in_names_py])
        self.out_name = output_name
        self.out_names = (C.c_char_p * 1)(output_name.encode())
        self.out_dim = out_dim
        self.out_shape = tuple(out_shape) if out_shape else (1, out_dim)

    def _fn(self, idx: int, restype, *argtypes):
        return C.CFUNCTYPE(restype, *argtypes)(self.api[idx])

    def _ok(self, status) -> None:
        if status:
            raise RuntimeError(f"onnxruntime {self.version}: {self._err(status).decode()}")

    def _value(self, x: np.ndarray):
        elem = {np.dtype(np.float32): 1, np.dtype(np.int64): 7}[x.dtype]
        shape = (C.c_int64 * max(x.ndim, 1))(*x.shape)
        v = _VP()
        self._ok(self._tensor(self.mem, x.ctypes.data, x.nbytes, shape, x.ndim, elem, C.byref(v)))
        return v

    def run(self, x) -> np.ndarray:
        """x: float32 array (single-input graph) or a list of arrays in input order;
        returns the (out_dim,) output named ``output_name``."""

        xs = [x] if isinstance(x, np.ndarray) else list(x)
        xs = [np.ascontiguousarray(a, dtype=np.int64 if a.dtype.kind == "i" else np.float32) for a in xs]
        assert len(xs) == len(self.in_names_py)
        ins = (_VP * len(xs))(*[self._value(a) for a in xs])
        y = np.zeros(self.out_shape, dtype=np.float32)
        outs = (_VP * 1)(self._value(y))
        self._ok(self._run(self.sess, None, self.in_names, ins, len(xs), self.out_names, 1, outs))
        return y.reshape(-1).copy()
