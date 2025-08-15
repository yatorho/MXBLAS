import fcntl
import functools
import hashlib
import os
import re
import subprocess
import uuid
from typing import Tuple, cast

from torch.utils.cpp_extension import CUDA_HOME

from mxblas.project.const import BUILD_FAILURE_INFO_PATH, BUILD_WARNING_INFO_PATH

from ..project import (
    CACHE_DIR_FLAG,
    DEBUG_FLAG,
    JIT_PRINT_NVCC_COMMAND_FLAG,
    NVCC_COMPILER_FLAG,
    PROJECT_NAME_ABBR_LOWER,
    PROJECT_NAME_FULL_LOWER,
    PTXAS_VERBOSE_FLAG,
)
from .runtime import Runtime, RuntimeCache
from .template import typename_map

runtime_cache = RuntimeCache()


def hash_to_hex(s: str) -> str:
    md5 = hashlib.md5()
    md5.update(s.encode("utf-8"))
    return md5.hexdigest()[0:12]


@functools.lru_cache(maxsize=None)
def get_jit_include_dir() -> str:
    return f"{os.path.dirname(os.path.abspath(__file__))}/../include"


@functools.lru_cache(maxsize=None)
def list_all_files(directory):
    file_list = []
    for dirpath, _, filenames in os.walk(directory, followlinks=True):
        for filename in filenames:
            file_list.append(os.path.join(dirpath, filename))
    return file_list


@functools.lru_cache(maxsize=None)
def get_repo_version() -> str:
    # Update include directories
    include_dir = f"{get_jit_include_dir()}/{PROJECT_NAME_ABBR_LOWER}"
    assert os.path.exists(include_dir), f"Cannot find include directory {include_dir}"
    md5 = hashlib.md5()
    # for filename in filter(lambda x: x.endswith('.cuh'), sorted(os.listdir(include_dir))):
    for filename in filter(
        lambda x: x.endswith(".cuh"), sorted(list_all_files(include_dir))
    ):
        # with open(f'{include_dir}/{filename}', 'rb') as f:
        with open(f"{filename}", "rb") as f:
            md5.update(f.read())

    return md5.hexdigest()[0:12]


@functools.lru_cache(maxsize=None)
def get_nvcc_compiler() -> Tuple[str, str]:
    paths = []
    if os.getenv(NVCC_COMPILER_FLAG):
        paths.append(os.getenv(NVCC_COMPILER_FLAG))
    paths.append(f"{CUDA_HOME}/bin/nvcc")

    # Try to find the first available NVCC compiler
    least_version_required = "12.3"
    version_pattern = re.compile(r"release (\d+\.\d+)")
    for path in paths:
        if os.path.exists(path):
            match = version_pattern.search(os.popen(f"{path} --version").read())
            version = match.group(1)
            assert match, f"Cannot get the version of NVCC compiler {path}"
            assert (
                version >= least_version_required
            ), f"NVCC {path} version {version} is lower than {least_version_required}"
            return path, version
    raise RuntimeError("Cannot find any available NVCC compiler")


@functools.lru_cache(maxsize=None)
def get_default_user_dir():
    if CACHE_DIR_FLAG in os.environ:
        path = os.getenv(CACHE_DIR_FLAG)
        os.makedirs(path, exist_ok=True)
        return path
    return os.path.expanduser("~") + f"/.{PROJECT_NAME_FULL_LOWER}"


@functools.lru_cache(maxsize=None)
def get_tmp_dir():
    return f"{get_default_user_dir()}/tmp"


@functools.lru_cache(maxsize=None)
def get_cache_dir():
    return f"{get_default_user_dir()}/cache"


def make_tmp_dir():
    tmp_dir = get_tmp_dir()
    os.makedirs(tmp_dir, exist_ok=True)
    return tmp_dir


def put(path, data, is_binary=False):
    # Write and do POSIX atomic replace
    tmp_file_path = f"{make_tmp_dir()}/file.tmp.{str(uuid.uuid4())}.{hash_to_hex(path)}"
    with open(tmp_file_path, "wb" if is_binary else "w") as f:
        f.write(data)
    os.replace(tmp_file_path, path)


def write_atomic_append(filepath, text):
    with open(filepath, "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(text)
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def build(name: str, arg_defs: tuple, code: str) -> Runtime:
    # Compiler flags
    nvcc_flags = [
        "-std=c++17",
        "-shared",
        "-O3",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-gencode=arch=compute_90a,code=sm_90a",
        "--ptxas-options=-v",
        "--ptxas-options=--register-usage-level=10"
        + (",--verbose" if PTXAS_VERBOSE_FLAG in os.environ else ""),
        # Suppress some unnecessary warnings, such as unused variables for certain `constexpr` branch cases
        "--diag-suppress=177,174,940",
        "-I/home/yatorho/doc/projs/cu_head",
        "-DDISABLE_MX_MAIN",
        "-lcuda",
    ]
    cxx_flags = ["-fPIC", "-O3", "-Wno-deprecated-declarations", "-Wno-abi"]
    flags = [*nvcc_flags, f'--compiler-options={",".join(cxx_flags)}']
    include_dirs = [get_jit_include_dir()]

    # Build signature
    signature = f"{name}$${get_repo_version()}$${code}$${get_nvcc_compiler()}$${flags}"
    name = f"kernel.{name}.{hash_to_hex(signature)}"
    path = f"{get_cache_dir()}/{name}"

    # Check runtime cache or file system hit
    global runtime_cache
    if runtime_cache[path] is not None:
        if os.getenv(DEBUG_FLAG, None):
            print(f"Using cached JIT runtime {name} during build")
        return cast(Runtime, runtime_cache[path])

    # Write the code
    os.makedirs(path, exist_ok=True)
    args_path = f"{path}/kernel.args"
    src_path = f"{path}/kernel.cu"
    put(
        args_path,
        ", ".join(
            [f"('{arg_def[0]}', {typename_map[arg_def[1]]})" for arg_def in arg_defs]
        ),
    )
    put(src_path, code)

    # Compile into a temporary SO file
    so_path = f"{path}/kernel.so"
    tmp_so_path = (
        f"{make_tmp_dir()}/nvcc.tmp.{str(uuid.uuid4())}.{hash_to_hex(so_path)}.so"
    )

    # Compile
    command = [
        get_nvcc_compiler()[0],
        src_path,
        "-o",
        tmp_so_path,
        *flags,
        *[f"-I{d}" for d in include_dirs],
    ]
    if os.getenv(DEBUG_FLAG, None) or os.getenv(JIT_PRINT_NVCC_COMMAND_FLAG, False):
        print(f"Compiling JIT runtime {name} with command {command}")
        print(f">>>> '{' '.join(command)}'")


    failure_info_path = os.getenv(BUILD_FAILURE_INFO_PATH)
    warning_info_path = os.getenv(BUILD_WARNING_INFO_PATH)

 
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if failure_info_path or warning_info_path:
        info_block = []
        info_block.append(f"\n===== JIT Compile Info =====\n")
        info_block.append(f"Command: {' '.join(command)}\n")
        info_block.append(f"Return code: {result.returncode}\n\n")
        info_block.append("==== STDOUT ====\n")
        info_block.append(result.stdout)
        info_block.append("\n==== STDERR ====\n")
        info_block.append(result.stderr)
        info_block.append(f"\n===== End of Info =====\n")

        info_text = ''.join(info_block)

        if failure_info_path and result.returncode != 0:
            try:
                write_atomic_append(failure_info_path, info_text)
            except Exception as file_err:
                print(f"Failed to write error to {failure_info_path}: {file_err}")
            
        if warning_info_path:
            try:
                write_atomic_append(warning_info_path, info_text)
            except Exception as file_err:
                print(f"Failed to write warning to {warning_info_path}: {file_err}")

    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to compile {src_path} with exit code {result.returncode}"
        )

    # Atomic replace SO file
    os.replace(tmp_so_path, so_path)

    # Put cache and return
    runtime_cache[path] = Runtime(path)
    return cast(Runtime, runtime_cache[path])
