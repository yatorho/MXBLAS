import os
from enum import Enum
from functools import lru_cache
from typing import Any, Collection, Dict, Iterable, List, Optional, Tuple

import torch

from mxblas.gemm.template import KernelTemplate
from mxblas.jit.runtime import Runtime
from mxblas.project.const import DEBUG_FLAG, PRINT_AUTOTUNE_FLAG

from ..jit.tuner import NoRuntimeGeneratedError, jit_tuner
from .descriptor import MXGEMMDescriptor
from .keys import key_value_to_cpp_Tvalue, str_to_keyT
from .template_manager import TemplateManager, to_pa_key_value


class RuntimeParameter(Enum):
    A_PTR = "A_PTR"
    B_PTR = "B_PTR"
    C_PTR = "C_PTR"
    AS_PTR = "AS_PTR"
    BS_PTR = "BS_PTR"
    CS_PTR = "CS_PTR"
    CUDA_STREAM = "CUDA_STREAM"


def apply_prune_rules(
    template: KernelTemplate,
    desc: MXGEMMDescriptor,
    space: Collection[Dict[str, Any]],
):
    conditions = template.prune_rules()

    for params in space:
        params |= to_pa_key_value(desc).items()

        if all(condition().evaluate(params) for condition in conditions):
            yield params


def to_cpp_value(space: Iterable[Dict[str, Any]]) -> tuple:
    cpp_values = []

    for params in space:
        cpp_value = {
            key: key_value_to_cpp_Tvalue(str_to_keyT(key), value)
            for key, value in params.items()
        }
        cpp_values.append(cpp_value)

    return tuple(cpp_values)


class KernelPool:

    def __init__(self):
        self.tuned: Dict[MXGEMMDescriptor, Tuple[Runtime, float]] = {}

    def get_time_or_reprofile(
        self,
        time: float,
        runtime: Runtime,
        kernel_tag: Optional[str],
        *args,
    ):
        if time != 0:
            return time

        from ..utils import GPU_bench

        def func():
            return runtime(*args)

        time = GPU_bench(func, iters=8, kernel_name=kernel_tag)

        return time

    def generate_kernel(
        self,
        desc: MXGEMMDescriptor,
        parameters: Dict[RuntimeParameter, torch.Tensor],
        space: Collection[Dict[str, Any]],
    ):
        if desc in self.tuned:
            runtime, _ = self.tuned[desc]
            if os.getenv(DEBUG_FLAG, None):
                print(f"Using cached runtime for descriptor: {desc}")
            return runtime

        path_and_templates: List[Tuple[str, KernelTemplate]] = list(
            TemplateManager().generate_template(desc)
        )
        for i, (path, template) in enumerate(path_and_templates):
            includes = (path,)
            code = """
using namespace mxblas;
run_kernel<{template_values}>(
    a_ptr,
    b_ptr,
    c_ptr,
    as_ptr,
    bs_ptr,
    cs_ptr,
    stream
);
""".format(
                template_values=", ".join(
                    f"{{{key_type}}}" for key_type in template.cpp_template_keys()
                ),
            )
            space_pruned = to_cpp_value(apply_prune_rules(template, desc, space))
            arg_defs = (
                ("a_ptr", parameters[RuntimeParameter.A_PTR].dtype),
                ("b_ptr", parameters[RuntimeParameter.B_PTR].dtype),
                ("c_ptr", parameters[RuntimeParameter.C_PTR].dtype),
                ("as_ptr", parameters[RuntimeParameter.AS_PTR].dtype),
                ("bs_ptr", parameters[RuntimeParameter.BS_PTR].dtype),
                ("cs_ptr", parameters[RuntimeParameter.CS_PTR].dtype),
                ("stream", torch.cuda.Stream),
            )
            args = (
                parameters[RuntimeParameter.A_PTR],
                parameters[RuntimeParameter.B_PTR],
                parameters[RuntimeParameter.C_PTR],
                parameters[RuntimeParameter.AS_PTR],
                parameters[RuntimeParameter.BS_PTR],
                parameters[RuntimeParameter.CS_PTR],
                parameters[RuntimeParameter.CUDA_STREAM],
            )

            if os.getenv(PRINT_AUTOTUNE_FLAG, None) or os.getenv(DEBUG_FLAG, None):
                print(
                    f"Generating and tuning template {template.name()} [{i + 1}/{len(path_and_templates)}]"
                )
                print(
                    f"Pruning space [{len(space)} -> {len(space_pruned)}] for template {template.name()} with descriptor {desc}."
                )

            try:
                runtime, time = jit_tuner.compile_and_tune(
                    name="mx_gemm_kernel",
                    keys={
                        "kernel_signature": str(desc)
                        + f" ::: template: {template.name()}"
                    },
                    space=space_pruned,
                    # space=(space_pruned[0] if not len(space_pruned) == 0 else {},),
                    includes=includes,
                    template=code,
                    arg_defs=arg_defs,
                    args=args,
                    allow_empty_space=True,
                    kernel_tag="mx_gemm",
                )
            except NoRuntimeGeneratedError as e:
                if os.getenv(PRINT_AUTOTUNE_FLAG, None) or os.getenv(DEBUG_FLAG, None):
                    print(f"Failed to generate runtime for descriptor: {desc}.")
                continue

            if desc in self.tuned:
                existing_runtime, existing_time = self.tuned[desc]

                existing_time = self.get_time_or_reprofile(
                    existing_time, existing_runtime, "mx_gemm", *args
                )
                time = self.get_time_or_reprofile(time, runtime, "mx_gemm", *args)

                if time < existing_time:
                    self.tuned[desc] = (runtime, time)
            else:
                # If the descriptor is not in the tuned dictionary, add it
                # with the current runtime and time.
                self.tuned[desc] = (runtime, time)

        if desc not in self.tuned:
            raise NoRuntimeGeneratedError(
                f"All templates failed to generate a runtime for descriptor: {desc}"
            )

        if os.getenv(PRINT_AUTOTUNE_FLAG, None) or os.getenv(DEBUG_FLAG, None):
            print(
                f"Generated kernel for descriptor: {desc} with time: {self.tuned[desc][1]:.4f} ms"
            )

        return self.tuned[desc][0]


pool = KernelPool()
