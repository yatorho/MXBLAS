import itertools
from abc import ABC, abstractmethod
from typing import override

from mxblas.gemm.descriptor import (
    Layout,
    ScalarDType,
    columnable_layouts,
    rowable_layouts,
)
from mxblas.gemm.filter import (
    BasicShapeFilter,
    BMNKTileFilter,
    ClusterFilter,
    CSMemSwizzleBNFilter,
    Filter,
    NumConsumerX64EqualBMFilter,
    SMemSizeFilter,
)
from mxblas.gemm.generator import IncludeGenerator, PredefinedCodeGenerator
from mxblas.gemm.keys import (
    DIVISIBLE_BY,
    EQ,
    K_BK,
    K_BM,
    K_BN,
    K_K,
    K_M,
    K_N,
    K_SK,
    K_SM,
    K_SN,
    LE,
    Condition,
    K_AS_Layout,
    K_BS_Layout,
    K_C_Layout,
    K_C_Type,
    K_Num_Threads,
    get_all_keys,
)
from mxblas.gemm.mainloop_tma_scales_generator import (
    ConsumerGenerator,
    HostLauncherGenerator,
    ProducerGenerator,
    ScalesUsingTMA,
    SharedMemoryStructGenerator,
    SignatureGenerator,
    TMAScalesGenerator,
)
from mxblas.gemm.template import KernelTemplate
from mxblas.gemm.template_manager import register_template


class TMAScalesTemplate(KernelTemplate, ABC):

    @property
    @abstractmethod
    def scales_using_tma(self) -> ScalesUsingTMA:
        raise NotImplementedError(
            "Subclasses must implement `scales_using_tma` property"
        )

    @property
    @abstractmethod
    def c_quant(self) -> bool:
        raise NotImplementedError("Subclasses must implement `c_quant` property")

    @property
    @abstractmethod
    def c_layout(self) -> Layout:
        raise NotImplementedError("Subclasses must implement `c_layout` property")

    @override
    def cpp_template_keys(self):
        return get_all_keys()

    def generate(self):
        generator = TMAScalesGenerator(
            scales_using_tma=self.scales_using_tma,
            include_generator=IncludeGenerator(),
            predefined_generator=PredefinedCodeGenerator(),
            shared_memory_generator=SharedMemoryStructGenerator(self.scales_using_tma),
            signature_generator=SignatureGenerator(self.scales_using_tma),
            producer_generator=ProducerGenerator(self.scales_using_tma),
            consumer_generator=ConsumerGenerator(
                self.scales_using_tma, self.c_quant, self.c_layout
            ),
            host_launcher_generator=HostLauncherGenerator(
                self.scales_using_tma, self.cpp_template_keys()
            ),
            key_types=self.cpp_template_keys(),
        )

        return generator.generate()


class TMAScalesFilter(Filter):
    def __init__(self, scales_using_tma: ScalesUsingTMA):
        self.scales_using_tma = scales_using_tma

    def __call__(self) -> Condition:
        condition = DIVISIBLE_BY(K_SK, K_BK)

        if self.scales_using_tma in [
            ScalesUsingTMA.A_SCALES,
            ScalesUsingTMA.BOTH_SCALES,
        ]:
            condition = condition.GE(K_BM // K_SM, 16)
        if self.scales_using_tma in [
            ScalesUsingTMA.B_SCALES,
            ScalesUsingTMA.BOTH_SCALES,
        ]:
            condition = condition.GE(K_BN // K_SN, 16)

        return condition


class MainLoopRegisterFilter(Filter):

    def __call__(self) -> Condition:
        MAX_MATH_REGISTERS_PER_THREAD = 224
        return LE(2 * K_BM * K_BN // (K_Num_Threads - 128), MAX_MATH_REGISTERS_PER_THREAD)


def create_template_class(
    scales_using_tma_p: ScalesUsingTMA,
    c_quant_p: bool,
    c_layout_p: Layout,
):
    class_name = f"{scales_using_tma_p.value}{'H' if not c_quant_p else 'Q'}{c_layout_p.value[0].upper()}Template"

    ### Initializer
    def __init__(self):
        self.scales_using_tma_ = scales_using_tma_p
        self.c_quant_ = c_quant_p
        self.c_layout_ = c_layout_p

    ### Properties
    @property
    def scales_using_tma(self) -> ScalesUsingTMA:
        return self.scales_using_tma_

    @property
    def c_quant(self) -> bool:
        return self.c_quant_

    @property
    def c_layout(self) -> Layout:
        return self.c_layout_

    ### Name
    def name(self):
        return f"{scales_using_tma_p.value} & {'High-Precision' if not c_quant_p else 'Quantization'} C & {c_layout_p.value} C"

    ### Match condition
    condition = Condition().GT(K_K, K_SK)  # main-loop
    if scales_using_tma_p in [ScalesUsingTMA.A_SCALES, ScalesUsingTMA.BOTH_SCALES]:
        condition = condition.IN(K_AS_Layout, columnable_layouts)
    else:
        condition = condition.IN(K_AS_Layout, rowable_layouts)
    if scales_using_tma_p in [ScalesUsingTMA.B_SCALES, ScalesUsingTMA.BOTH_SCALES]:
        condition = condition.IN(K_BS_Layout, rowable_layouts)
    else:
        condition = condition.IN(K_BS_Layout, columnable_layouts)

    condition = condition.GT(K_M, K_SM).GT(K_N, K_SN)

    if c_quant_p:
        condition = condition.IN(K_C_Type, [ScalarDType.FP8_E5M2, ScalarDType.FP8_E4M3])
    else:
        condition = condition.IN(K_C_Type, [ScalarDType.FP16, ScalarDType.BF16])

    def match_condition(self):
        return condition.IN(K_C_Layout, [c_layout_p])

    ### Prune rules
    filters = [
        BasicShapeFilter(),
        SMemSizeFilter(),
        NumConsumerX64EqualBMFilter(),
        ClusterFilter(),
        BMNKTileFilter(),
        CSMemSwizzleBNFilter(),
        TMAScalesFilter(scales_using_tma_p),
        MainLoopRegisterFilter(),
    ]

    def prune_rules(self):
        return filters

    cls = type(
        class_name,
        (TMAScalesTemplate,),
        {
            "__init__": __init__,
            "__module__": __name__,
            "scales_using_tma": scales_using_tma,
            "c_quant": c_quant,
            "c_layout": c_layout,
            "name": name,
            "match_condition": match_condition,
            "prune_rules": prune_rules,
        },
    )

    return cls


scales_using_tmas = [
    ScalesUsingTMA.A_SCALES,
    ScalesUsingTMA.B_SCALES,
    ScalesUsingTMA.BOTH_SCALES,
]
c_quants = [False, True]
c_layouts = [Layout.ROW_MAJOR, Layout.COLUMN_MAJOR]

for scale_using_tma, c_quant, c_layout in itertools.product(
    scales_using_tmas, c_quants, c_layouts
):
    cls = create_template_class(
        scales_using_tma_p=scale_using_tma,
        c_quant_p=c_quant,
        c_layout_p=c_layout,
    )

    register_template(cls)
    register_template(cls)
