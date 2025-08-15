"""
This is a naive template implementation for MX-GEMM.
It provides a basic structure, including MAIN-LOOP and EPILOGUE phases, FULL and PARTIAL operations.
"""

import itertools
from abc import ABC, abstractmethod
from typing import Any, Dict, Type, override

from mxblas.gemm.filters import (
    BasicShapeFilter,
    BMNKTileFilter,
    BMNKTileFilterV2,
    ClusterFilter,
    CSMemSwizzleBNFilter,
    Filter,
    NumConsumerX64EqualBMFilter,
    SMemSizeFilter,
)
from mxblas.gemm.generator import (
    BaseSignatureGenerator,
    ConsumerGenerator,
    HostLauncherGenerator,
    IncludeGenerator,
    NaiveGenerator,
    PredefinedCodeGenerator,
    ProducerGenerator,
    PromotionOperationType,
    PromotionPhaseType,
    SharedMemoryStructGenerator,
    TemplateType,
)
from mxblas.gemm.template_manager import register_template

from .descriptor import Layout, ScalarDType, columnable_layouts, rowable_layouts
from .keys import (
    EQ,
    GE,
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
    LT,
    NE,
    Condition,
    K_A_Layout,
    K_AB_Scale_Type,
    K_AS_Layout,
    K_B_Layout,
    K_BS_Layout,
    K_C_Layout,
    K_C_Scale_Type,
    K_C_Type,
    K_CS_Layout,
    K_Quant,
    get_all_keys,
)
from .template import KernelTemplate

# from .generator import


class NaiveTemplate(KernelTemplate, ABC):
    def __init__(self):
        super().__init__()

        signature_gen = BaseSignatureGenerator()
        predefined_gen = PredefinedCodeGenerator()
        shared_memory_gen = SharedMemoryStructGenerator()
        include_gen = IncludeGenerator()
        producer_gen = ProducerGenerator()
        host_launch_gen = HostLauncherGenerator(self.cpp_template_keys())

        self.include_generator = include_gen
        self.predefined_generator = predefined_gen
        self.shared_memory_generator = shared_memory_gen
        self.signature_generator = signature_gen
        self.producer_generator = producer_gen
        self.host_launcher_generator = host_launch_gen

    @property
    @abstractmethod
    def consumer_generator(self) -> ConsumerGenerator:
        raise NotImplementedError("Subclasses must implement consumer_generator")

    @override
    def cpp_template_keys(self):
        return get_all_keys()

    @override
    def generate(self):
        generator = NaiveGenerator(
            include_generator=self.include_generator,
            predefined_generator=self.predefined_generator,
            shared_memory_generator=self.shared_memory_generator,
            signature_generator=self.signature_generator,
            producer_generator=self.producer_generator,
            consumer_generator=self.consumer_generator,
            host_launcher_generator=self.host_launcher_generator,
            key_types=self.cpp_template_keys(),
        )
        return generator.generate()


class FTypeFilter(Filter):

    def __call__(self):
        # return EQ(K_SN % K_BN, 0).EQ(K_SM % K_BM, 0).EQ(K_SK % K_BK, 0)
        return GE(K_SN, K_BN).GE(K_SM, K_BM).DIVISIBLE_BY(K_SK, K_BK)


class MTypeFilter(Filter):

    def __call__(self):
        return LT(K_SK, K_K)


class PTypeFilter(Filter):

    def __call__(self):
        return LE(K_SN, K_BN).LE(K_SM, K_BM).DIVISIBLE_BY(K_SK, K_BK)


class ETypeFilter(Filter):

    def __call__(self):
        return EQ(K_SK, K_K)


def create_template_class(
    phase: PromotionPhaseType,
    operation: PromotionOperationType,
    quant: bool,
    layout: Layout,
) -> Type[NaiveTemplate]:
    class_name = f"{phase.value[0].upper()}{operation.value[0].upper()}{'H' if not quant else 'Q'}{layout.value[0].upper()}Template"

    def __init__(self):
        super(cls, self).__init__()
        self.consumer_generator_ = ConsumerGenerator(
            c_quant=quant,
            c_layout=layout,
            template_type=TemplateType(phase, operation),
        )

    @property
    def consumer_generator(self):
        return self.consumer_generator_

    def name(self):
        return f"{phase.value} & {operation.value} & {'High-Precision' if not quant else 'Quantization'} C & {layout.value} C"

    ### Matching layouts
    condition = (
        Condition()
        .IN(K_AS_Layout, rowable_layouts)
        .IN(K_BS_Layout, columnable_layouts)
        .EQ(K_CS_Layout, K_C_Layout)
    )

    if phase == PromotionPhaseType.MAIN_LOOP:
        condition = condition.GT(K_K, K_SK)
    else:
        condition = condition.EQ(K_K, K_SK)

    if operation == PromotionOperationType.PARTIAL:
        condition = condition.GT(K_M, K_SM).GT(K_N, K_SN)
    else:
        condition = condition.GE(K_M, K_SM).GE(K_N, K_SN)

    if quant:
        condition = condition.IN(K_C_Type, [ScalarDType.FP8_E5M2, ScalarDType.FP8_E4M3])
    else:
        condition = condition.IN(K_C_Type, [ScalarDType.FP16, ScalarDType.BF16])

    def match_condition(self):
        return condition.IN(K_C_Layout, [layout])

    filters = [
        BasicShapeFilter(),
        SMemSizeFilter(),
        NumConsumerX64EqualBMFilter(),
        ClusterFilter(),
        BMNKTileFilter(),
        # BMNKTileFilterV2(),
        CSMemSwizzleBNFilter(),
    ]

    filters.append(
        FTypeFilter() if operation == PromotionOperationType.FULL else PTypeFilter()
    )
    filters.append(
        MTypeFilter() if phase == PromotionPhaseType.MAIN_LOOP else ETypeFilter()
    )

    def prune_rules(self):
        return filters

    cls = type(
        class_name,
        (NaiveTemplate,),
        {
            "__init__": __init__,
            "__module__": __name__,
            "consumer_generator": consumer_generator,
            "name": name,
            "match_condition": match_condition,
            "prune_rules": prune_rules,
        },
    )

    return cls


phases = [PromotionPhaseType.MAIN_LOOP, PromotionPhaseType.EPILOGUE]
operations = [PromotionOperationType.FULL, PromotionOperationType.PARTIAL]
quants = [False, True]  # H/Q
layouts = [Layout.ROW_MAJOR, Layout.COLUMN_MAJOR]  # R/C

for phase, op, quant, layout in itertools.product(phases, operations, quants, layouts):
    cls = create_template_class(phase, op, quant, layout)
    register_template(cls)
