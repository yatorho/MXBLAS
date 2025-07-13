"""
This is a naive template implementation for MX-GEMM.
It provides a basic structure, including MAIN-LOOP and EPILOGUE phases, FULL and PARTIAL operations.
"""

import itertools
from abc import ABC, abstractmethod
from typing import Any, Dict, Type, override

from mxblas.gemm.filter import (
    BMNKTileFilter,
    ClusterFilter,
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
    TemplateType,
)
from mxblas.gemm.template_manager import register_template

from .descriptor import Layout, ScalarDType
from .keys import (
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
    Condition,
    K_AB_Scale_Type,
    K_C_Layout,
    K_C_Scale_Type,
    K_C_Type,
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
        include_gen = IncludeGenerator()
        producer_gen = ProducerGenerator()
        host_launch_gen = HostLauncherGenerator(self.cpp_template_keys())

        self.include_generator = include_gen
        self.predefined_generator = predefined_gen
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
            signature_generator=self.signature_generator,
            producer_generator=self.producer_generator,
            consumer_generator=self.consumer_generator,
            host_launcher_generator=self.host_launcher_generator,
            key_types=self.cpp_template_keys(),
        )
        return generator.generate()


class FTypeFilter(Filter):

    def __call__(self):
        return EQ(K_SN % K_BN, 0).EQ(K_SM % K_BM, 0).EQ(K_SK % K_BK, 0)


def create_template_class(
    phase: str, operation: str, quant: bool, layout: str
) -> Type[NaiveTemplate]:
    class_name = (
        f"{phase[0]}{operation[0]}{'H' if not quant else 'Q'}{layout[0]}Template"
    )

    phase_enum = (
        PromotionPhaseType.MAIN_LOOP
        if phase == "Main-Loop"
        else PromotionPhaseType.EPILOGUE
    )
    operation_enum = (
        PromotionOperationType.FULL
        if operation == "Full"
        else PromotionOperationType.PARTIAL
    )
    layout_enum = Layout.ROW_MAJOR if layout == "Row-Major" else Layout.COLUMN_MAJOR

    def __init__(self):
        super(cls, self).__init__()
        self.consumer_generator_ = ConsumerGenerator(
            c_quant=quant,
            c_layout=layout_enum,
            template_type=TemplateType(phase_enum, operation_enum),
        )

    @property
    def consumer_generator(self):
        return self.consumer_generator_

    def name(self):
        return f"{phase} & {operation} & {'High-Precision' if not quant else 'Quantization'} C & {layout} C"

    condition = Condition().EQ(K_AB_Scale_Type, K_C_Scale_Type)

    if phase_enum == PromotionPhaseType.MAIN_LOOP:
        condition = condition.DIVISIBLE_BY(K_K, K_SK).GT(K_K, K_SK)
    else:
        condition = condition.EQ(K_K, K_SK)

    condition = condition.DIVISIBLE_BY(K_M, K_SM).DIVISIBLE_BY(K_N, K_SN)
    if operation_enum == PromotionOperationType.PARTIAL:
        condition = condition.GT(K_M, K_SM).GT(K_N, K_SN)
    else:
        condition = condition.GE(K_M, K_SM).GE(K_N, K_SN)

    if quant:
        condition = condition.IN(K_C_Type, [ScalarDType.FP8_E5M2, ScalarDType.FP8_E4M3])
    else:
        condition = condition.IN(K_C_Type, [ScalarDType.FP16, ScalarDType.BF16])

    def match_condition(self):
        return condition.IN(K_C_Layout, [layout_enum])

    filters = [
        SMemSizeFilter(),
        NumConsumerX64EqualBMFilter(),
        ClusterFilter(),
        BMNKTileFilter(),
    ]

    if operation_enum == PromotionOperationType.FULL:
        filters.append(FTypeFilter())

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


phases = ["Main-Loop", "Epilogue"]
operations = ["Full", "Partial"]
quants = [False, True]  # H/Q
layouts = ["Row-Major", "Column-Major"]

for phase, op, quant, layout in itertools.product(phases, operations, quants, layouts):
    cls = create_template_class(phase, op, quant, layout)
    register_template(cls)
