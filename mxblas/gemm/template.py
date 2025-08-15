from abc import ABC, abstractmethod
from typing import Iterable

from mxblas.gemm.filter import Filter
from mxblas.gemm.keys import (
    EQ,
    K_K,
    K_M,
    K_N,
    K_SK,
    K_SM,
    K_SN,
    Condition,
    K_A_Layout,
    K_AB_Scale_Type,
    K_AB_Type,
    K_B_Layout,
    K_C_Layout,
    K_C_Scale_Type,
    K_C_Type,
    K_Quant,
    Key_T,
)

from .descriptor import Layout, ScalarDType


class KernelTemplate(ABC):
    """
    A abstract class representing a kernel template for MX-GEMM.

    Templates means a predefined code contructure that can be tuned with CPP template instances.
    """

    @abstractmethod
    def name(self) -> str:
        """
        Returns the unique name of the kernel template."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def match_condition(self) -> Condition:
        """
        Checks if the given key-value pairs match the template's key-value ranges.
        This is used to determine if the template can be used for the given parameters.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def generate(self):
        """
        Returns the code of the kernel template."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def cpp_template_keys(self) -> Iterable[Key_T]:
        """
        Returns the list of keys used in the CPP template for this kernel template.
        This is used to generate the code with the correct keys.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def prune_rules(self) -> Iterable[Filter]:
        """
        Returns the list of rules that can be used to prune the pa x pr space."""
        raise NotImplementedError("Subclasses must implement this method.")


common_conditions = (
    Condition()
    .EQ(K_A_Layout, Layout.ROW_MAJOR)
    .EQ(K_B_Layout, Layout.COLUMN_MAJOR)
    .EQ(K_C_Layout, Layout.ROW_MAJOR)
    .IN(K_AB_Type, [ScalarDType.FP8_E4M3, ScalarDType.FP8_E5M2])
    .DIVISIBLE_BY(K_M, K_SM)
    .DIVISIBLE_BY(K_N, K_SN)
    .DIVISIBLE_BY(K_K, K_SK)
    .EQ(K_AB_Scale_Type, K_C_Scale_Type)
    .If(EQ(K_Quant, True))
    .IN(K_C_Type, [ScalarDType.FP8_E5M2, ScalarDType.FP8_E4M3])
    .Else()
    .IN(K_C_Type, [ScalarDType.FP16, ScalarDType.BF16])
    .build()
)
