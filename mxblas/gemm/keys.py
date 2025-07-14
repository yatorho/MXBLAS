import os
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Self,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    override,
)

from mxblas.project.const import DEBUG_FLAG, PRINT_CONDITIONS_FLAG, PRINT_MATCHING_FLAG

from .descriptor import Layout, ScalarDType, scalar_type_bytes, value_dtype_to_cpp_type


class KeyTSingletonMeta(type):
    _instances: Dict[type, "Key_T"] = {}

    def __call__(cls, *args, **kwargs):
        if not issubclass(cls, Key_T):
            raise TypeError(f"{cls.__name__} is not a subclass of Key_T")
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class Expr:
    def evaluate(self, context: Dict[str, Any]) -> Any:
        raise NotImplementedError


Constable = Union[int, float, str]


class Const(Expr):
    def __init__(self, value: Constable):
        self.value = value

    def evaluate(self, context: Dict[str, Any]) -> Any:
        return self.value

    def __repr__(self):
        return repr(self.value)


class ExprMixin(Expr):
    def __add__(self, other):
        return BinaryExpr(self, BinaryExpr.Operator.ADD, other)

    def __radd__(self, other):
        return BinaryExpr(other, BinaryExpr.Operator.ADD, self)

    def __sub__(self, other):
        return BinaryExpr(self, BinaryExpr.Operator.SUB, other)

    def __rsub__(self, other):
        return BinaryExpr(other, BinaryExpr.Operator.SUB, self)

    def __mul__(self, other):
        return BinaryExpr(self, BinaryExpr.Operator.MUL, other)

    def __rmul__(self, other):
        return BinaryExpr(other, BinaryExpr.Operator.MUL, self)

    def __truediv__(self, other):
        return BinaryExpr(self, BinaryExpr.Operator.DIV, other)

    def __rtruediv__(self, other):
        return BinaryExpr(other, BinaryExpr.Operator.DIV, self)

    def __mod__(self, other):
        return BinaryExpr(self, BinaryExpr.Operator.MOD, other)

    def __rmod__(self, other):
        return BinaryExpr(other, BinaryExpr.Operator.MOD, self)

    def __floordiv__(self, other):
        return BinaryExpr(self, BinaryExpr.Operator.FLOOR_DIV, other)

    def __rfloordiv__(self, other):
        return BinaryExpr(other, BinaryExpr.Operator.FLOOR_DIV, self)


class BinaryExpr(ExprMixin, Expr):

    class Operator(Enum):
        ADD = "+"
        SUB = "-"
        MUL = "*"
        DIV = "/"
        MOD = "%"
        FLOOR_DIV = "//"

    def __init__(
        self, left: Union[Expr, Constable], op: Operator, right: Union[Expr, Constable]
    ):
        self.left = left if isinstance(left, Expr) else Const(left)
        self.op = op
        self.right = right if isinstance(right, Expr) else Const(right)

    def evaluate(self, context: Dict[str, Any]) -> Any:
        lval = self.left.evaluate(context)
        rval = self.right.evaluate(context)
        if self.op == BinaryExpr.Operator.ADD:
            return lval + rval
        if self.op == BinaryExpr.Operator.SUB:
            return lval - rval
        if self.op == BinaryExpr.Operator.MUL:
            return lval * rval
        if self.op == BinaryExpr.Operator.DIV:
            return lval / rval
        if self.op == BinaryExpr.Operator.MOD:
            return lval % rval
        if self.op == BinaryExpr.Operator.FLOOR_DIV:
            return lval // rval
        raise NotImplementedError(f"Operator {self.op} not supported")

    def __repr__(self):
        return f"({self.left} {self.op.value} {self.right})"


class IfExpr(ExprMixin, Expr):
    def __init__(
        self,
        condition: "Condition",
        then_expr: Union[Expr, Constable],
        else_expr: Union[Expr, Constable],
    ):
        self.condition = condition
        self.then_expr = then_expr if isinstance(then_expr, Expr) else Const(then_expr)
        self.else_expr = else_expr if isinstance(else_expr, Expr) else Const(else_expr)

    def evaluate(self, context: Dict[str, Any]) -> Any:
        if self.condition.evaluate(context):
            return self.then_expr.evaluate(context)
        else:
            return self.else_expr.evaluate(context)

    def __repr__(self):
        return f"(if {self.condition} then {self.then_expr} else {self.else_expr})"


class Key_T(ExprMixin, metaclass=KeyTSingletonMeta):
    name: str
    value_type: type

    def evaluate(self, context: Dict[str, Any]) -> Any:
        value = context[self.name]
        if type(value) is not self.value_type:
            raise TypeError(
                f"Expected type {self.value_type} for key '{self.name}', "
                f"but got {type(value)} with value {value}"
            )
        return value

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return str(self)


def str_to_keyT(key_s: str) -> Key_T:
    key_t_s = key_s + "_T"

    matched = list(
        filter(
            lambda class_instance: key_t_s == class_instance[0].__name__,
            KeyTSingletonMeta._instances.items(),
        )
    )
    if len(matched) != 1:
        raise ValueError(
            f"Key type {key_s} not found or ambiguous in KeyTSingletonMeta"
        )

    return matched[0][1]


def collect_all_pa_keys() -> List[Key_T]:
    instance = KeyTSingletonMeta._instances
    return list(key_t for key_t in instance.values() if isinstance(key_t, PaKey_T))


def collect_all_pr_keys() -> List[Key_T]:
    instance = KeyTSingletonMeta._instances
    return list(key_t for key_t in instance.values() if isinstance(key_t, PrKey_T))


def get_all_keys() -> List[Key_T]:
    return collect_all_pa_keys() + collect_all_pr_keys()


class PaKey_T(Key_T):
    pass


class K_M_T(PaKey_T):
    name = "K_M"
    value_type = int


class K_N_T(PaKey_T):
    name = "K_N"
    value_type = int


class K_K_T(PaKey_T):
    name = "K_K"
    value_type = int


class K_SM_T(PaKey_T):
    name = "K_SM"
    value_type = int


class K_SN_T(PaKey_T):
    name = "K_SN"
    value_type = int


class K_SK_T(PaKey_T):
    name = "K_SK"
    value_type = int


class K_Quant_T(PaKey_T):
    name = "K_Quant"
    value_type = bool


class K_QM_T(PaKey_T):
    name = "K_QM"
    value_type = int


class K_QN_T(PaKey_T):
    name = "K_QN"
    value_type = int


class K_A_Layout_T(PaKey_T):
    name = "K_A_Layout"
    value_type = Layout


class K_B_Layout_T(PaKey_T):
    name = "K_B_Layout"
    value_type = Layout


class K_C_Layout_T(PaKey_T):
    name = "K_C_Layout"
    value_type = Layout


class K_AS_Layout_T(PaKey_T):
    name = "K_AS_Layout"
    value_type = Layout


class K_BS_Layout_T(PaKey_T):
    name = "K_BS_Layout"
    value_type = Layout


class K_CS_Layout_T(PaKey_T):
    name = "K_CS_Layout"
    value_type = Layout


class K_AB_Type_T(PaKey_T):
    name = "K_AB_Type"
    value_type = ScalarDType


class K_C_Type_T(PaKey_T):
    name = "K_C_Type"
    value_type = ScalarDType


class K_AB_Scale_Type_T(PaKey_T):
    name = "K_AB_Scale_Type"
    value_type = ScalarDType


class K_C_Scale_Type_T(PaKey_T):
    name = "K_C_Scale_Type"
    value_type = ScalarDType


K_M = K_M_T()
K_N = K_N_T()
K_K = K_K_T()
K_SM = K_SM_T()
K_SN = K_SN_T()
K_SK = K_SK_T()
K_Quant = K_Quant_T()
K_QM = K_QM_T()
K_QN = K_QN_T()
K_A_Layout = K_A_Layout_T()
K_B_Layout = K_B_Layout_T()
K_C_Layout = K_C_Layout_T()
K_AS_Layout = K_AS_Layout_T()
K_BS_Layout = K_BS_Layout_T()
K_CS_Layout = K_CS_Layout_T()
K_AB_Type = K_AB_Type_T()
K_C_Type = K_C_Type_T()
K_AB_Scale_Type = K_AB_Scale_Type_T()
K_C_Scale_Type = K_C_Scale_Type_T()


class PrKey_T(Key_T):
    pass


class SMemSwizzleBits(Enum):
    DISABLE = "DISABLE"
    B32 = "B32"
    B64 = "B64"
    B128 = "B128"


swizzle_stride_bytes = {
    SMemSwizzleBits.B32: 32,
    SMemSwizzleBits.B64: 64,
    SMemSwizzleBits.B128: 128,
}


def swizzle_stride(swizzle: SMemSwizzleBits, dtype: ScalarDType):
    swizzle_bytes = swizzle_stride_bytes[swizzle]
    bytes_per_element = scalar_type_bytes[dtype]
    return swizzle_bytes // bytes_per_element


class CTASwizzleDim(Enum):
    ROW_LEAD = "ROW_LEAD"
    COL_LEAD = "COL_LEAD"


class K_BM_T(PrKey_T):
    name = "K_BM"
    value_type = int


class K_BN_T(PrKey_T):
    name = "K_BN"
    value_type = int


class K_BK_T(PrKey_T):
    name = "K_BK"
    value_type = int


class K_Num_SMs_T(PrKey_T):
    name = "K_Num_SMs"
    value_type = int


class K_Num_Threads_T(PrKey_T):
    name = "K_Num_Threads"
    value_type = int


class K_Num_Stages_T(PrKey_T):
    name = "K_Num_Stages"
    value_type = int


class K_Cluster_M_T(PrKey_T):
    name = "K_Cluster_M"
    value_type = int


class K_Cluster_N_T(PrKey_T):
    name = "K_Cluster_N"
    value_type = int


class K_WGMMA_M_T(PrKey_T):
    name = "K_WGMMA_M"
    value_type = int


class K_WGMMA_N_T(PrKey_T):
    name = "K_WGMMA_N"
    value_type = int


class K_WGMMA_K_T(PrKey_T):
    name = "K_WGMMA_K"
    value_type = int


class K_AB_SMem_Swizzle_T(PrKey_T):
    name = "K_AB_SMem_Swizzle"
    value_type = SMemSwizzleBits


class K_C_SMem_Swizzle_T(PrKey_T):
    name = "K_C_SMem_Swizzle"
    value_type = SMemSwizzleBits


class K_Num_TMA_Registers_T(PrKey_T):
    name = "K_Num_TMA_Registers"
    value_type = int


class K_Num_Math_Registers_T(PrKey_T):
    name = "K_Num_Math_Registers"
    value_type = int


class K_CTA_Swizzle_Lead_Dim_T(PrKey_T):
    name = "K_CTA_Swizzle_Lead_Dim"
    value_type = CTASwizzleDim


class K_CTA_Swizzle_Lead_Size_T(PrKey_T):
    name = "K_CTA_Swizzle_Lead_Size"
    value_type = int


K_BM = K_BM_T()
K_BN = K_BN_T()
K_BK = K_BK_T()
K_Num_SMs = K_Num_SMs_T()
K_Num_Threads = K_Num_Threads_T()
K_Num_Stages = K_Num_Stages_T()
K_Cluster_M = K_Cluster_M_T()
K_Cluster_N = K_Cluster_N_T()
K_WGMMA_M = K_WGMMA_M_T()
K_WGMMA_N = K_WGMMA_N_T()
K_WGMMA_K = K_WGMMA_K_T()
K_AB_SMem_Swizzle = K_AB_SMem_Swizzle_T()
K_C_SMem_Swizzle = K_C_SMem_Swizzle_T()
K_Num_TMA_Registers = K_Num_TMA_Registers_T()
K_Num_Math_Registers = K_Num_Math_Registers_T()
K_CTA_Swizzle_Lead_Dim = K_CTA_Swizzle_Lead_Dim_T()
K_CTA_Swizzle_Lead_Size = K_CTA_Swizzle_Lead_Size_T()


def key_type_to_cpp_Ttype(key_type: Key_T) -> str:
    """
    Converts a Key_T type to its corresponding C++ type.
    """
    type = key_type.value_type
    if type is int:
        cpp_type = "int32_t"
    elif type is bool:
        cpp_type = "bool"
    elif type is Layout:
        cpp_type = "Layout"  # Layout is an enum in C++
    elif type is ScalarDType:
        cpp_type = "typename"
    elif type is SMemSwizzleBits:
        cpp_type = "SMemSwizzleBits"
    elif type is CTASwizzleDim:
        cpp_type = "int32_t"  # CTASwizzleDim is represented as an int in C++
    else:
        raise ValueError(f"Unknown key type: {type}")

    return cpp_type


def key_value_to_cpp_Tvalue(key_type: Key_T, value: Any) -> str:
    type = key_type.value_type
    if type is int:
        cpp_value = str(value)
    elif type is bool:
        assert isinstance(
            value, bool
        ), f"Expected bool for {key_type.name}, got {type(value)}"
        cpp_value = "true" if value else "false"
    elif type is Layout:
        assert isinstance(
            value, Layout
        ), f"Expected Layout for {key_type.name}, got {type(value)}"
        cpp_value = "Layout::" + value.value
    elif type is ScalarDType:
        assert isinstance(
            value, ScalarDType
        ), f"Expected ScalarDType for {key_type.name}, got {type(value)}"
        cpp_value = value_dtype_to_cpp_type[value]
    elif type is SMemSwizzleBits:
        assert isinstance(
            value, SMemSwizzleBits
        ), f"Expected SMemSizzle for {key_type.name}, got {type(value)}"
        cpp_value = "SMemSwizzleBits::" + value.value
    elif type is CTASwizzleDim:
        assert isinstance(
            value, CTASwizzleDim
        ), f"Expected CTASwizzleDim for {key_type.name}, got {type(value)}"
        cpp_value = "0" if value == CTASwizzleDim.ROW_LEAD else "1"

    else:
        raise ValueError(f"Unknown key type: {type}")

    return cpp_value


# @lru_cache(maxsize=None)
# def concat_keys():
#     return PA_KEYS | BASE_PR_KEYS


# def check_key_value(kvs: Dict[str, Any]):
#     for key, value in kvs.items():
#         if key not in PA_KEYS and key not in BASE_PR_KEYS:
#             raise ValueError(f"Unknown key: {key}")
#         expected_type = kvt_to_value_type(
#             cast(KeyValueType, PA_KEYS.get(key, BASE_PR_KEYS.get(key)))
#         )
#         if not isinstance(value, expected_type):
#             raise TypeError(
#                 f"Expected type {expected_type} for key '{key}', "
#                 f"but got {type(value)} with value {value}"
#             )


class IfElseNode:
    def __init__(
        self, cond: "Condition", if_branch: "Condition", else_branch: "Condition"
    ):
        self.cond = cond
        self.if_branch = if_branch
        self.else_branch = else_branch

    def evaluate(self, context):
        if self.cond.evaluate(context):
            return self.if_branch.evaluate(context)
        else:
            return self.else_branch.evaluate(context)


class Condition:

    class Operator(Enum):
        GT = ">"
        GE = ">="
        LT = "<"
        LE = "<="
        EQ = "=="
        NE = "!="
        IN = "in"
        DIVISIBLE_BY = "divisible_by"

        def __call__(self, context: Dict[str, Any], args: List) -> Tuple[bool, str]:
            def get_value(context: Dict[str, Any], arg: Union[Key_T, Any]) -> Any:
                # if isinstance(arg, Key_T):
                if isinstance(arg, Expr):
                    return arg.evaluate(context)
                return arg

            left_val = get_value(context, args[0])
            right_val = None
            if self in {Condition.Operator.IN}:
                right_val = [get_value(context, a) for a in args[1:]]
            else:
                right_val = get_value(context, args[1])

            expr_str = f"{args[0]} {self.value} {args[1]}"
            msg = f"{expr_str} >>> {left_val} {self.value} {right_val}"

            if self == Condition.Operator.GT:
                return left_val > right_val, msg
            elif self == Condition.Operator.GE:
                return left_val >= right_val, msg
            elif self == Condition.Operator.LT:
                return left_val < right_val, msg
            elif self == Condition.Operator.LE:
                return left_val <= right_val, msg
            elif self == Condition.Operator.EQ:
                return left_val == right_val, msg
            elif self == Condition.Operator.NE:
                return left_val != right_val, msg
            elif self == Condition.Operator.IN:
                return left_val in right_val, msg
            elif self == Condition.Operator.DIVISIBLE_BY:
                if right_val == 0:
                    raise ZeroDivisionError("Divisor cannot be zero")
                return left_val % right_val == 0, msg

            else:
                raise NotImplementedError(f"Operator {self} is not implemented")

        def __repr__(self) -> str:
            return self.value

    def __init__(
        self,
        expressions: Optional[List[Tuple[Operator, List]]] = None,
        children: Optional[List["IfElseNode"]] = None,
    ):
        self.expressions = expressions or []
        self.children = children or []

    def _add_expression(self, op: Operator, args: List) -> Self:
        self.expressions.append((op, args))
        return self

    def GT(self, left: Union[Key_T, Any], right: Union[Key_T, Any]) -> Self:
        return self._add_expression(Condition.Operator.GT, [left, right])

    def GE(self, left: Union[Key_T, Any], right: Union[Key_T, Any]) -> Self:
        return self._add_expression(Condition.Operator.GE, [left, right])

    def LT(self, left: Union[Key_T, Any], right: Union[Key_T, Any]) -> Self:
        return self._add_expression(Condition.Operator.LT, [left, right])

    def LE(self, left: Union[Key_T, Any], right: Union[Key_T, Any]) -> Self:
        return self._add_expression(Condition.Operator.LE, [left, right])

    def EQ(self, left: Union[Key_T, Any], right: Union[Key_T, Any]) -> Self:
        return self._add_expression(Condition.Operator.EQ, [left, right])

    def NE(self, left: Union[Key_T, Any], right: Union[Key_T, Any]) -> Self:
        return self._add_expression(Condition.Operator.NE, [left, right])

    def IN(
        self, var: Union[Key_T, Any], values: List[Union[Key_T, Any]]
    ) -> "Condition":
        """
        Checks if the variable `var` is in the list of `values`.
        """
        if not isinstance(values, list):
            raise TypeError(f"Expected list for values, got {type(values)}")
        return self._add_expression(Condition.Operator.IN, [var, *values])

    def DIVISIBLE_BY(self, var: Union[Key_T, Any], divisor: Union[Key_T, Any]) -> Self:
        return self._add_expression(Condition.Operator.DIVISIBLE_BY, [var, divisor])

    def _evaluate_expressions(self, context: Dict[str, Any]) -> bool:
        for i, (op, args) in enumerate(self.expressions):
            try:
                result, msg = op(context, args)
                if not result:
                    if os.getenv(PRINT_CONDITIONS_FLAG, None):
                        print(f"The {i}th condition check failed: {msg}")
                    return False
            except (KeyError, TypeError):
                raise ValueError(
                    f"Condition check failed for context: {context} with expression: {op} {args}"
                )
        return True

    def evaluate(self, context: Dict[str, Any]) -> bool:
        if not self._evaluate_expressions(context):
            return False

        for child in self.children:
            if not child.evaluate(context):
                return False

        return True

    def If(self, cond: "Condition") -> "IfBuilder":
        return IfBuilder(self, cond)

    def __and__(self, other: "Condition") -> "Condition":
        """
        Combines two conditions with a logical AND.
        """
        if not isinstance(other, Condition):
            raise TypeError(f"Expected Condition, got {type(other)}")
        return Condition(self.expressions + other.expressions)

    def __repr__(self) -> str:
        return f"Condition({len(self.expressions)} rules)"


class IfBuilder(Condition):
    def __init__(self, parent: "Condition", cond: "Condition"):
        self.parent = parent
        self.if_cond = cond
        self.if_branch = Condition()
        self.else_branch = Condition()
        self._current = self.if_branch

    def Else(self):
        self._current = self.else_branch
        return self

    def build(self):
        node = IfElseNode(self.if_cond, self.if_branch, self.else_branch)
        self.parent.children.append(node)
        return self.parent

    # Override Condition methods to delegate to current branch
    @override
    def _add_expression(self, op, args):
        # Delegate all expressions to the current branch
        self._current._add_expression(op, args)
        return self

    @override
    def evaluate(self, context):
        raise NotImplementedError(
            "IfBuilder is a builder, call .build() to finalize it."
        )


def GT(left: Union[Key_T, Any], right: Union[Key_T, Any]) -> Condition:
    return Condition().GT(left, right)


def GE(left: Union[Key_T, Any], right: Union[Key_T, Any]) -> Condition:
    return Condition().GE(left, right)


def LT(left: Union[Key_T, Any], right: Union[Key_T, Any]) -> Condition:
    return Condition().LT(left, right)


def LE(left: Union[Key_T, Any], right: Union[Key_T, Any]) -> Condition:
    return Condition().LE(left, right)


def EQ(left: Union[Key_T, Any], right: Union[Key_T, Any]) -> Condition:
    return Condition().EQ(left, right)


def NE(left: Union[Key_T, Any], right: Union[Key_T, Any]) -> Condition:
    return Condition().NE(left, right)


def IN(var: Union[Key_T, Any], values: List[Union[Key_T, Any]]) -> Condition:
    return Condition().IN(var, values)


def DIVISIBLE_BY(var: Union[Key_T, Any], divisor: Union[Key_T, Any]) -> Condition:
    return Condition().DIVISIBLE_BY(var, divisor)
