"""Defines an instruction for a tensor product."""

from typing import Tuple, List
from dataclasses import dataclass, field, replace
from cuda_mace.cuda.irreps import Irreps
import collections
import math

@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=True)
class Instruction:
    """Defines an instruction for a tensor product."""

    i_in1: int
    i_in2: int
    i_out: int
    connection_mode: str
    has_weight: bool
    path_weight: float
    weight_std: float
    first_input_multiplicity: int
    second_input_multiplicity: int
    output_multiplicity: int
    path_shape: Tuple[int, ...] = field(init=False)
    num_elements: int = field(init=False)

    def __post_init__(self):
        if self.connection_mode not in [
            "uvw",
            "uvu",
            "uvv",
            "uuw",
            "uuu",
            "uvuv",
            "uvu<v",
            "u<vw",
        ]:
            raise ValueError(
                f"Unsupported connection_mode {self.connection_mode} for instruction."
            )

        path_shape = {
            "uvw": (
                self.first_input_multiplicity,
                self.second_input_multiplicity,
                self.output_multiplicity,
            ),
            "uvu": (self.first_input_multiplicity, self.second_input_multiplicity),
            "uvv": (self.first_input_multiplicity, self.second_input_multiplicity),
            "uuw": (self.first_input_multiplicity, self.output_multiplicity),
            "uuu": (self.first_input_multiplicity,),
            "uvuv": (self.first_input_multiplicity, self.second_input_multiplicity),
            "uvu<v": (
                self.first_input_multiplicity
                * (self.second_input_multiplicity - 1)
                // 2,
            ),
            "u<vw": (
                self.first_input_multiplicity
                * (self.second_input_multiplicity - 1)
                // 2,
                self.output_multiplicity,
            ),
        }[self.connection_mode]
        super().__setattr__("path_shape", path_shape)

        num_elements = {
            "uvw": (self.first_input_multiplicity * self.second_input_multiplicity),
            "uvu": self.second_input_multiplicity,
            "uvv": self.first_input_multiplicity,
            "uuw": self.first_input_multiplicity,
            "uuu": 1,
            "uvuv": 1,
            "uvu<v": 1,
            "u<vw": self.first_input_multiplicity
            * (self.second_input_multiplicity - 1)
            // 2,
        }[self.connection_mode]
        super().__setattr__("num_elements", num_elements)

    def replace(self, **changes) -> "Instruction":
        return replace(self, **changes)

    def __repr__(self) -> str:
        return (
            "Instruction("
            + ", ".join(
                [
                    f"i={self.i_in1},{self.i_in2},{self.i_out}",
                    f"mode={self.connection_mode}",
                    f"has_weight={self.has_weight}",
                    f"path_weight={self.path_weight}",
                    f"weight_std={self.weight_std}",
                    f"mul={self.first_input_multiplicity},{self.second_input_multiplicity},{self.output_multiplicity}",
                    f"path_shape={self.path_shape}",
                    f"num_elements={self.num_elements}",
                ]
            )
            + ")"
        )


#taken from e3nn-jax to remove dependence on jax...
def _normalize_instruction_path_weights(
    instructions: List[Instruction],
    first_input_irreps: Irreps,
    second_input_irreps: Irreps,
    output_irreps: Irreps,
    first_input_variance: List[float],
    second_input_variance: List[float],
    output_variance: List[float],
    irrep_normalization: str,
    path_normalization_exponent: float,
    gradient_normalization_exponent: float,
) -> List[Instruction]:
    """Returns instructions with normalized path weights."""

    def var(instruction):
        return (
            first_input_variance[instruction.i_in1]
            * second_input_variance[instruction.i_in2]
            * instruction.num_elements
        )

    # Precompute normalization factors.
    path_normalization_sums = collections.defaultdict(lambda: 0.0)
    for instruction in instructions:
        path_normalization_sums[instruction.i_out] += var(instruction) ** (
            1.0 - path_normalization_exponent
        )

    path_normalization_factors = {
        instruction: var(instruction) ** path_normalization_exponent
        * path_normalization_sums[instruction.i_out]
        for instruction in instructions
    }

    def update(instruction: Instruction) -> float:
        """Computes normalized path weight for a single instructions, with precomputed path normalization factors."""

        if irrep_normalization not in ["component", "norm", "none"]:
            raise ValueError(f"Unsupported irrep normalization: {irrep_normalization}.")

        mul_ir_in1 = first_input_irreps[instruction.i_in1]
        mul_ir_in2 = second_input_irreps[instruction.i_in2]
        mul_ir_out = output_irreps[instruction.i_out]

        assert mul_ir_in1.ir.p * mul_ir_in2.ir.p == mul_ir_out.ir.p
        assert (
            abs(mul_ir_in1.ir.l - mul_ir_in2.ir.l)
            <= mul_ir_out.ir.l
            <= mul_ir_in1.ir.l + mul_ir_in2.ir.l
        )

        if irrep_normalization == "component":
            alpha = mul_ir_out.ir.dim
        if irrep_normalization == "norm":
            alpha = mul_ir_in1.ir.dim * mul_ir_in2.ir.dim
        if irrep_normalization == "none":
            alpha = 1

        x = path_normalization_factors[instruction]
        if x > 0.0:
            alpha /= x

        alpha *= output_variance[instruction.i_out]
        alpha *= instruction.path_weight

        if instruction.has_weight:
            return instruction.replace(
                path_weight=math.sqrt(alpha) ** gradient_normalization_exponent,
                weight_std=math.sqrt(alpha) ** (1.0 - gradient_normalization_exponent),
            )
        else:
            return instruction.replace(path_weight=math.sqrt(alpha))

    return [update(instruction) for instruction in instructions]