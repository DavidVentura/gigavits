"""Add a `durations` output (int64 [batch, phonemes], frames per input ID) to a Piper VITS ONNX export.

Usage: expose_durations.py <in.onnx> <out.onnx>
"""

import sys

import onnx
from onnx import TensorProto, helper, numpy_helper

import numpy as np

DURATIONS = "durations"
MIN_SQUEEZE_AXES_AS_INPUT_OPSET = 13


class GraphIndex:
    def __init__(self, graph: onnx.GraphProto):
        self.producer = {out: node for node in graph.node for out in node.output}
        self.consumers: dict[str, list[onnx.NodeProto]] = {}
        for node in graph.node:
            for name in node.input:
                self.consumers.setdefault(name, []).append(node)

    def consumer_ops(self, tensor: str) -> set[str]:
        return {node.op_type for node in self.consumers.get(tensor, [])}

    def is_exp_times_masks(self, tensor: str) -> bool:
        node = self.producer.get(tensor)
        while node is not None and node.op_type == "Mul":
            node = next(
                (self.producer[i] for i in node.input if i in self.producer and self.producer[i].op_type in ("Mul", "Exp")),
                None,
            )
        return node is not None and node.op_type == "Exp"


def find_w_ceil(graph: onnx.GraphProto) -> onnx.NodeProto:
    # w_ceil = ceil(exp(logw) * x_mask * length_scale) feeds both
    # y_lengths = clamp_min(sum(w_ceil), 1) and the alignment path via cumsum(w_ceil).
    index = GraphIndex(graph)
    candidates = [
        node
        for node in graph.node
        if node.op_type == "Ceil"
        and {"ReduceSum", "CumSum"} <= index.consumer_ops(node.output[0])
        and index.is_exp_times_masks(node.input[0])
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one w_ceil Ceil node, found {len(candidates)}: {[n.name for n in candidates]}"
        )
    return candidates[0]


def opset_version(model: onnx.ModelProto) -> int:
    return next(op.version for op in model.opset_import if op.domain in ("", "ai.onnx"))


def expose_durations(model: onnx.ModelProto) -> onnx.ModelProto:
    graph = model.graph
    if [o.name for o in graph.output] != ["output"]:
        raise RuntimeError(f"expected a single `output` graph output, got {[o.name for o in graph.output]}")
    if opset_version(model) < MIN_SQUEEZE_AXES_AS_INPUT_OPSET:
        raise RuntimeError(f"opset {opset_version(model)} < {MIN_SQUEEZE_AXES_AS_INPUT_OPSET} is not supported")
    token_input = next(i for i in graph.input if i.name == "input")
    batch_dim, phoneme_dim = token_input.type.tensor_type.shape.dim

    w_ceil = find_w_ceil(graph).output[0]
    axes = "durations/squeeze_axes"
    squeezed = "durations/squeezed"
    graph.initializer.append(numpy_helper.from_array(np.array([1], dtype=np.int64), axes))
    graph.node.extend(
        [
            # w_ceil is [batch, 1, phonemes]
            helper.make_node("Squeeze", [w_ceil, axes], [squeezed], name="durations/Squeeze"),
            helper.make_node("Cast", [squeezed], [DURATIONS], name="durations/Cast", to=TensorProto.INT64),
        ]
    )
    out = helper.make_tensor_value_info(DURATIONS, TensorProto.INT64, ["batch", "phonemes"])
    out.type.tensor_type.shape.dim[0].CopyFrom(batch_dim)
    out.type.tensor_type.shape.dim[1].CopyFrom(phoneme_dim)
    graph.output.append(out)
    return model


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    onnx.save(expose_durations(onnx.load(src)), dst)


if __name__ == "__main__":
    main()
