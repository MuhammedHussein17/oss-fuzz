#!/usr/bin/python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generate seed corpora for ONNX fuzz targets.
 
Usage (matches build.sh):
    python3 make_seed_corpus.py \
        $OUT/fuzz_version_converter_seed_corpus.zip \
        $OUT/fuzz_parser_seed_corpus.zip \
        $OUT/fuzz_shape_inference_seed_corpus.zip
"""
from __future__ import annotations
 
import sys
import zipfile
from onnx import TensorProto, helper
 
 
# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
 
def _make_model(op_type: str, opset_version: int, inputs: list[str], attrs=None) -> bytes:
    if attrs is None:
        attrs = {}
    graph_inputs = []
    if "X" in inputs:
        graph_inputs.append(helper.make_tensor_value_info("X", TensorProto.FLOAT, [1]))
    if "scales" in inputs:
        graph_inputs.append(
            helper.make_tensor_value_info("scales", TensorProto.FLOAT, [1])
        )
    graph_outputs = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])]
    node = helper.make_node(op_type, inputs, ["Y"], **attrs)
    graph = helper.make_graph([node], f"{op_type.lower()}-seed", graph_inputs, graph_outputs)
    model = helper.make_model(
        graph,
        producer_name="oss-fuzz",
        opset_imports=[helper.make_opsetid("", opset_version)],
    )
    return model.SerializeToString()
 
 
def _write_zip(path: str, entries: dict[str, bytes | str]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            if isinstance(data, str):
                data = data.encode()
            zf.writestr(name, data)
 
 
# ---------------------------------------------------------------------------
# fuzz_version_converter seeds  (unchanged)
# ---------------------------------------------------------------------------
 
def _version_converter_seeds() -> dict[str, bytes]:
    return {
        "cast_9_missing_input.onnx":        _make_model("Cast", 9, [], {"to": TensorProto.FLOAT}),
        "softmax_12_missing_input.onnx":    _make_model("Softmax", 12, []),
        "softmax_13_missing_input.onnx":    _make_model("Softmax", 13, []),
        "upsample_6_missing_input.onnx":    _make_model("Upsample", 6, []),
        "upsample_9_missing_scales.onnx":   _make_model("Upsample", 9, ["X"]),
        "upsample_9_valid.onnx":            _make_model("Upsample", 9, ["X", "scales"]),
    }
 
 
# ---------------------------------------------------------------------------
# fuzz_parser seeds  (unchanged)
# ---------------------------------------------------------------------------
 
_PARSER_SEEDS: dict[str, str] = {
    "basic_matmul_softmax.txt": """\
<
  ir_version: 7,
  opset_import: ["" : 10]
>
agraph (float[N, 128] X, float[128, 10] W, float[10] B) => (float[N] C)
{
   T = MatMul(X, W)
   S = Add(T, B)
   C = Softmax(S)
}
""",
    "multi_opset.txt": """\
<
  ir_version: 7,
  opset_import: ["" : 10, "com.microsoft" : 1]
>
agraph (float[N, 128] X, float[128, 10] W, float[10] B) => (float[N] C)
{
   T = MatMul(X, W)
   S = Add(T, B)
   C = Softmax(S)
}
""",
    "model_with_metadata.txt": """\
<
  ir_version: 9,
  opset_import: ["" : 15],
  producer_name: "oss-fuzz-seed",
  producer_version: "1.0",
  model_version: 1,
  doc_string: "seed model for fuzz_parser"
>
agraph (float[N] x) => (float[N] y)
{
   y = Relu(x)
}
""",
    "function_with_attributes.txt": """\
<
  ir_version: 9,
  opset_import: ["" : 15, "custom_domain" : 1],
  producer_name: "oss-fuzz-seed",
  producer_version: "1.0",
  model_version: 1,
  doc_string: "model with local function"
>
agraph (float[N] x) => (float[N] out)
{
   out = custom_domain.Selu<alpha=2.0, gamma=3.0>(x)
}
<
  domain: "custom_domain",
  opset_import: ["" : 15],
  doc_string: "custom Selu function"
>
Selu
<alpha: float=1.6732631921768188, gamma: float=1.0507010221481323>
(X) => (C)
{
    constant_alpha = Constant<value_float: float=@alpha>()
    constant_gamma = Constant<value_float: float=@gamma>()
    alpha_x = CastLike(constant_alpha, X)
    gamma_x = CastLike(constant_gamma, X)
    exp_x = Exp(X)
    alpha_x_exp_x = Mul(alpha_x, exp_x)
    alpha_x_exp_x_ = Sub(alpha_x_exp_x, alpha_x)
    neg = Mul(gamma_x, alpha_x_exp_x_)
    pos = Mul(gamma_x, X)
    _zero = Constant<value_float=0.0>()
    zero = CastLike(_zero, X)
    less_eq = LessOrEqual(X, zero)
    C = Where(less_eq, neg, pos)
}
""",
    "cast_with_initializer.txt": """\
<
  ir_version: 10,
  opset_import: ["" : 19]
>
agraph (float[N] X) => (int64[N] C)
<
  int64[1] weight = {0}
>
{
   C = Cast<to=7>(X)
}
""",
    "float_special_values.txt": """\
<
  ir_version: 8,
  opset_import: ["" : 18]
>
agraph (float[1] X) => (float[1] Y)
{
    pos_inf = Constant<value_float=inf>()
    neg_inf = Constant<value_float=-inf>()
    not_a_num = Constant<value_float=nan>()
    Y = Add(X, pos_inf)
}
""",
}
 
 
# ---------------------------------------------------------------------------
# fuzz_shape_inference seeds  (NEW)
# ---------------------------------------------------------------------------
 
def _si_linear() -> bytes:
    """Linear chain: Relu -> Sigmoid. Exercises basic unary shape pass-through."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
    graph = helper.make_graph(
        [helper.make_node("Relu", ["X"], ["T"]),
         helper.make_node("Sigmoid", ["T"], ["Y"])],
        "linear", [X], [Y],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 15)]
    ).SerializeToString()
 
 
def _si_concat() -> bytes:
    """Concat on axis 0. Exercises shape-data propagation for variadic inputs."""
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 4])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [3, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [5, 4])
    graph = helper.make_graph(
        [helper.make_node("Concat", ["A", "B"], ["Y"], axis=0)],
        "concat", [A, B], [Y],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 15)]
    ).SerializeToString()
 
 
def _si_matmul() -> bytes:
    """MatMul [4,8] x [8,2] -> [4,2]. Exercises 2-D shape propagation."""
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [4, 8])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [8, 2])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 2])
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["A", "B"], ["Y"])],
        "matmul", [A, B], [Y],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 15)]
    ).SerializeToString()
 
 
def _si_reshape() -> bytes:
    """Reshape driven by a Constant shape tensor. Exercises shape-data propagation."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [8, 1])
    shape_tensor = helper.make_tensor("shape", TensorProto.INT64, [2], [8, 1])
    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["shape_out"], value=shape_tensor),
         helper.make_node("Reshape", ["X", "shape_out"], ["Y"])],
        "reshape", [X], [Y],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 15)]
    ).SerializeToString()
 
 
def _si_if() -> bytes:
    """If op with then/else branches. Exercises subgraph-recursive shape inference."""
    cond_t = helper.make_tensor("cv", TensorProto.BOOL, [], [True])
    cond_node = helper.make_node("Constant", [], ["cond"], value=cond_t)
 
    def _branch(name: str, value: float):
        t = helper.make_tensor(name, TensorProto.FLOAT, [1], [value])
        return helper.make_graph(
            [helper.make_node("Constant", [], [name], value=t)],
            f"{name}_graph", [],
            [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1])],
        )
 
    if_node = helper.make_node(
        "If", ["cond"], ["result"],
        then_branch=_branch("then_out", 1.0),
        else_branch=_branch("else_out", 0.0),
    )
    graph = helper.make_graph(
        [cond_node, if_node], "if_graph", [],
        [helper.make_tensor_value_info("result", TensorProto.FLOAT, [1])],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 15)]
    ).SerializeToString()
 
 
def _si_loop() -> bytes:
    """Loop op with scan output. Exercises Loop-subgraph recursive descent."""
    trip_t = helper.make_tensor("trip", TensorProto.INT64, [], [3])
    trip_node = helper.make_node("Constant", [], ["trip_count"], value=trip_t)
    cond_t = helper.make_tensor("ci", TensorProto.BOOL, [], [True])
    cond_node = helper.make_node("Constant", [], ["cond"], value=cond_t)
 
    body_cond_t = helper.make_tensor("bc", TensorProto.BOOL, [], [True])
    scan_t = helper.make_tensor("sv", TensorProto.FLOAT, [1], [1.0])
    body = helper.make_graph(
        [helper.make_node("Constant", [], ["cond_out"], value=body_cond_t),
         helper.make_node("Constant", [], ["scan_out"], value=scan_t)],
        "loop_body",
        [helper.make_tensor_value_info("iter_count", TensorProto.INT64, []),
         helper.make_tensor_value_info("cond_in", TensorProto.BOOL, [])],
        [helper.make_tensor_value_info("cond_out", TensorProto.BOOL, []),
         helper.make_tensor_value_info("scan_out", TensorProto.FLOAT, [1])],
    )
    loop_node = helper.make_node(
        "Loop", ["trip_count", "cond"], ["loop_out"], body=body
    )
    graph = helper.make_graph(
        [trip_node, cond_node, loop_node], "loop_graph", [],
        [helper.make_tensor_value_info("loop_out", TensorProto.FLOAT, None)],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 15)]
    ).SerializeToString()
 
 
def _shape_inference_seeds() -> dict[str, bytes]:
    # The toggle byte expected by fuzz_shape_inference.TestOneInput:
    #   bit 0x04 = use_structured path  (0 = raw bytes path)
    # All seeds here are serialized ModelProtos fed via the raw-bytes path,
    # so the trailing toggle byte is 0x00 (raw, non-strict, no check_type).
    toggle = bytes([0x00])
    return {
        "linear_relu_sigmoid.onnx":  _si_linear()  + toggle,
        "concat_axis0.onnx":         _si_concat()  + toggle,
        "matmul_4x8_8x2.onnx":      _si_matmul()  + toggle,
        "reshape_2x4_to_8x1.onnx":  _si_reshape() + toggle,
        "if_then_else.onnx":         _si_if()      + toggle,
        "loop_scan_output.onnx":     _si_loop()    + toggle,
    }
 
 
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
 
def main() -> int:
    if len(sys.argv) != 4:
        print(
            "Usage: make_seed_corpus.py "
            "<version_converter.zip> <parser.zip> <shape_inference.zip>",
            file=sys.stderr,
        )
        return 1
 
    version_converter_out, parser_out, shape_inference_out = sys.argv[1:4]
 
    _write_zip(version_converter_out, _version_converter_seeds())
    _write_zip(parser_out, _PARSER_SEEDS)
    _write_zip(shape_inference_out, _shape_inference_seeds())
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())
