#pragma once
// ─── OnnxExporter ─────────────────────────────────────────────────────────────
//  Exports a trained NeuralStudio MLP as a standard ONNX model (opset 13) so
//  it can be used outside the app — e.g. in Python with onnxruntime:
//
//      import onnxruntime as ort, numpy as np
//      sess = ort.InferenceSession("model.onnx")
//      x = np.array([[...raw feature values...]], dtype=np.float32)
//      y = sess.run(None, {"input": x})[0]
//
//  The exported graph is fully self-contained:
//    • input  "input"  — shape [batch, inputCount], RAW (un-normalized) values;
//      the graph embeds the min/range normalization as Sub + Div nodes, so the
//      Python side needs no preprocessing.
//    • per hidden/output layer: Gemm (transB=1, weights stored [out, in]) +
//      the layer's activation (Relu / Sigmoid / Tanh; Linear = no node).
//    • multiclass models end with a Softmax node, so "output" holds class
//      probabilities; binary ends in the sigmoid output layer; regression
//      outputs the raw value.
//
//  Implementation note: ONNX files are protobuf-encoded ModelProto messages.
//  To avoid adding a protobuf dependency, this writer hand-encodes the tiny
//  subset of the protobuf wire format that ONNX needs (varints and
//  length-delimited fields). The exact encoding was validated against
//  onnxruntime (multiclass / binary / regression / batched inputs).
// ─────────────────────────────────────────────────────────────────────────────
#include "core/nn/NeuralNetwork.h"
#include <QString>

namespace NeuralStudio {

class OnnxExporter {
public:
    // Throws std::runtime_error on failure (unbuilt net, I/O error).
    static void save(const NeuralNetwork& net, const QString& path);

    static QString fileFilter() { return "ONNX Model (*.onnx)"; }
};

} // namespace NeuralStudio
