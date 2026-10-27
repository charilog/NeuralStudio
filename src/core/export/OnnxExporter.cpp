#include "OnnxExporter.h"
#include "core/nn/Layer.h"
#include <QFile>
#include <QByteArray>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace NeuralStudio {
namespace {

// ─── Minimal protobuf wire-format primitives ─────────────────────────────────
//  wire type 0 = varint, wire type 2 = length-delimited. That is all ONNX's
//  ModelProto needs for this graph. Field keys are (field_number << 3) | wire.

void putVarint(QByteArray& out, quint64 n) {
    do {
        quint8 b = n & 0x7F;
        n >>= 7;
        if (n) b |= 0x80;
        out.append(static_cast<char>(b));
    } while (n);
}
void putKey    (QByteArray& out, int field, int wire) { putVarint(out, (quint64(field) << 3) | wire); }
void fVarint   (QByteArray& out, int field, quint64 v){ putKey(out, field, 0); putVarint(out, v); }
void fBytes    (QByteArray& out, int field, const QByteArray& b) {
    putKey(out, field, 2); putVarint(out, quint64(b.size())); out.append(b);
}
void fStr      (QByteArray& out, int field, const QString& s) { fBytes(out, field, s.toUtf8()); }

// ─── ONNX message builders (mirroring the verified Python prototype) ──────────

// TensorProto: dims=1, data_type=2 (FLOAT=1), float raw_data=9, name=8
QByteArray tensorF32(const QString& name, const std::vector<qint64>& dims,
                     const std::vector<float>& values) {
    QByteArray t;
    for (qint64 d : dims) fVarint(t, 1, quint64(d));
    fVarint(t, 2, 1);                 // data_type = FLOAT
    fStr   (t, 8, name);
    QByteArray raw(reinterpret_cast<const char*>(values.data()),
                   int(values.size() * sizeof(float)));   // little-endian on all targets
    fBytes (t, 9, raw);
    return t;
}

// TensorShapeProto.Dimension: dim_value=1 (varint) OR dim_param=2 (string)
QByteArray dimValue(qint64 v)          { QByteArray d; fVarint(d, 1, quint64(v)); return d; }
QByteArray dimParam(const QString& s)  { QByteArray d; fStr(d, 2, s); return d; }

// ValueInfoProto: name=1, type=2 → TypeProto{tensor_type=1{elem_type=1, shape=2{dim=1...}}}
QByteArray valueInfo(const QString& name, const QByteArray& dim0, qint64 dim1) {
    QByteArray shape;
    fBytes(shape, 1, dim0);
    fBytes(shape, 1, dimValue(dim1));
    QByteArray tensorType;
    fVarint(tensorType, 1, 1);        // elem_type = FLOAT
    fBytes (tensorType, 2, shape);
    QByteArray typeProto;
    fBytes(typeProto, 1, tensorType);
    QByteArray v;
    fStr  (v, 1, name);
    fBytes(v, 2, typeProto);
    return v;
}

// AttributeProto (INT): name=1, i=3, type=20 (INT=2)
QByteArray attrInt(const QString& name, qint64 v) {
    QByteArray a;
    fStr   (a, 1, name);
    fVarint(a, 3, quint64(v));
    fVarint(a, 20, 2);
    return a;
}

// NodeProto: input=1*, output=2*, name=3, op_type=4, attribute=5*
QByteArray node(const QString& op, const QStringList& inputs,
                const QStringList& outputs, const QString& name,
                const QByteArray& attr = {}) {
    QByteArray n;
    for (const QString& i : inputs)  fStr(n, 1, i);
    for (const QString& o : outputs) fStr(n, 2, o);
    fStr(n, 3, name);
    fStr(n, 4, op);
    if (!attr.isEmpty()) fBytes(n, 5, attr);
    return n;
}

} // namespace

// ─── save ─────────────────────────────────────────────────────────────────────
void OnnxExporter::save(const NeuralNetwork& net, const QString& path) {
    if (!net.isBuilt())
        throw std::runtime_error("No trained network to export.");

    auto& layers = const_cast<NeuralNetwork&>(net).layers();
    const int inputCount = net.inputSize();

    QVector<QByteArray> nodes;
    QVector<QByteArray> inits;

    // Embedded input normalization: x_norm = (x - min) / range — identical to
    // NeuralNetwork::normalize(), so the ONNX consumer feeds RAW values.
    {
        std::vector<float> mn(net.normMin().begin(),   net.normMin().end());
        std::vector<float> rg(net.normRange().begin(), net.normRange().end());
        if (int(mn.size()) != inputCount || int(rg.size()) != inputCount)
            throw std::runtime_error("Normalization stats missing or wrong size.");
        inits << tensorF32("norm_min",   {inputCount}, mn);
        inits << tensorF32("norm_range", {inputCount}, rg);
        nodes << node("Sub", {"input", "norm_min"},   {"x_sub"},  "normalize_sub");
        nodes << node("Div", {"x_sub", "norm_range"}, {"x_norm"}, "normalize_div");
    }

    // Per-layer Gemm (+ activation). Weights are stored [out][in], which is
    // exactly Gemm's B operand with transB=1 (row-major flatten).
    QString prev = "x_norm";
    for (int li = 0; li < int(layers.size()); ++li) {
        const Layer& L = layers[li];
        const int nOut = L.outputSize(), nIn = L.inputSize();

        std::vector<float> wFlat; wFlat.reserve(size_t(nOut) * nIn);
        for (int o = 0; o < nOut; ++o)
            for (int i = 0; i < nIn; ++i)
                wFlat.push_back(float(L.weights[o][i]));
        std::vector<float> bFlat(L.biases.begin(), L.biases.end());

        const QString wName = QString("W%1").arg(li);
        const QString bName = QString("B%1").arg(li);
        inits << tensorF32(wName, {nOut, nIn}, wFlat);
        inits << tensorF32(bName, {nOut},      bFlat);

        const QString gemmOut = QString("g%1").arg(li);
        nodes << node("Gemm", {prev, wName, bName}, {gemmOut},
                      QString("gemm%1").arg(li), attrInt("transB", 1));

        QString actOp;
        switch (L.activation()) {
            case Activation::ReLU:    actOp = "Relu";    break;
            case Activation::Sigmoid: actOp = "Sigmoid"; break;
            case Activation::Tanh:    actOp = "Tanh";    break;
            case Activation::Linear:  break;   // no node needed
        }
        if (!actOp.isEmpty()) {
            const QString actOut = QString("a%1").arg(li);
            nodes << node(actOp, {gemmOut}, {actOut}, QString("act%1").arg(li));
            prev = actOut;
        } else {
            prev = gemmOut;
        }
    }

    // Multiclass ends with Softmax so "output" carries probabilities; other
    // tasks route the last tensor to "output" through an Identity node.
    if (net.taskType() == TaskType::MultiClassClassification)
        nodes << node("Softmax", {prev}, {"output"}, "softmax", attrInt("axis", 1));
    else if (net.taskType() == TaskType::Regression && net.hasOutputScaling()) {
        // regression output scaling: output = shift + scale * (linear output layer)
        const int nOut = net.outputSize();
        std::vector<float> sh(nOut), sc(nOut);
        for (int o = 0; o < nOut; ++o) { sh[o] = float(net.outShift(o)); sc[o] = float(net.outScale(o)); }
        inits << tensorF32("out_scale", {nOut}, sc);
        inits << tensorF32("out_shift", {nOut}, sh);
        nodes << node("Mul", {prev, "out_scale"}, {"y_scaled"}, "out_mul");
        nodes << node("Add", {"y_scaled", "out_shift"}, {"output"}, "out_add");
    } else
        nodes << node("Identity", {prev}, {"output"}, "out_identity");

    // GraphProto: node=1*, name=2, initializer=5*, input=11*, output=12*
    QByteArray g;
    for (const auto& n : nodes) fBytes(g, 1, n);
    fStr(g, 2, "neuralstudio_mlp");
    for (const auto& t : inits) fBytes(g, 5, t);
    fBytes(g, 11, valueInfo("input",  dimParam("batch"), inputCount));
    fBytes(g, 12, valueInfo("output", dimParam("batch"), net.outputSize()));

    // ModelProto: ir_version=1, producer_name=2, graph=7, opset_import=8
    QByteArray opset;
    fStr   (opset, 1, "");   // default ONNX domain
    fVarint(opset, 2, 13);   // opset version 13
    QByteArray m;
    fVarint(m, 1, 8);        // ir_version 8
    fStr   (m, 2, "NeuralStudio");
    fBytes (m, 7, g);
    fBytes (m, 8, opset);

    QFile f(path);
    if (!f.open(QIODevice::WriteOnly))
        throw std::runtime_error(("Cannot write: " + path).toStdString());
    f.write(m);
}

} // namespace NeuralStudio
