#include "ModelSerializer.h"
#include "utils/Logger.h"

#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <stdexcept>

namespace NeuralStudio {

// ─── save ─────────────────────────────────────────────────────────────────────
void ModelSerializer::save(const NeuralNetwork& net, const QString& path) {
    QJsonObject root;
    root["version"]    = "1.1";
    root["task"]       = static_cast<int>(net.taskType());
    root["inputSize"]  = net.inputSize();
    root["outputSize"] = net.outputSize();
    root["outMin"]     = net.outputMin();
    root["outMax"]     = net.outputMax();

    // Class values
    QJsonArray classArr;
    for (double v : net.classValues()) classArr.append(v);
    root["classValues"] = classArr;

    // Normalization
    QJsonArray minArr, rangeArr;
    for (double v : net.normMin())   minArr  .append(v);
    for (double v : net.normRange()) rangeArr.append(v);
    root["normMin"]   = minArr;
    root["normRange"] = rangeArr;
    QJsonArray meanArr;   // train means: the value used for missing inputs
    for (size_t i = 0; i < net.normFill().size() && i < net.normMin().size(); ++i)
        meanArr.append(net.normMin()[i] + net.normFill()[i] * net.normRange()[i]);
    root["normMean"] = meanArr;

    // Layers
    QJsonArray layersArr;
    for (const auto& layer : const_cast<NeuralNetwork&>(net).layers()) {
        QJsonObject lObj;
        lObj["in"]         = layer.inputSize();
        lObj["out"]        = layer.outputSize();
        lObj["activation"] = static_cast<int>(layer.activation());

        QJsonArray biasArr;
        for (double b : layer.biases) biasArr.append(b);
        lObj["biases"] = biasArr;

        QJsonArray wArr;
        for (const auto& row : layer.weights) {
            QJsonArray rowArr;
            for (double w : row) rowArr.append(w);
            wArr.append(rowArr);
        }
        lObj["weights"] = wArr;
        layersArr.append(lObj);
    }
    root["layers"] = layersArr;
    if (net.hasOutputScaling()) {   // regression output scaling (absent = identity, older models)
        QJsonArray sh, sc;
        for (double v : net.outShifts()) sh.append(v);
        for (double v : net.outScales()) sc.append(v);
        root["outShift"] = sh;
        root["outScale"] = sc;
    }

    QFile f(path);
    if (!f.open(QIODevice::WriteOnly))
        throw std::runtime_error("Cannot write model file: " + path.toStdString());
    f.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
    NS_INFO << "Model saved to" << path;
}

// ─── load ─────────────────────────────────────────────────────────────────────
void ModelSerializer::load(NeuralNetwork& net, const QString& path) {
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly))
        throw std::runtime_error("Cannot open model file: " + path.toStdString());

    QJsonParseError err;
    QJsonDocument doc = QJsonDocument::fromJson(f.readAll(), &err);
    if (doc.isNull())
        throw std::runtime_error("Invalid model file: " + err.errorString().toStdString());

    QJsonObject root = doc.object();

    const TaskType task       = static_cast<TaskType>(root["task"].toInt());
    const int      inputSize  = root["inputSize"].toInt();
    const int      outputSize = root["outputSize"].toInt();

    // Rebuild layer structure
    QJsonArray layersArr = root["layers"].toArray();
    NetworkConfig cfg;
    cfg.task = task;
    // Hidden layers = all but last
    for (int i = 0; i < layersArr.size() - 1; ++i) {
        QJsonObject lo = layersArr[i].toObject();
        cfg.hiddenSizes.push_back(lo["out"].toInt());
        cfg.hiddenActivations.push_back(static_cast<Activation>(lo["activation"].toInt()));
    }
    if (layersArr.isEmpty() || inputSize <= 0 || outputSize <= 0)
        throw std::runtime_error("Invalid model file (no layers or sizes): " + path.toStdString());
    net.build(inputSize, outputSize, cfg);

    // Restore weights (sizes are checked: a damaged file used to write out of range)
    auto& layers = net.layers();
    if (static_cast<int>(layers.size()) != layersArr.size())
        throw std::runtime_error("Invalid model file (layer count): " + path.toStdString());
    for (int li = 0; li < layersArr.size(); ++li) {
        QJsonObject lo = layersArr[li].toObject();
        auto& layer = layers[li];
        const QJsonArray bChk = lo["biases"].toArray(), wChk = lo["weights"].toArray();
        bool okSize = bChk.size() == layer.outputSize() && wChk.size() == layer.outputSize();
        for (int o = 0; okSize && o < wChk.size(); ++o) okSize = wChk[o].toArray().size() == layer.inputSize();
        if (!okSize || static_cast<Activation>(lo["activation"].toInt()) != layer.activation())
            throw std::runtime_error("Invalid model file (layer " + std::to_string(li) + " does not match): " + path.toStdString());

        QJsonArray biasArr = lo["biases"].toArray();
        for (int b = 0; b < biasArr.size(); ++b)
            layer.biases[b] = biasArr[b].toDouble();

        QJsonArray wArr = lo["weights"].toArray();
        for (int o = 0; o < wArr.size(); ++o) {
            QJsonArray rowArr = wArr[o].toArray();
            for (int i = 0; i < rowArr.size(); ++i)
                layer.weights[o][i] = rowArr[i].toDouble();
        }
    }

    // Restore normalization
    QJsonArray minArr   = root["normMin"].toArray();
    QJsonArray rangeArr = root["normRange"].toArray();
    std::vector<ColumnStats> stats(minArr.size());
    for (int i = 0; i < minArr.size(); ++i) {
        stats[i].min = minArr[i].toDouble();
        double rng   = rangeArr[i].toDouble();
        stats[i].max = stats[i].min + rng;
        const QJsonArray meanArr = root["normMean"].toArray();
        stats[i].mean = i < meanArr.size() ? meanArr[i].toDouble() : stats[i].min + 0.5 * rng;
    }
    net.setNormalization(stats);

    // Class values
    std::vector<double> classVals;
    for (auto v : root["classValues"].toArray()) classVals.push_back(v.toDouble());
    if (!classVals.empty()) net.setClassValues(classVals);
    net.setOutputMapping(root["outMin"].toDouble(), root["outMax"].toDouble());
    if (root.contains("outScale")) {
        std::vector<double> sh, sc;
        for (auto v : root["outShift"].toArray()) sh.push_back(v.toDouble());
        for (auto v : root["outScale"].toArray()) sc.push_back(v.toDouble());
        net.setOutputScaling(sh, sc);
    }

    NS_INFO << "Model loaded from" << path << "—" << net.summary();
}

} // namespace NeuralStudio
