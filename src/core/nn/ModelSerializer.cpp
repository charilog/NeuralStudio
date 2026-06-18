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
    root["version"]    = "1.0";
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
    net.build(inputSize, outputSize, cfg);

    // Restore weights
    auto& layers = net.layers();
    for (int li = 0; li < layersArr.size(); ++li) {
        QJsonObject lo = layersArr[li].toObject();
        auto& layer = layers[li];

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
    }
    net.setNormalization(stats);

    // Class values
    std::vector<double> classVals;
    for (auto v : root["classValues"].toArray()) classVals.push_back(v.toDouble());
    if (!classVals.empty()) net.setClassValues(classVals);
    net.setOutputMapping(root["outMin"].toDouble(), root["outMax"].toDouble());

    NS_INFO << "Model loaded from" << path << "—" << net.summary();
}

} // namespace NeuralStudio
