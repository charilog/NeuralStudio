#include "NeuralDesignerLoader.h"

#include "utils/Logger.h"
#include <QFile>
#include <QFileInfo>
#include <QTextStream>
#include <stdexcept>
#include <limits>

namespace NeuralStudio {

// ─── Metadata ────────────────────────────────────────────────────────────────
QStringList NeuralDesignerLoader::extensions() const {
    return {"train", "test", "data"};
}

QString NeuralDesignerLoader::formatName() const {
    return "NeuralStudio";
}

// ─── canLoad ─────────────────────────────────────────────────────────────────
//  Quick peek: file must start with two positive integers on separate lines.
bool NeuralDesignerLoader::canLoad(const QString& path) const {
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) return false;
    QTextStream in(&f);

    bool ok1 = false, ok2 = false;
    int n1 = in.readLine().trimmed().toInt(&ok1);
    int n2 = in.readLine().trimmed().toInt(&ok2);
    return ok1 && ok2 && n1 > 0 && n2 > 0;
}

// ─── load ────────────────────────────────────────────────────────────────────
std::unique_ptr<Dataset> NeuralDesignerLoader::load(const QString& path) {
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        throw std::runtime_error("Cannot open file: " + path.toStdString());

    QTextStream in(&file);
    bool ok = false;

    // ── Header: line 1 — input count ────────────────────────────────────────
    const int numInputs = in.readLine().trimmed().toInt(&ok);
    if (!ok || numInputs <= 0)
        throw std::runtime_error(
            "Neural Designer format: line 1 must be a positive integer (input count). "
            "File: " + path.toStdString());

    // ── Header: line 2 — declared sample count (hint only) ──────────────────
    const int declaredSamples = in.readLine().trimmed().toInt(&ok);
    if (!ok || declaredSamples <= 0)
        throw std::runtime_error(
            "Neural Designer format: line 2 must be a positive integer (sample count). "
            "File: " + path.toStdString());

    // ── Data rows ────────────────────────────────────────────────────────────
    std::vector<std::vector<double>> rawRows;
    rawRows.reserve(declaredSamples);
    int totalCols = -1;
    int lineNo    = 2;

    while (!in.atEnd()) {
        ++lineNo;
        const QString line = in.readLine().trimmed();
        if (line.isEmpty()) continue;

        const QStringList tokens = line.split(' ', Qt::SkipEmptyParts);
        if (tokens.isEmpty()) continue;

        // First data row determines total column count
        if (totalCols == -1) {
            totalCols = tokens.size();
            if (totalCols <= numInputs)
                throw std::runtime_error(
                    QString("Neural Designer format: data has %1 columns but header "
                            "declares %2 inputs — no room for outputs. Line %3.")
                    .arg(totalCols).arg(numInputs).arg(lineNo).toStdString());
        } else if (tokens.size() != totalCols) {
            throw std::runtime_error(
                QString("Neural Designer format: column count mismatch at line %1 "
                        "(expected %2, got %3).")
                .arg(lineNo).arg(totalCols).arg(tokens.size()).toStdString());
        }

        std::vector<double> row;
        row.reserve(totalCols);
        for (const QString& tok : tokens) {
            bool tokenOk = false;
            const double v = tok.toDouble(&tokenOk);
            row.push_back(tokenOk ? v : std::numeric_limits<double>::quiet_NaN());
        }
        rawRows.push_back(std::move(row));
    }

    if (rawRows.empty())
        throw std::runtime_error("Neural Designer format: no data rows found in file.");

    // Warn if declared count differs from actual
    if (static_cast<int>(rawRows.size()) != declaredSamples) {
        NS_WARN << QString("Neural Designer: declared %1 samples but found %2. "
                           "Using actual count.")
                      .arg(declaredSamples)
                      .arg(rawRows.size());
    }

    // ── Build Dataset ────────────────────────────────────────────────────────
    const int numOutputs = totalCols - numInputs;
    const int nSamples   = static_cast<int>(rawRows.size());

    auto ds = std::make_unique<Dataset>();
    ds->sourceFormat = "neuraldesigner";
    ds->sourcePath   = path;
    ds->name         = QFileInfo(path).baseName();
    ds->inputCount   = numInputs;
    ds->outputCount  = numOutputs;
    ds->sampleCount  = nSamples;

    // Default column names (user can rename later via UI)
    for (int i = 0; i < numInputs;  ++i)
        ds->inputNames  << QString("Input_%1").arg(i + 1);
    for (int i = 0; i < numOutputs; ++i)
        ds->outputNames << QString("Output_%1").arg(i + 1);

    // Allocate and fill
    ds->inputs .assign(nSamples, std::vector<double>(numInputs));
    ds->outputs.assign(nSamples, std::vector<double>(numOutputs));

    for (int r = 0; r < nSamples; ++r) {
        for (int c = 0; c < numInputs;  ++c)
            ds->inputs[r][c]  = rawRows[r][c];
        for (int c = 0; c < numOutputs; ++c)
            ds->outputs[r][c] = rawRows[r][numInputs + c];
    }

    ds->computeStatistics();

    NS_INFO << "Loaded" << ds->summary();
    return ds;
}

} // namespace NeuralStudio
