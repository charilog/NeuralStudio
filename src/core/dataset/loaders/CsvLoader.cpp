#include "CsvLoader.h"

#include "utils/Logger.h"
#include <QFile>
#include <QFileInfo>
#include <QTextStream>
#include <stdexcept>
#include <limits>
#include <map>

namespace NeuralStudio {

// ─── Metadata ────────────────────────────────────────────────────────────────
QStringList CsvLoader::extensions() const { return {"csv", "tsv", "txt"}; }
QString     CsvLoader::formatName() const { return "CSV / TSV"; }

bool CsvLoader::canLoad(const QString& path) const {
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) return false;
    // Just check we can read at least one non-empty line
    QTextStream in(&f);
    while (!in.atEnd()) {
        if (!in.readLine().trimmed().isEmpty()) return true;
    }
    return false;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────
bool CsvLoader::isNumeric(const QString& s) {
    bool ok = false;
    s.toDouble(&ok);
    return ok;
}

double CsvLoader::parseValue(const QString& s, bool& ok) {
    double v = s.trimmed().toDouble(&ok);
    return ok ? v : std::numeric_limits<double>::quiet_NaN();
}

// Auto-detect delimiter by finding which character produces the most
// consistent column count across the first few lines.
char CsvLoader::detectDelimiter(const QStringList& firstLines) {
    const QList<char> candidates = {',', ';', '\t', ' '};

    char best  = ',';
    int  bestScore = -1;

    for (char delim : candidates) {
        std::map<int, int> counts; // col_count → frequency
        for (const QString& line : firstLines) {
            if (line.trimmed().isEmpty()) continue;
            int n = line.split(QChar(delim), Qt::SkipEmptyParts).size();
            counts[n]++;
        }
        if (counts.empty()) continue;
        // Highest frequency wins; prefer more columns (more informative split)
        auto it = std::max_element(counts.begin(), counts.end(),
            [](const auto& a, const auto& b){ return a.second < b.second; });
        const int score = it->second * it->first; // frequency * column_count
        if (score > bestScore) { bestScore = score; best = delim; }
    }
    return best;
}

// ─── load ────────────────────────────────────────────────────────────────────
std::unique_ptr<Dataset> CsvLoader::load(const QString& path) {
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        throw std::runtime_error("Cannot open file: " + path.toStdString());

    QTextStream in(&file);
    in.setEncoding(QStringConverter::Utf8);

    // ── Read all non-empty lines ─────────────────────────────────────────────
    QStringList allLines;
    while (!in.atEnd()) {
        QString line = in.readLine();
        if (!line.trimmed().isEmpty()) allLines << line;
    }
    if (allLines.size() < 2)
        throw std::runtime_error("CSV: file has fewer than 2 non-empty lines.");

    // ── Detect delimiter ─────────────────────────────────────────────────────
    const QStringList peekLines = allLines.mid(0, std::min(allLines.size(), qsizetype(5)));
    const char delim = detectDelimiter(peekLines);
    NS_DEBUG << QString("CSV: detected delimiter '%1'").arg(QChar(delim));

    // ── Split all lines ──────────────────────────────────────────────────────
    auto splitLine = [&](const QString& line) {
        QStringList toks = line.split(QChar(delim), Qt::SkipEmptyParts);
        for (auto& t : toks) t = t.trimmed().remove('"');
        return toks;
    };

    // ── Header detection ─────────────────────────────────────────────────────
    QStringList header;
    int dataStartIdx = 0;
    {
        const QStringList firstToks = splitLine(allLines[0]);
        // If ANY token in the first row is non-numeric → header row
        bool hasHeader = false;
        for (const QString& t : firstToks)
            if (!isNumeric(t)) { hasHeader = true; break; }

        if (hasHeader) {
            header = firstToks;
            dataStartIdx = 1;
            NS_DEBUG << "CSV: first row treated as header.";
        } else {
            // Auto-generate column names
            for (int i = 0; i < firstToks.size(); ++i)
                header << QString("Col_%1").arg(i + 1);
            dataStartIdx = 0;
        }
    }

    const int totalCols = header.size();
    if (totalCols < 2)
        throw std::runtime_error("CSV: need at least 2 columns (1 input + 1 output).");

    // Clamp output count
    const int numOutputs = std::min(m_outputCount, totalCols - 1);
    const int numInputs  = totalCols - numOutputs;

    // ── Parse data rows ──────────────────────────────────────────────────────
    std::vector<std::vector<double>> rawRows;
    rawRows.reserve(allLines.size() - dataStartIdx);

    for (int i = dataStartIdx; i < allLines.size(); ++i) {
        const QStringList toks = splitLine(allLines[i]);
        if (toks.size() != totalCols) {
            NS_WARN << QString("CSV: skipping line %1 (expected %2 cols, got %3)")
                          .arg(i + 1).arg(totalCols).arg(toks.size());
            continue;
        }
        std::vector<double> row;
        row.reserve(totalCols);
        for (const QString& t : toks) {
            bool ok = false;
            row.push_back(parseValue(t, ok));
        }
        rawRows.push_back(std::move(row));
    }

    if (rawRows.empty())
        throw std::runtime_error("CSV: no valid data rows found.");

    // ── Build Dataset ────────────────────────────────────────────────────────
    const int nSamples = static_cast<int>(rawRows.size());
    auto ds = std::make_unique<Dataset>();
    ds->sourceFormat = "csv";
    ds->sourcePath   = path;
    ds->name         = QFileInfo(path).baseName();
    ds->inputCount   = numInputs;
    ds->outputCount  = numOutputs;
    ds->sampleCount  = nSamples;

    for (int i = 0; i < numInputs;  ++i) ds->inputNames  << header[i];
    for (int i = 0; i < numOutputs; ++i) ds->outputNames << header[numInputs + i];

    ds->inputs .assign(nSamples, std::vector<double>(numInputs));
    ds->outputs.assign(nSamples, std::vector<double>(numOutputs));

    for (int r = 0; r < nSamples; ++r) {
        for (int c = 0; c < numInputs;  ++c) ds->inputs [r][c] = rawRows[r][c];
        for (int c = 0; c < numOutputs; ++c) ds->outputs[r][c] = rawRows[r][numInputs + c];
    }

    ds->computeStatistics();
    NS_INFO << "Loaded" << ds->summary();
    return ds;
}

} // namespace NeuralStudio
