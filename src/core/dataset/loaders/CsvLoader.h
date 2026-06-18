#pragma once

#include "IDataLoader.h"

namespace NeuralStudio {

// ─── CsvLoader ────────────────────────────────────────────────────────────────
//  Parses CSV / TSV files.
//
//  Auto-detection rules:
//    • Delimiter : tries ',', ';', '\t' — picks the one with most consistent
//                  column counts across first 5 rows.
//    • Header row: if the first row contains any non-numeric token the entire
//                  first row is treated as column names.
//
//  Input / output split:
//    By default the LAST column is the output.
//    Call setOutputCount(n) before load() to override.
//    (Phase 2: the UI will allow the user to pick columns interactively.)
// ─────────────────────────────────────────────────────────────────────────────
class CsvLoader : public IDataLoader {
public:
    explicit CsvLoader(int outputCount = 1) : m_outputCount(outputCount) {}

    std::unique_ptr<Dataset> load(const QString& path) override;
    QStringList              extensions()  const override;
    QString                  formatName()  const override;
    bool                     canLoad(const QString& path) const override;

    void setOutputCount(int n) { m_outputCount = n; }
    int  outputCount()   const { return m_outputCount; }

private:
    int     m_outputCount = 1;

    // Internal helpers
    static char   detectDelimiter(const QStringList& firstLines);
    static bool   isNumeric(const QString& s);
    static double parseValue(const QString& s, bool& ok);
};

} // namespace NeuralStudio
