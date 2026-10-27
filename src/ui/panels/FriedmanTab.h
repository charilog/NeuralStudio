#pragma once
#include "core/batch/BatchJob.h"
#include "core/stats/Statistics.h"
#include <QWidget>
#include <QComboBox>
#include <QPushButton>
#include <QTableWidget>
#include <QLabel>
#include <vector>

namespace NeuralStudio { class BatchPlotWidget; }

namespace NeuralStudio {

// ─── FriedmanTab ──────────────────────────────────────────────────────────────
//  Friedman non-parametric test across k optimizers × N datasets.
//
//  Layout:
//    [Metric ▼] [Task ▼]  [Export PNG]
//    Friedman χ²(k-1) = 14.32, p = 0.006 **
//    ┌─────────────────────────┬────────────────────────────────┐
//    │  Boxplot of raw ranks   │  Average Ranks table           │
//    │  per optimizer           │  Rank | Optimizer | Avg.Rank  │
//    └─────────────────────────┴────────────────────────────────┘
// ─────────────────────────────────────────────────────────────────────────────
class FriedmanTab : public QWidget {
    Q_OBJECT
public:
    explicit FriedmanTab(QWidget* parent = nullptr);
    void refresh(const std::vector<BatchJobResult>& results);
    bool hasValidData() const { return m_valid; }

private slots:
    void onExportPng();
    void onExportPvalsXlsx();   // Nemenyi pairwise matrix → .xlsx
    void onSelectionChanged();

private:
    void buildUi();
    void computeAndDisplay();

    bool buildMatrix(const QString& metricKey, const QString& taskFilter,
                     QVector<QString>& optNames,
                     QVector<QVector<double>>& matrix);

    double extractMetric(const BatchJobResult& r, const QString& metric) const;

    QComboBox*       m_metricCombo = nullptr;
    QComboBox*       m_taskCombo   = nullptr;
    QPushButton*     m_exportBtn   = nullptr;
    QPushButton*     m_exportPBtn  = nullptr;   // export pairwise p-values
    BatchPlotWidget* m_plot        = nullptr;
    QTableWidget*    m_rankTable   = nullptr;
    QTableWidget*    m_pairTable   = nullptr;   // pairwise post-hoc p-values
    QLabel*          m_statsLbl    = nullptr;
    QLabel*          m_pairLbl     = nullptr;   // header for pairwise section

    std::vector<BatchJobResult> m_results;
    int m_averaged = 0;   // results merged into a (optimizer, dataset) mean
    bool m_valid = false;
};

} // namespace NeuralStudio
