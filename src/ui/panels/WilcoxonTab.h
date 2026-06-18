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

// ─── WilcoxonTab ──────────────────────────────────────────────────────────────
//  Pairwise Wilcoxon signed-rank tests.
//
//  Layout:
//    [Metric ▼] [Task ▼]  [Export PNG]
//    ┌─────────────────────┬───────────────────────────────────┐
//    │  Boxplot of metric  │  Pairwise p-value matrix          │
//    │  per optimizer       │  (colour-coded by significance)   │
//    └─────────────────────┴───────────────────────────────────┘
//    Legend: * p<0.05  ** p<0.01  *** p<0.001  ns = not significant
// ─────────────────────────────────────────────────────────────────────────────
class WilcoxonTab : public QWidget {
    Q_OBJECT
public:
    explicit WilcoxonTab(QWidget* parent = nullptr);

    // Call after each batch completes (or partially)
    void refresh(const std::vector<BatchJobResult>& results);

    // Returns true if there is enough data for the test
    bool hasValidData() const { return m_valid; }

private slots:
    void onExportPng();
    void onSelectionChanged();

private:
    void buildUi();
    void computeAndDisplay();

    // Extract metric values: result[optimizer][dataset_index]
    // Returns false if insufficient data (< 2 optimizers, < 2 shared datasets)
    bool buildMatrix(const QString& metric, const QString& taskFilter,
                     QVector<QString>&           optNames,
                     QVector<QVector<double>>&   matrix,
                     QVector<QString>&           datasetNames);

    double extractMetric(const BatchJobResult& r, const QString& metric) const;

    // ── UI ───────────────────────────────────────────────────────────────────
    QComboBox*      m_metricCombo = nullptr;
    QComboBox*      m_taskCombo   = nullptr;
    QPushButton*    m_exportBtn   = nullptr;
    BatchPlotWidget* m_plot       = nullptr;
    QTableWidget*   m_pTable      = nullptr;
    QLabel*         m_infoLbl     = nullptr;

    std::vector<BatchJobResult> m_results;
    bool m_valid = false;
};

} // namespace NeuralStudio
