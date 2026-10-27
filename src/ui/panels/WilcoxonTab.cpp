#include "WilcoxonTab.h"
#include "BatchMetricMatrix.h"
#include "BatchPlotWidget.h"
#include "core/export/XlsxWriter.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QSplitter>
#include <QGroupBox>
#include <QHeaderView>
#include <QFileDialog>
#include <QMessageBox>
#include <QTableWidgetItem>
#include <QColor>
#include <QFont>
#include <QSet>
#include <algorithm>

namespace NeuralStudio {

WilcoxonTab::WilcoxonTab(QWidget* parent) : QWidget(parent) { buildUi(); }

void WilcoxonTab::buildUi() {
    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(8,8,8,8); root->setSpacing(6);

    // ── Top bar ───────────────────────────────────────────────────────────────
    auto* topRow = new QHBoxLayout;
    topRow->addWidget(new QLabel("Metric:"));
    m_metricCombo = new QComboBox; m_metricCombo->setFixedWidth(160);
    topRow->addWidget(m_metricCombo);
    topRow->addSpacing(12);
    topRow->addWidget(new QLabel("Task:"));
    m_taskCombo = new QComboBox; m_taskCombo->setFixedWidth(150);
    m_taskCombo->addItems({"All", "Regression", "Binary Classification", "Multi-class"});
    topRow->addWidget(m_taskCombo);
    topRow->addStretch();
    m_exportBtn = new QPushButton("Export PNG  300 DPI");
    m_exportBtn->setMinimumWidth(160);
    m_exportBtn->setStyleSheet("QPushButton{background:#185fa5;color:white;"
                               "border-radius:5px;font-weight:600;padding:4px 12px;}");
    topRow->addWidget(m_exportBtn);
    m_exportPBtn = new QPushButton("Export p-values XLSX");
    m_exportPBtn->setMinimumWidth(160);
    m_exportPBtn->setStyleSheet(m_exportBtn->styleSheet());
    topRow->addWidget(m_exportPBtn);
    root->addLayout(topRow);

    // ── Info label ────────────────────────────────────────────────────────────
    m_infoLbl = new QLabel(
        "Wilcoxon signed-rank test (two-tailed, lower metric = better). "
        "Table shows RAW (uncorrected) per-pair p-values — for multiple-"
        "comparison-adjusted conclusions use the Friedman tab's Nemenyi "
        "post-hoc. Exact permutation distribution when a pair has no ties, "
        "otherwise the normal approximation.");
    m_infoLbl->setStyleSheet("color:#A0A8C0;font-size:11px;font-style:italic;");
    root->addWidget(m_infoLbl);

    // ── Splitter: boxplot | p-value table ─────────────────────────────────────
    auto* splitter = new QSplitter(Qt::Horizontal);

    auto* plotGroup = new QGroupBox("Performance Distribution");
    auto* pg = new QVBoxLayout(plotGroup);
    m_plot = new BatchPlotWidget;
    pg->addWidget(m_plot);
    splitter->addWidget(plotGroup);

    auto* tGroup = new QGroupBox("Pairwise p-values  (* p<.05  ** p<.01  *** p<.001)");
    auto* tg = new QVBoxLayout(tGroup);
    m_pTable = new QTableWidget;
    m_pTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_pTable->setAlternatingRowColors(false);
    m_pTable->setSortingEnabled(false);
    m_pTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    m_pTable->verticalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    tg->addWidget(m_pTable);
    splitter->addWidget(tGroup);
    splitter->setSizes({500, 450});

    root->addWidget(splitter, 1);

    // Connections
    connect(m_metricCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &WilcoxonTab::onSelectionChanged);
    connect(m_taskCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &WilcoxonTab::onSelectionChanged);
    connect(m_exportBtn, &QPushButton::clicked, this, &WilcoxonTab::onExportPng);
    connect(m_exportPBtn, &QPushButton::clicked, this, &WilcoxonTab::onExportPvalsXlsx);
}

// ─── refresh ─────────────────────────────────────────────────────────────────
void WilcoxonTab::refresh(const std::vector<BatchJobResult>& results) {
    m_results = results;

    // Rebuild metric combo based on available data
    const QString prev = m_metricCombo->currentText();
    m_metricCombo->blockSignals(true);
    m_metricCombo->clear();

    bool hasClass = false, hasRegr = false, hasTest = false;
    for (const auto& r : results) {
        if (r.task == TaskType::Regression) hasRegr = true;
        else hasClass = true;
        if (r.hasTestSet) hasTest = true;
    }
    if (hasClass) {
        m_metricCombo->addItem("Val Err%",  "valErrorPct");
        if (hasTest) m_metricCombo->addItem("Test Err%", "testErrorPct");
        m_metricCombo->addItem("Train Err%","trainErrorPct");
    }
    if (hasRegr) {
        m_metricCombo->addItem("Val MAE",   "valMAE");
        if (hasTest) m_metricCombo->addItem("Test MAE",  "testMAE");
        m_metricCombo->addItem("Train MAE", "trainMAE");
        m_metricCombo->addItem("Val RMSE",  "valRMSE");
        if (hasTest) m_metricCombo->addItem("Test RMSE","testRMSE");
        if (hasTest) m_metricCombo->addItem("Test R²",  "testR2");
        m_metricCombo->addItem("Val R²",    "valR2");
    }

    // Restore previous selection if possible
    int idx = m_metricCombo->findText(prev);
    if (idx >= 0) m_metricCombo->setCurrentIndex(idx);
    m_metricCombo->blockSignals(false);

    computeAndDisplay();
}

// ─── extractMetric ────────────────────────────────────────────────────────────
double WilcoxonTab::extractMetric(const BatchJobResult& r, const QString& m) const {
    if (m == "valErrorPct")   return r.valErrorPct;
    if (m == "testErrorPct")  return r.testErrorPct;
    if (m == "trainErrorPct") return r.trainErrorPct;
    if (m == "valMAE")        return r.valMAE;
    if (m == "testMAE")       return r.testMAE;
    if (m == "trainMAE")      return r.trainMAE;
    if (m == "valRMSE")       return r.valRMSE;
    if (m == "testRMSE")      return r.testRMSE;
    if (m == "valR2")         return r.valR2;
    if (m == "testR2")        return r.testR2;
    return -1.0;
}

// ─── buildMatrix ─────────────────────────────────────────────────────────────
bool WilcoxonTab::buildMatrix(const QString& metricKey, const QString& taskFilter,
                               QVector<QString>& optNames,
                               QVector<QVector<double>>& matrix,
                               QVector<QString>& datasetNames)
{
    // shared builder: task-based validity (negative R² kept), repeats and folds averaged
    return buildBatchMetricMatrix(m_results, metricKey, taskFilter, 2, optNames, matrix, datasetNames, &m_averaged);
}

// ─── computeAndDisplay ───────────────────────────────────────────────────────
void WilcoxonTab::computeAndDisplay() {
    m_valid = false;
    m_pTable->clear();
    m_pTable->setRowCount(0); m_pTable->setColumnCount(0);
    m_plot->setData({}, {});

    if (m_results.empty() || m_metricCombo->count() == 0) {
        m_infoLbl->setText("No data yet.");
        return;
    }

    const QString metricKey = m_metricCombo->currentData().toString();
    const QString metricLbl = m_metricCombo->currentText();
    const QString taskFilter = m_taskCombo->currentText() == "All" ? ""
                               : m_taskCombo->currentText();

    QVector<QString>            optNames;
    QVector<QVector<double>>    matrix;
    QVector<QString>            dsNames;

    if (!buildMatrix(metricKey, taskFilter, optNames, matrix, dsNames)) {
        m_infoLbl->setText(
            QString("Need ≥2 optimizers and ≥2 common datasets with valid '%1' values.")
                .arg(metricLbl));
        return;
    }

    const int k = optNames.size();
    const int N = dsNames.size();
    m_valid = true;

    const QString lowPowerNote = (N < 5)
        ? QString("  \u26A0 Only %1 dataset%2 \u2014 p-values are valid but statistical "
                   "power is very low (a result can rarely reach p<0.05 with this few "
                   "paired observations).")
              .arg(N).arg(N == 1 ? "" : "s")
        : "";
    m_infoLbl->setText(
        QString("Wilcoxon signed-rank test (two-tailed), Holm-adjusted over the %5 pairs  |  "
                "%1 optimizers × %2 datasets  |  Metric: %3%4")
            .arg(k).arg(N).arg(metricLbl).arg(lowPowerNote).arg(k * (k - 1) / 2)
          + (m_averaged > 0 ? QString("  |  %1 repeated / fold results averaged (one value per dataset)").arg(m_averaged) : QString()));

    // ── Boxplot ───────────────────────────────────────────────────────────────
    QVector<QString> dispNames;
    QVector<QVector<double>> vals;
    for (int j = 0; j < k; ++j) {
        dispNames.push_back(optNames[j].toUpper());
        vals.push_back(QVector<double>(matrix[j].begin(), matrix[j].end()));
    }
    m_plot->setData(dispNames, vals);
    m_plot->setYAxisLabel(metricLbl);
    m_plot->setTitle("Distribution across " + QString::number(N) + " datasets");
    const bool lowerBetter = !metricKey.contains("R2");
    m_plot->setLowerIsBetter(lowerBetter);

    // ── Pairwise Wilcoxon ─────────────────────────────────────────────────────
    m_pTable->setRowCount(k); m_pTable->setColumnCount(k);
    m_pTable->setHorizontalHeaderLabels(dispNames);
    m_pTable->setVerticalHeaderLabels(dispNames);

    // Compute all pairwise tests and collect brackets for the plot.
    // k optimizers give k(k-1)/2 tests: the raw p-values are adjusted with Holm's step-down
    // method (family-wise error 5 %), and the stars / colours / brackets follow the adjusted
    // value. (Unadjusted pairwise p-values were shown, so with e.g. 8 optimizers = 28 tests
    // about one "significant" pair is expected by chance alone.)
    QVector<BatchPlotWidget::Bracket> brackets;
    std::vector<std::vector<WilcoxonResult>> wr(k, std::vector<WilcoxonResult>(k));
    std::vector<std::vector<double>> adj(k, std::vector<double>(k, -1.0));
    {
        struct P { int a, b; double p; };
        std::vector<P> ps;
        for (int a = 0; a < k; ++a)
            for (int b = 0; b < a; ++b) {
                std::vector<double> va(matrix[a].begin(), matrix[a].end());
                std::vector<double> vb(matrix[b].begin(), matrix[b].end());
                wr[a][b] = Statistics::wilcoxon(va, vb);
                if (wr[a][b].valid) ps.push_back({a, b, wr[a][b].pValue});
            }
        std::sort(ps.begin(), ps.end(), [](const P& x, const P& y) { return x.p < y.p; });
        const int m = static_cast<int>(ps.size());
        double run = 0.0;
        for (int i = 0; i < m; ++i) {
            run = std::max(run, std::min(1.0, (m - i) * ps[i].p));
            adj[ps[i].a][ps[i].b] = run;
        }
    }
    auto starsOf = [](double p) { return p < 0.001 ? QString("***") : p < 0.01 ? QString("**")
                                       : p < 0.05 ? QString("*") : QString("ns"); };

    for (int a = 0; a < k; ++a) {
        for (int b = 0; b < k; ++b) {
            auto* it = new QTableWidgetItem;
            it->setTextAlignment(Qt::AlignCenter);

            if (a == b) {
                it->setText("—");
                it->setBackground(QColor(35, 38, 55));
                it->setForeground(QColor(80, 85, 110));
            } else if (a > b) {
                // Lower triangle: show p-value + stars
                const WilcoxonResult& res = wr[a][b];
                const double pAdj = adj[a][b];

                QString stars = res.valid ? starsOf(pAdj) : QString("\u2014");
                QString cell  = res.valid
                    ? QString("p=%1  Holm %2\n%3").arg(res.pValue, 0, 'f', 3).arg(pAdj, 0, 'f', 3).arg(stars)
                    : "n/a";
                it->setText(cell);
                it->setToolTip(res.valid ? QString("raw p = %1, Holm-adjusted p = %2 (%3 pairs)")
                                               .arg(res.pValue, 0, 'g', 4).arg(pAdj, 0, 'g', 4).arg(k * (k - 1) / 2)
                                         : QString("no non-zero paired difference"));

                if (res.valid) {
                    if (pAdj < 0.001) it->setBackground(QColor(20, 80, 50));
                    else if (pAdj < 0.01)  it->setBackground(QColor(20, 60, 40));
                    else if (pAdj < 0.05)  it->setBackground(QColor(25, 50, 35));
                    else if (pAdj < 0.1)   it->setBackground(QColor(60, 55, 25));
                    else                   it->setBackground(QColor(35, 38, 55));
                }
                it->setForeground(res.valid && pAdj < 0.05 ? QColor("#4EDDAD") : QColor(180, 185, 200));

                // Always add bracket for every pair so all p-values appear on the PNG
                // (res.valid=false when every paired difference is zero — shown as "n/a")
                brackets.push_back({b, a, stars, res.valid ? pAdj : -1.0});
            } else {
                it->setText("(mirror)");
                it->setBackground(QColor(28, 30, 44));
                it->setForeground(QColor(60, 65, 90));
            }
            m_pTable->setItem(a, b, it);
        }
    }
    // Sort brackets by significance (most significant shown first/lowest)
    std::sort(brackets.begin(), brackets.end(),
              [](const BatchPlotWidget::Bracket& x, const BatchPlotWidget::Bracket& y){
                  return x.pValue < y.pValue;
              });
    m_plot->setBrackets(brackets);
}

// ─── onSelectionChanged ──────────────────────────────────────────────────────
void WilcoxonTab::onSelectionChanged() { computeAndDisplay(); }

// ─── onExportPng ─────────────────────────────────────────────────────────────
void WilcoxonTab::onExportPng() {
    const QString path = QFileDialog::getSaveFileName(
        this, "Export Wilcoxon boxplot", "wilcoxon_boxplot.png",
        "PNG Images (*.png)");
    if (path.isEmpty()) return;
    if (!m_plot->exportPng(path))
        QMessageBox::warning(this, "Export Failed", "Could not save: " + path);
    else
        QMessageBox::information(this, "Export Done",
            "Saved at 300 DPI:\n" + path);
}

// ─── onExportPvalsXlsx ───────────────────────────────────────────────────────
//  Exports the k×k pairwise p-value matrix as a spreadsheet: first column
//  holds the row optimizer name, remaining columns mirror the on-screen
//  table (multi-line cells are flattened to single-line text).
void WilcoxonTab::onExportPvalsXlsx() {
    if (!m_valid || !m_pTable || m_pTable->rowCount() == 0) {
        QMessageBox::information(this, "Export", "No p-value table yet — run a batch first.");
        return;
    }
    const QString path = QFileDialog::getSaveFileName(
        this, "Export pairwise p-values", "wilcoxon_pvalues.xlsx",
        XlsxWriter::fileFilter());
    if (path.isEmpty()) return;
    try {
        XlsxWriter xls;
        auto* sh = xls.addSheet("Wilcoxon p-values");
        QStringList hdr; hdr << "";   // corner cell above row names
        for (int c2 = 0; c2 < m_pTable->columnCount(); ++c2)
            hdr << (m_pTable->horizontalHeaderItem(c2)
                        ? m_pTable->horizontalHeaderItem(c2)->text() : "");
        sh->setHeader(hdr);
        for (int r = 0; r < m_pTable->rowCount(); ++r) {
            std::vector<XlsxCell> row;
            row.push_back(XlsxCell(m_pTable->verticalHeaderItem(r)
                                       ? m_pTable->verticalHeaderItem(r)->text() : QString()));
            for (int c2 = 0; c2 < m_pTable->columnCount(); ++c2) {
                QString t = m_pTable->item(r, c2) ? m_pTable->item(r, c2)->text() : "";
                t.replace('\n', ' ');
                row.push_back(XlsxCell(t));
            }
            sh->addRow(row);
        }
        xls.save(path);
        QMessageBox::information(this, "Export Done", "Saved:\n" + path);
    } catch (const std::exception& ex) {
        QMessageBox::critical(this, "Export Error", ex.what());
    }
}

} // namespace NeuralStudio
