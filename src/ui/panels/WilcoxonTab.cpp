#include "WilcoxonTab.h"
#include "BatchPlotWidget.h"
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
    root->addLayout(topRow);

    // ── Info label ────────────────────────────────────────────────────────────
    m_infoLbl = new QLabel("Wilcoxon signed-rank test (two-tailed, lower metric = better).");
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
    // Collect unique optimizer names and datasets
    QSet<QString> optSet, dsSet;
    for (const auto& r : m_results) {
        // Task filter
        if (taskFilter == "Regression" && r.task != TaskType::Regression) continue;
        if (taskFilter == "Binary Classification" && r.task != TaskType::BinaryClassification) continue;
        if (taskFilter == "Multi-class" && r.task != TaskType::MultiClassClassification) continue;

        double v = extractMetric(r, metricKey);
        if (v < 0) continue;   // N/A for this task type
        optSet.insert(r.optimizerName.toLower());
        dsSet.insert(r.datasetName);
    }
    if (optSet.size() < 2 || dsSet.size() < 2) return false;

    optNames = QVector<QString>(optSet.begin(), optSet.end());
    std::sort(optNames.begin(), optNames.end());
    datasetNames = QVector<QString>(dsSet.begin(), dsSet.end());
    std::sort(datasetNames.begin(), datasetNames.end());

    // Build lookup: (optName, dsName) → metric value
    QMap<QString, double> lookup;
    for (const auto& r : m_results) {
        if (taskFilter == "Regression" && r.task != TaskType::Regression) continue;
        if (taskFilter == "Binary Classification" && r.task != TaskType::BinaryClassification) continue;
        if (taskFilter == "Multi-class" && r.task != TaskType::MultiClassClassification) continue;
        double v = extractMetric(r, metricKey);
        if (v < 0) continue;
        lookup[r.optimizerName.toLower() + "|" + r.datasetName] = v;
    }

    // Keep only datasets where ALL optimizers have valid values
    QVector<QString> validDs;
    for (const auto& ds : datasetNames) {
        bool all = true;
        for (const auto& opt : optNames)
            if (!lookup.contains(opt + "|" + ds)) { all = false; break; }
        if (all) validDs.push_back(ds);
    }
    datasetNames = validDs;
    if (datasetNames.size() < 2) return false;

    // Fill matrix[optimizer][dataset]
    matrix.resize(optNames.size());
    for (int j = 0; j < optNames.size(); ++j) {
        matrix[j].resize(datasetNames.size());
        for (int i = 0; i < datasetNames.size(); ++i)
            matrix[j][i] = lookup[optNames[j] + "|" + datasetNames[i]];
    }
    return true;
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

    m_infoLbl->setText(
        QString("Wilcoxon signed-rank test (two-tailed)  |  %1 optimizers × %2 datasets  |  "
                "Metric: %3")
            .arg(k).arg(N).arg(metricLbl));

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

    // Compute all pairwise tests and collect brackets for the plot
    QVector<BatchPlotWidget::Bracket> brackets;

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
                std::vector<double> va(matrix[a].begin(), matrix[a].end());
                std::vector<double> vb(matrix[b].begin(), matrix[b].end());
                auto res = Statistics::wilcoxon(va, vb);

                QString stars = QString::fromStdString(res.stars());
                QString cell  = res.valid
                    ? QString("p=%1\n%2").arg(res.pValue, 0, 'f', 3).arg(stars)
                    : "n/a";
                it->setText(cell);

                if (res.valid) {
                    if (res.pValue < 0.001) it->setBackground(QColor(20, 80, 50));
                    else if (res.pValue < 0.01)  it->setBackground(QColor(20, 60, 40));
                    else if (res.pValue < 0.05)  it->setBackground(QColor(25, 50, 35));
                    else if (res.pValue < 0.1)   it->setBackground(QColor(60, 55, 25));
                    else                         it->setBackground(QColor(35, 38, 55));
                }
                it->setForeground(res.pValue < 0.05 ? QColor("#4EDDAD") : QColor(180, 185, 200));

                // Always add bracket for every pair so all p-values appear on the PNG
                // (res.valid=false when n<3 non-zero differences — shown as "n/a")
                brackets.push_back({b, a, stars, res.valid ? res.pValue : -1.0});
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

} // namespace NeuralStudio
