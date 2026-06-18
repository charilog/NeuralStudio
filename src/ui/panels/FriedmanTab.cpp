#include "FriedmanTab.h"
#include "BatchPlotWidget.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QSplitter>
#include <QGroupBox>
#include <QHeaderView>
#include <QFileDialog>
#include <QMessageBox>
#include <QTableWidgetItem>
#include <QFont>
#include <QSet>
#include <algorithm>
#include <numeric>

namespace NeuralStudio {

FriedmanTab::FriedmanTab(QWidget* parent) : QWidget(parent) { buildUi(); }

void FriedmanTab::buildUi() {
    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(8,8,8,8); root->setSpacing(6);

    auto* topRow = new QHBoxLayout;
    topRow->addWidget(new QLabel("Metric:"));
    m_metricCombo = new QComboBox; m_metricCombo->setFixedWidth(160);
    topRow->addWidget(m_metricCombo);
    topRow->addSpacing(12);
    topRow->addWidget(new QLabel("Task:"));
    m_taskCombo = new QComboBox; m_taskCombo->setFixedWidth(150);
    m_taskCombo->addItems({"All","Regression","Binary Classification","Multi-class"});
    topRow->addWidget(m_taskCombo);
    topRow->addStretch();
    m_exportBtn = new QPushButton("Export PNG  300 DPI");
    m_exportBtn->setMinimumWidth(160);
    m_exportBtn->setStyleSheet("QPushButton{background:#185fa5;color:white;"
                               "border-radius:5px;font-weight:600;padding:4px 12px;}");
    topRow->addWidget(m_exportBtn);
    root->addLayout(topRow);

    m_statsLbl = new QLabel("Friedman test (non-parametric, lower rank = better).");
    m_statsLbl->setStyleSheet("color:#A0A8C0;font-size:12px;");
    root->addWidget(m_statsLbl);

    auto* splitter = new QSplitter(Qt::Horizontal);

    auto* plotGroup = new QGroupBox("Metric Distribution per Optimizer");
    auto* pg = new QVBoxLayout(plotGroup);
    m_plot = new BatchPlotWidget;
    pg->addWidget(m_plot);
    splitter->addWidget(plotGroup);

    auto* tGroup = new QGroupBox("Post-hoc Analysis  (Friedman avg. ranks + pairwise Wilcoxon)");
    auto* tg = new QVBoxLayout(tGroup);
    tg->setSpacing(4);

    // ── Average Ranks mini-table ──────────────────────────────────────────────
    m_rankTable = new QTableWidget;
    m_rankTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_rankTable->setColumnCount(3);
    m_rankTable->setHorizontalHeaderLabels({"Rank", "Optimizer", "Avg. Rank"});
    m_rankTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    m_rankTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    m_rankTable->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    m_rankTable->setAlternatingRowColors(true);
    m_rankTable->setMaximumHeight(200);
    m_rankTable->setStyleSheet(
        "QTableWidget{background:#1f1f2a;alternate-background-color:#262635;}"
        "QHeaderView::section{background:#2a2a38;color:#E8E8F0;padding:4px;"
        "font-weight:600;}");
    tg->addWidget(m_rankTable);

    // ── Pairwise comparison table ─────────────────────────────────────────────
    m_pairLbl = new QLabel(
        "Pairwise Wilcoxon p-values (Bonferroni-corrected)  "
        "— * p<.05  ** p<.01  *** p<.001");
    m_pairLbl->setStyleSheet("color:#9098C0;font-size:11px;font-weight:600;"
                              "margin-top:6px;");
    tg->addWidget(m_pairLbl);

    m_pairTable = new QTableWidget;
    m_pairTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_pairTable->setAlternatingRowColors(false);
    m_pairTable->setSortingEnabled(false);
    m_pairTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    m_pairTable->verticalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    m_pairTable->setStyleSheet(
        "QTableWidget{background:#1f1f2a;gridline-color:#2a2a3a;}"
        "QHeaderView::section{background:#2a2a38;color:#E8E8F0;padding:4px;"
        "font-weight:600;}");
    tg->addWidget(m_pairTable, 1);
    splitter->addWidget(tGroup);
    splitter->setSizes({500, 350});

    root->addWidget(splitter, 1);

    connect(m_metricCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &FriedmanTab::onSelectionChanged);
    connect(m_taskCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &FriedmanTab::onSelectionChanged);
    connect(m_exportBtn, &QPushButton::clicked, this, &FriedmanTab::onExportPng);
}

void FriedmanTab::refresh(const std::vector<BatchJobResult>& results) {
    m_results = results;
    const QString prev = m_metricCombo->currentText();
    m_metricCombo->blockSignals(true);
    m_metricCombo->clear();

    bool hasClass=false, hasRegr=false, hasTest=false;
    for (const auto& r : results) {
        if (r.task == TaskType::Regression) hasRegr = true;
        else hasClass = true;
        if (r.hasTestSet) hasTest = true;
    }
    if (hasClass) {
        m_metricCombo->addItem("Val Err%",   "valErrorPct");
        if (hasTest) m_metricCombo->addItem("Test Err%", "testErrorPct");
        m_metricCombo->addItem("Train Err%", "trainErrorPct");
    }
    if (hasRegr) {
        m_metricCombo->addItem("Val MAE",    "valMAE");
        if (hasTest) m_metricCombo->addItem("Test MAE",  "testMAE");
        m_metricCombo->addItem("Train MAE",  "trainMAE");
        m_metricCombo->addItem("Val RMSE",   "valRMSE");
        if (hasTest) m_metricCombo->addItem("Test RMSE","testRMSE");
        if (hasTest) m_metricCombo->addItem("Test R²",  "testR2");
        m_metricCombo->addItem("Val R²",     "valR2");
    }
    int idx = m_metricCombo->findText(prev);
    if (idx >= 0) m_metricCombo->setCurrentIndex(idx);
    m_metricCombo->blockSignals(false);
    computeAndDisplay();
}

double FriedmanTab::extractMetric(const BatchJobResult& r, const QString& m) const {
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

bool FriedmanTab::buildMatrix(const QString& metricKey, const QString& taskFilter,
                               QVector<QString>& optNames,
                               QVector<QVector<double>>& matrix) {
    QSet<QString> optSet, dsSet;
    for (const auto& r : m_results) {
        if (taskFilter == "Regression" && r.task != TaskType::Regression) continue;
        if (taskFilter == "Binary Classification" && r.task != TaskType::BinaryClassification) continue;
        if (taskFilter == "Multi-class" && r.task != TaskType::MultiClassClassification) continue;
        double v = extractMetric(r, metricKey);
        if (v < 0) continue;
        optSet.insert(r.optimizerName.toLower());
        dsSet.insert(r.datasetName);
    }
    if (optSet.size() < 3 || dsSet.size() < 2) return false;

    optNames = QVector<QString>(optSet.begin(), optSet.end());
    std::sort(optNames.begin(), optNames.end());
    QVector<QString> dsNames(dsSet.begin(), dsSet.end());
    std::sort(dsNames.begin(), dsNames.end());

    QMap<QString,double> lookup;
    for (const auto& r : m_results) {
        if (taskFilter == "Regression" && r.task != TaskType::Regression) continue;
        if (taskFilter == "Binary Classification" && r.task != TaskType::BinaryClassification) continue;
        if (taskFilter == "Multi-class" && r.task != TaskType::MultiClassClassification) continue;
        double v = extractMetric(r, metricKey);
        if (v < 0) continue;
        lookup[r.optimizerName.toLower() + "|" + r.datasetName] = v;
    }

    // Keep datasets where ALL optimizers have values
    QVector<QString> validDs;
    for (const auto& ds : dsNames) {
        bool all = true;
        for (const auto& opt : optNames)
            if (!lookup.contains(opt+"|"+ds)) { all=false; break; }
        if (all) validDs.push_back(ds);
    }
    if (validDs.size() < 2) return false;

    matrix.resize(optNames.size());
    for (int j=0; j<optNames.size(); ++j) {
        matrix[j].resize(validDs.size());
        for (int i=0; i<validDs.size(); ++i)
            matrix[j][i] = lookup[optNames[j]+"|"+validDs[i]];
    }
    return true;
}

void FriedmanTab::computeAndDisplay() {
    m_valid = false;
    m_rankTable->setRowCount(0);
    m_pairTable->setRowCount(0);
    m_pairTable->setColumnCount(0);
    m_plot->setData({},{});
    m_plot->setBrackets({});

    if (m_results.empty() || m_metricCombo->count() == 0) {
        m_statsLbl->setText("No data yet."); return;
    }

    const QString metricKey  = m_metricCombo->currentData().toString();
    const QString metricLbl  = m_metricCombo->currentText();
    const QString taskFilter = m_taskCombo->currentText() == "All" ? "" : m_taskCombo->currentText();

    QVector<QString> optNames;
    QVector<QVector<double>> matrix;
    if (!buildMatrix(metricKey, taskFilter, optNames, matrix)) {
        m_statsLbl->setText(
            QString("Need ≥3 optimizers and ≥2 common datasets with valid '%1' values.")
                .arg(metricLbl));
        return;
    }

    const int k = optNames.size();
    const int N = matrix[0].size();

    // Convert to std types for Statistics::friedman
    std::vector<std::string> names;
    for (const auto& n : optNames) names.push_back(n.toStdString());
    std::vector<std::vector<double>> data;
    for (const auto& v : matrix) data.push_back(std::vector<double>(v.begin(), v.end()));

    auto fr = Statistics::friedman(names, data);
    if (!fr.valid) { m_statsLbl->setText("Friedman computation failed."); return; }

    m_valid = true;

    // ── Stats label ──────────────────────────────────────────────────────────
    QString starsStr = QString::fromStdString(fr.stars());
    m_statsLbl->setText(
        QString("Friedman  χ²(%1) = %2,  p = %3  %4    "
                "(%5 optimizers × %6 datasets,  metric: %7)")
            .arg(fr.df).arg(fr.chi2, 0,'f',3)
            .arg(fr.pValue < 0.001 ? "< 0.001" : QString::number(fr.pValue,'f',3))
            .arg(starsStr)
            .arg(k).arg(N).arg(metricLbl));
    m_statsLbl->setStyleSheet(
        QString("font-size:12px;font-weight:600;color:%1;")
            .arg(fr.pValue < 0.05 ? "#4EDDAD" : "#C8C8D8"));

    // ── Boxplot (sorted by avg rank) ─────────────────────────────────────────
    QVector<QString> dispNames;
    QVector<QVector<double>> vals;
    for (int ri = 0; ri < k; ++ri) {
        int j = fr.rankOrder[ri];
        dispNames.push_back(optNames[j].toUpper());
        vals.push_back(QVector<double>(matrix[j].begin(), matrix[j].end()));
    }
    m_plot->setData(dispNames, vals);
    m_plot->setYAxisLabel(metricLbl);
    m_plot->setTitle("Sorted by Friedman avg. rank  (best → worst)");
    m_plot->setLowerIsBetter(!metricKey.contains("R2"));

    // ── Average Ranks table ───────────────────────────────────────────────────
    m_rankTable->setRowCount(k);
    for (int ri = 0; ri < k; ++ri) {
        int j = fr.rankOrder[ri];
        const double ar = fr.avgRanks[j];
        auto makeItem = [](const QString& txt, Qt::AlignmentFlag align = Qt::AlignCenter){
            auto* it = new QTableWidgetItem(txt);
            it->setTextAlignment(align);
            return it;
        };
        auto* rItem = makeItem(QString::number(ri+1));
        auto* nItem = makeItem(optNames[j].toUpper(), Qt::AlignLeft);
        auto* aItem = makeItem(QString::number(ar,'f',3));
        if (ri==0) {
            for (auto* it : {rItem,nItem,aItem}) it->setForeground(QColor("#4EDDAD"));
        } else if (ri==k-1) {
            for (auto* it : {rItem,nItem,aItem}) it->setForeground(QColor("#FF6B6B"));
        }
        m_rankTable->setItem(ri,0,rItem);
        m_rankTable->setItem(ri,1,nItem);
        m_rankTable->setItem(ri,2,aItem);
    }

    // ── Pairwise post-hoc: Wilcoxon + Bonferroni correction ─────────────────
    // H0 per pair: no difference between optimizer i and j.
    // Bonferroni: multiply raw p by number of comparisons = k*(k-1)/2
    const int numPairs = k * (k-1) / 2;

    // Use the SORTED order (best→worst) for both headers and display
    QVector<QString> sortedNames;
    for (int ri=0; ri<k; ++ri) sortedNames.push_back(optNames[fr.rankOrder[ri]].toUpper());

    m_pairTable->setRowCount(k);
    m_pairTable->setColumnCount(k);
    m_pairTable->setHorizontalHeaderLabels(sortedNames);
    m_pairTable->setVerticalHeaderLabels(sortedNames);

    QVector<BatchPlotWidget::Bracket> brackets;

    for (int ra = 0; ra < k; ++ra) {
        int ja = fr.rankOrder[ra];   // optimizer index for row ra
        for (int rb = 0; rb < k; ++rb) {
            int jb = fr.rankOrder[rb];
            auto* it = new QTableWidgetItem;
            it->setTextAlignment(Qt::AlignCenter);

            if (ra == rb) {
                it->setText("—");
                it->setBackground(QColor(35,38,55));
                it->setForeground(QColor(70,75,100));
            } else if (ra > rb) {
                // Lower triangle: run Wilcoxon + Bonferroni
                std::vector<double> va(matrix[ja].begin(), matrix[ja].end());
                std::vector<double> vb(matrix[jb].begin(), matrix[jb].end());
                auto res = Statistics::wilcoxon(va, vb);

                double adjP = res.valid
                    ? std::min(1.0, res.pValue * numPairs)
                    : -1.0;

                // Stars based on ADJUSTED p
                QString stars;
                if      (!res.valid)  stars = "—";
                else if (adjP < 0.001) stars = "***";
                else if (adjP < 0.01)  stars = "**";
                else if (adjP < 0.05)  stars = "*";
                else                   stars = "ns";

                QString cellText = res.valid
                    ? QString("p=%1\n%2").arg(adjP, 0,'f',3).arg(stars)
                    : "n/a";
                it->setText(cellText);

                if (res.valid) {
                    if      (adjP < 0.001) it->setBackground(QColor(20,80,50));
                    else if (adjP < 0.01)  it->setBackground(QColor(20,60,40));
                    else if (adjP < 0.05)  it->setBackground(QColor(25,50,35));
                    else if (adjP < 0.1)   it->setBackground(QColor(60,55,25));
                    else                   it->setBackground(QColor(35,38,55));
                }
                it->setForeground(
                    (res.valid && adjP < 0.05) ? QColor("#4EDDAD") : QColor(170,175,200));

                // Bracket for the boxplot (use sorted indices ra/rb)
                brackets.push_back({rb, ra, stars, res.valid ? adjP : -1.0});
            } else {
                // Upper triangle: mirror indicator
                it->setText("(mirror)");
                it->setBackground(QColor(28,30,44));
                it->setForeground(QColor(55,60,85));
            }
            m_pairTable->setItem(ra, rb, it);
        }
    }

    // Sort brackets by significance for the boxplot display
    std::sort(brackets.begin(), brackets.end(),
              [](const BatchPlotWidget::Bracket& a, const BatchPlotWidget::Bracket& b){
                  if (a.pValue < 0) return false;
                  if (b.pValue < 0) return true;
                  return a.pValue < b.pValue;
              });
    m_plot->setBrackets(brackets);
}

void FriedmanTab::onSelectionChanged() { computeAndDisplay(); }

void FriedmanTab::onExportPng() {
    const QString path = QFileDialog::getSaveFileName(
        this, "Export Friedman boxplot", "friedman_boxplot.png",
        "PNG Images (*.png)");
    if (path.isEmpty()) return;
    if (!m_plot->exportPng(path))
        QMessageBox::warning(this, "Export Failed", "Could not save: " + path);
    else
        QMessageBox::information(this, "Export Done", "Saved at 300 DPI:\n" + path);
}

} // namespace NeuralStudio
