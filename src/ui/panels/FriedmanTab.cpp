#include "FriedmanTab.h"
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
    m_exportPBtn = new QPushButton("Export p-values XLSX");
    m_exportPBtn->setMinimumWidth(160);
    m_exportPBtn->setStyleSheet(m_exportBtn->styleSheet());
    topRow->addWidget(m_exportPBtn);
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
        "Nemenyi post-hoc on average ranks (Demšar 2006) — pairs where the "
        "rank gap exceeds the Critical Difference are significant  "
        "— * p<.05 (by CD)  ns = not significant");
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
    connect(m_exportPBtn, &QPushButton::clicked, this, &FriedmanTab::onExportPvalsXlsx);
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
    // shared builder: task-based validity (negative R² kept), repeats and folds averaged
    QVector<QString> ds;
    return buildBatchMetricMatrix(m_results, metricKey, taskFilter, 3, optNames, matrix, ds, &m_averaged);
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

    // Friedman/Nemenyi rank with the LOWER-IS-BETTER convention. R² is the
    // one available metric where HIGHER is better, so negate the analysis
    // copy (ranking of -R² == inverse ranking of R²); the display matrix
    // stays untouched so the boxplot shows the real values. Without this,
    // selecting an R² metric silently inverted every rank, the best/worst
    // highlighting, and the post-hoc conclusions.
    const bool higherIsBetter = metricKey.contains("R2");
    if (higherIsBetter)
        for (auto& row : data)
            for (double& v : row) v = -v;

    auto fr = Statistics::friedman(names, data);
    if (!fr.valid) { m_statsLbl->setText("Friedman computation failed."); return; }

    m_valid = true;

    // ── Stats label ──────────────────────────────────────────────────────────
    QString starsStr = QString::fromStdString(fr.stars());
    const QString lowPowerNote = (N < 5)
        ? QString("   \u26A0 Only %1 datasets \u2014 limited power").arg(N)
        : "";
    m_statsLbl->setText(
        QString("Friedman  χ²(%1) = %2,  p = %3  %4    "
                "(%5 optimizers × %6 datasets,  metric: %7)%8")
            .arg(fr.df).arg(fr.chi2, 0,'f',3)
            .arg(fr.pValue < 0.001 ? "< 0.001" : QString::number(fr.pValue,'f',3))
            .arg(starsStr)
            .arg(k).arg(N).arg(metricLbl).arg(lowPowerNote));
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

    // ── Pairwise post-hoc: NEMENYI test on average ranks ────────────────────
    // After a significant Friedman omnibus test, the field-standard all-pairs
    // post-hoc (Demšar 2006) is the Nemenyi test: two optimizers differ
    // significantly when their average-rank gap exceeds the Critical
    // Difference. This pools information across ALL datasets via the ranks,
    // so it keeps usable power even when N is small — unlike pairwise
    // Wilcoxon + Bonferroni, whose minimum achievable p (2/2^N) alone can
    // exceed 0.05/numPairs and make significance impossible by construction.
    // 'data' (std::vector<std::vector<double>>) was already built above for
    // the Statistics::friedman(names, data) call — reuse it directly.
    auto nem = Statistics::nemenyi(data);

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
                // Lower triangle: Nemenyi significance from the rank gap vs CD
                const double gap = nem.valid
                    ? std::abs(nem.avgRanks[ja] - nem.avgRanks[jb]) : -1.0;
                const double pApprox = nem.valid ? nem.pPair[ja][jb] : -1.0;

                // Significance is decided by the Critical Difference, the
                // proper Nemenyi criterion; the p-value is the matching
                // family-wise Nemenyi p (p < 0.05 exactly when gap > CD).
                QString stars = "—";
                bool sig05 = false, sig10 = false;
                if (nem.valid) {
                    sig05 = gap > nem.cd05;
                    sig10 = gap > nem.cd10;
                    stars = sig05 ? "*" : "ns";
                }

                QString cellText = nem.valid
                    ? QString("Δrank=%1\n%2").arg(gap, 0,'f',2).arg(stars)
                    : "n/a";
                it->setText(cellText);
                it->setToolTip(nem.valid
                    ? QString("Average-rank gap %1 (CD₀.₀₅=%2, CD₀.₁₀=%3)\n"
                              "Nemenyi p=%4")
                          .arg(gap,0,'f',3).arg(nem.cd05,0,'f',3)
                          .arg(nem.cd10,0,'f',3).arg(pApprox,0,'f',3)
                    : "Not enough data");

                if (nem.valid) {
                    if      (sig05) it->setBackground(QColor(20,80,50));
                    else if (sig10) it->setBackground(QColor(60,55,25));
                    else            it->setBackground(QColor(35,38,55));
                }
                it->setForeground(sig05 ? QColor("#4EDDAD") : QColor(170,175,200));

                // Bracket for the boxplot: use the approximate p so the most
                // separated pairs sort to the top.
                brackets.push_back({rb, ra, stars, nem.valid ? pApprox : -1.0});
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

// ─── onExportPvalsXlsx ───────────────────────────────────────────────────────
//  Exports the Nemenyi pairwise matrix (rank gaps / significance) exactly
//  as shown on screen; first column carries the row optimizer name.
void FriedmanTab::onExportPvalsXlsx() {
    if (!m_valid || !m_pairTable || m_pairTable->rowCount() == 0) {
        QMessageBox::information(this, "Export", "No pairwise table yet — run a batch first.");
        return;
    }
    const QString path = QFileDialog::getSaveFileName(
        this, "Export pairwise comparisons", "friedman_nemenyi_pairs.xlsx",
        XlsxWriter::fileFilter());
    if (path.isEmpty()) return;
    try {
        XlsxWriter xls;
        auto* sh = xls.addSheet("Nemenyi pairs");
        QStringList hdr; hdr << "";
        for (int c2 = 0; c2 < m_pairTable->columnCount(); ++c2)
            hdr << (m_pairTable->horizontalHeaderItem(c2)
                        ? m_pairTable->horizontalHeaderItem(c2)->text() : "");
        sh->setHeader(hdr);
        for (int r = 0; r < m_pairTable->rowCount(); ++r) {
            std::vector<XlsxCell> row;
            row.push_back(XlsxCell(m_pairTable->verticalHeaderItem(r)
                                       ? m_pairTable->verticalHeaderItem(r)->text() : QString()));
            for (int c2 = 0; c2 < m_pairTable->columnCount(); ++c2) {
                QString t = m_pairTable->item(r, c2) ? m_pairTable->item(r, c2)->text() : "";
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
