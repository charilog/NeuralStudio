#include "DataPanel.h"

#include "core/dataset/DatasetLoader.h"
#include "utils/Logger.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QFileDialog>
#include <QMimeData>
#include <QMessageBox>
#include <QFont>
#include <QColor>
#include <QTableWidgetItem>
#include <QSizePolicy>
#include <QFileInfo>
#include <map>

namespace NeuralStudio {

DataPanel::DataPanel(QWidget* parent) : QWidget(parent) {
    setAcceptDrops(true);
    buildUi();
    setStatus("No dataset loaded. Click 'Load File...' or drag a file here.");
}

void DataPanel::buildUi() {
    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(16, 16, 16, 16);
    root->setSpacing(8);

    // Top toolbar
    auto* toolbar = new QHBoxLayout;
    toolbar->setSpacing(8);
    m_loadBtn = new QPushButton("Load File...");
    m_loadBtn->setFixedHeight(32);
    m_loadBtn->setMinimumWidth(120);
    connect(m_loadBtn, &QPushButton::clicked, this, &DataPanel::onLoadClicked);

    m_fileLabel = new QLabel("—");
    m_fileLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);

    m_formatBadge = new QLabel;
    m_formatBadge->setAlignment(Qt::AlignCenter);
    m_formatBadge->setFixedHeight(22);
    m_formatBadge->setMinimumWidth(110);
    m_formatBadge->hide();

    toolbar->addWidget(m_loadBtn);
    toolbar->addWidget(m_fileLabel, 1);
    toolbar->addWidget(m_formatBadge);

    // Companion file label (auto-loaded test file indicator)
    m_companionLbl = new QLabel;
    m_companionLbl->setStyleSheet(
        "QLabel{background:#0f6e56;color:white;border-radius:4px;"
        "padding:4px 10px;font-size:11px;}");
    m_companionLbl->hide();

    // Info bar
    m_infoBar = new QLabel;
    m_infoBar->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    m_infoBar->setFixedHeight(28);
    m_infoBar->setStyleSheet(
        "QLabel{background:palette(midlight);border-radius:4px;"
        "padding:0 10px;font-size:12px;color:palette(text);}");

    // Class balance bar
    m_balanceLbl = new QLabel;
    m_balanceLbl->setWordWrap(true);
    m_balanceLbl->setStyleSheet(
        "QLabel{background:palette(midlight);border-radius:4px;"
        "padding:6px 10px;font-size:11px;font-family:monospace;}");
    m_balanceLbl->hide();

    // Tabs
    m_tabs = new QTabWidget;
    m_previewTable = new QTableWidget;
    m_previewTable->setAlternatingRowColors(true);
    m_previewTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_previewTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_previewTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Interactive);
    m_previewTable->horizontalHeader()->setDefaultSectionSize(90);
    m_previewTable->verticalHeader()->setDefaultSectionSize(22);
    m_tabs->addTab(m_previewTable, "Preview");

    m_statsTable = new QTableWidget;
    m_statsTable->setAlternatingRowColors(true);
    m_statsTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_statsTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_statsTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    m_statsTable->verticalHeader()->setDefaultSectionSize(22);
    m_tabs->addTab(m_statsTable, "Statistics");

    root->addLayout(toolbar);
    root->addWidget(m_companionLbl);
    root->addWidget(m_infoBar);
    root->addWidget(m_balanceLbl);
    root->addWidget(m_tabs, 1);
}

void DataPanel::onLoadClicked() {
    const QString path = QFileDialog::getOpenFileName(
        this, "Open Dataset", QString(), DatasetLoader::fileDialogFilter());
    if (!path.isEmpty()) loadFile(path);
}

void DataPanel::loadFile(const QString& path) {
    try {
        m_bundle = DatasetLoader::loadBundle(path);
    } catch (const std::exception& ex) {
        NS_ERROR << "Load failed:" << ex.what();
        QMessageBox::critical(this, "Load Error",
            QString("Failed to load file:\n%1\n\nError: %2").arg(path, ex.what()));
        setStatus(QString("Error: %1").arg(ex.what()), true);
        return;
    }

    // Update file label (always show train file name)
    m_fileLabel->setText(QFileInfo(m_bundle.train->sourcePath).fileName());

    // Format badge
    const QString fmt = m_bundle.train->sourceFormat;
    QString badgeText, badgeStyle;
    if (fmt == "neuraldesigner") {
        badgeText  = "NeuralStudio";
        badgeStyle = "background:#dff2e9;color:#0f6e56;border-radius:4px;"
                     "padding:2px 8px;font-size:11px;font-weight:500;";
    } else if (fmt == "csv") {
        badgeText  = "CSV / TSV";
        badgeStyle = "background:#e6f1fb;color:#185fa5;border-radius:4px;"
                     "padding:2px 8px;font-size:11px;font-weight:500;";
    } else {
        badgeText  = fmt;
        badgeStyle = "background:palette(midlight);border-radius:4px;"
                     "padding:2px 8px;font-size:11px;";
    }
    m_formatBadge->setText(badgeText);
    m_formatBadge->setStyleSheet(QString("QLabel{%1}").arg(badgeStyle));
    m_formatBadge->show();

    // Companion banner
    if (m_bundle.hasTest()) {
        const QString testName = QFileInfo(m_bundle.test->sourcePath).fileName();
        m_companionLbl->setText(
            QString("✓  Companion test file auto-loaded:  %1  (%2 samples)")
                .arg(testName).arg(m_bundle.test->sampleCount));
        m_companionLbl->show();
    } else {
        m_companionLbl->hide();
    }

    updateInfoBar();
    updateBalanceInfo();
    populatePreview();
    populateStats();

    emit datasetLoaded(m_bundle.train.get());
    emit testDatasetLoaded(m_bundle.test.get());  // may be nullptr
}

void DataPanel::updateInfoBar() {
    if (!m_bundle.train) { m_infoBar->setText(""); return; }
    const auto* ds = m_bundle.train.get();
    const auto unique = ds->uniqueOutputValues(0);
    QString classInfo = unique.size() <= 20
        ? QString("  |  %1 unique output values").arg(unique.size()) : "";
    m_infoBar->setText(
        QString("  %1 samples    %2 inputs    %3 output%4%5")
            .arg(ds->sampleCount).arg(ds->inputCount)
            .arg(ds->outputCount).arg(ds->outputCount > 1 ? "s" : "")
            .arg(classInfo));
}

void DataPanel::updateBalanceInfo() {
    if (!m_bundle.train) { m_balanceLbl->hide(); return; }
    const auto* ds = m_bundle.train.get();
    const auto unique = ds->uniqueOutputValues(0);

    // Only show for classification-like datasets
    if (unique.size() < 2 || unique.size() > 20) {
        m_balanceLbl->hide();
        return;
    }

    // Count samples per class
    std::map<double, int> counts;
    for (int r = 0; r < ds->sampleCount; ++r)
        counts[ds->outputAt(r, 0)]++;

    // Find min/max class counts to detect imbalance
    int minN = ds->sampleCount, maxN = 0;
    for (auto& [k, v] : counts) {
        minN = std::min(minN, v);
        maxN = std::max(maxN, v);
    }
    const double imbalance = (minN > 0) ? double(maxN) / double(minN) : 0.0;

    // Build per-class text
    QStringList parts;
    for (auto& [val, n] : counts) {
        double pct = 100.0 * n / ds->sampleCount;
        parts << QString("class %1: %2 (%3%)")
            .arg(val, 0, 'f', 1).arg(n).arg(pct, 0, 'f', 1);
    }

    QString prefix;
    if (imbalance >= 3.0)
        prefix = "<b style='color:#C0392B'>⚠ Imbalanced</b>  ";
    else if (imbalance >= 1.5)
        prefix = "<b style='color:#E8700A'>Slightly imbalanced</b>  ";
    else
        prefix = "<b style='color:#1D9E75'>✓ Balanced</b>  ";

    m_balanceLbl->setText(prefix + QString("(ratio %1:1)  —  ").arg(imbalance, 0, 'f', 1)
                          + parts.join("  ·  "));
    m_balanceLbl->show();
}

void DataPanel::populatePreview() {
    if (!m_bundle.train) return;
    const auto* ds = m_bundle.train.get();
    const int rows = std::min(ds->sampleCount, kPreviewRows);
    const int cols = ds->inputCount + ds->outputCount;

    m_previewTable->clearContents();
    m_previewTable->setRowCount(rows);
    m_previewTable->setColumnCount(cols);
    QStringList headers = ds->inputNames + ds->outputNames;
    m_previewTable->setHorizontalHeaderLabels(headers);

    for (int c = ds->inputCount; c < cols; ++c) {
        auto* h = m_previewTable->horizontalHeaderItem(c);
        if (h) { QFont f = h->font(); f.setBold(true); h->setFont(f); }
    }

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < ds->inputCount; ++c) {
            auto* it = new QTableWidgetItem(QString::number(ds->inputAt(r,c),'f',4));
            it->setTextAlignment(Qt::AlignRight|Qt::AlignVCenter);
            m_previewTable->setItem(r, c, it);
        }
        for (int c = 0; c < ds->outputCount; ++c) {
            auto* it = new QTableWidgetItem(QString::number(ds->outputAt(r,c),'f',4));
            it->setTextAlignment(Qt::AlignRight|Qt::AlignVCenter);
            it->setBackground(QColor(230,241,255,80));
            QFont f = it->font(); f.setBold(true); it->setFont(f);
            m_previewTable->setItem(r, ds->inputCount + c, it);
        }
    }
    m_tabs->setTabText(0, ds->sampleCount > kPreviewRows
        ? QString("Preview (first %1 rows)").arg(kPreviewRows)
        : QString("Preview (%1 rows)").arg(rows));
}

void DataPanel::populateStats() {
    if (!m_bundle.train) return;
    const auto* ds = m_bundle.train.get();
    const QStringList colHeaders = {"Column","Role","Min","Max","Mean","Std Dev","Median","Missing"};
    const int total = ds->inputCount + ds->outputCount;

    m_statsTable->clearContents();
    m_statsTable->setRowCount(total);
    m_statsTable->setColumnCount(colHeaders.size());
    m_statsTable->setHorizontalHeaderLabels(colHeaders);

    auto addRow = [&](int row, const QString& name, const QString& role, const ColumnStats& s) {
        auto cell = [&](int col, const QString& t, bool bold=false){
            auto* it = new QTableWidgetItem(t);
            it->setTextAlignment(Qt::AlignRight|Qt::AlignVCenter);
            if (bold) { QFont f=it->font(); f.setBold(true); it->setFont(f); }
            if (col==0||col==1) it->setTextAlignment(Qt::AlignLeft|Qt::AlignVCenter);
            m_statsTable->setItem(row, col, it);
        };
        cell(0,name,true); cell(1,role);
        cell(2,QString::number(s.min,'f',4));
        cell(3,QString::number(s.max,'f',4));
        cell(4,QString::number(s.mean,'f',4));
        cell(5,QString::number(s.stddev,'f',4));
        cell(6,QString::number(s.median,'f',4));
        cell(7,s.missing>0?QString::number(s.missing):"—");
        if (s.missing>0) m_statsTable->item(row,7)->setForeground(QColor("#c0392b"));
        if (role=="output") {
            for (int c=0;c<colHeaders.size();++c)
                if (m_statsTable->item(row,c))
                    m_statsTable->item(row,c)->setBackground(QColor(230,241,255,80));
        }
    };

    for (int i=0;i<ds->inputCount;++i)  addRow(i,                    ds->inputNames[i],  "input",  ds->inputStats[i]);
    for (int i=0;i<ds->outputCount;++i) addRow(ds->inputCount + i,   ds->outputNames[i], "output", ds->outputStats[i]);

    m_statsTable->horizontalHeader()->setSectionResizeMode(0,QHeaderView::ResizeToContents);
    m_statsTable->horizontalHeader()->setSectionResizeMode(1,QHeaderView::ResizeToContents);
    m_tabs->setTabText(1, QString("Statistics (%1 variables)").arg(total));
}

void DataPanel::setStatus(const QString& msg, bool error) {
    m_infoBar->setText("  " + msg);
    m_infoBar->setStyleSheet(
        QString("QLabel{background:%1;border-radius:4px;padding:0 10px;font-size:12px;color:%2;}")
            .arg(error?"#fff0f0":"palette(midlight)", error?"#c0392b":"palette(text)"));
}

void DataPanel::dragEnterEvent(QDragEnterEvent* e) {
    if (e->mimeData()->hasUrls()) e->acceptProposedAction();
}
void DataPanel::dropEvent(QDropEvent* e) {
    const auto urls = e->mimeData()->urls();
    if (!urls.isEmpty()) loadFile(urls.first().toLocalFile());
}

} // namespace NeuralStudio
