#include "GEStudioPanel.h"
#include "core/export/XlsxWriter.h"
#include "core/ge/Grammar.h"
#include "core/ge/NetSpec.h"
#include "core/ge/GEExperiment.h"
#include "GEExperimentDialog.h"
#include "GEModelPicker.h"
#include "core/ge/Baselines.h"
#include <QMenu>
#include <QTabBar>
#include <QTextBrowser>
#include <QApplication>
#include <QCheckBox>
#include <QChart>
#include <QChartView>
#include <QComboBox>
#include <QCoreApplication>
#include <QDateTime>
#include <QDesktopServices>
#include <QDir>
#include <QDoubleSpinBox>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QFileDialog>
#include <QFileInfo>
#include <QFontDatabase>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QLineSeries>
#include <QMessageBox>
#include <QMimeData>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QSet>
#include <QSettings>
#include <QTextStream>
#include <QFontMetrics>
#include <QGuiApplication>
#include <QCursor>
#include <QSpinBox>
#include <QTabWidget>
#include <QTableWidget>
#include <QThread>
#include <QToolButton>
#include <QUrl>
#include <QValueAxis>
#include <QVBoxLayout>
#include <algorithm>
#include <cmath>
#include <functional>

namespace NeuralStudio {

namespace {
const char* kAccent = "#FF7B72";
const QString kGeRest = QStringLiteral("ge2..K");   // pseudo id of the Models list: ge2 … geK

QPushButton* mkBtn(const QString& t, const QString& bg = QString()) {
    auto* b = new QPushButton(t);
    b->setMinimumHeight(28);
    if (!bg.isEmpty())
        b->setStyleSheet(QString("QPushButton{background:%1;color:white;border-radius:4px;"
            "font-weight:600;padding:0 12px;}QPushButton:disabled{background:palette(midlight);}").arg(bg));
    return b;
}
QSpinBox* mkSpin(int lo, int hi, int v) {
    auto* s = new QSpinBox; s->setRange(lo, hi); s->setValue(v); return s;
}
QDoubleSpinBox* mkDSpin(double lo, double hi, double v, double step, int dec = 2) {
    auto* s = new QDoubleSpinBox; s->setRange(lo, hi); s->setSingleStep(step);
    s->setDecimals(dec); s->setValue(v); return s;
}
QString fmtErr(double v, bool cls) {
    if (v < 0) return "-";
    return cls ? QString::number(v, 'f', 2) + " %" : QString::number(v, 'g', 5);
}
QTableWidgetItem* cell(const QString& t, bool right = false) {
    auto* it = new QTableWidgetItem(t);
    it->setFlags(it->flags() & ~Qt::ItemIsEditable);
    if (right) it->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);
    return it;
}
// A tab widget as tall as its current page (QTabWidget's size hint takes the tallest page;
// the minimum already follows the size policies set in onSetupTabChanged).
class CurrentPageTabs : public QTabWidget {
public:
    QSize sizeHint() const override { return fit(QTabWidget::sizeHint(), &QWidget::sizeHint); }
private:
    QSize fit(QSize all, QSize (QWidget::*hint)() const) const {
        QWidget* cur = currentWidget();
        if (!cur) return all;
        int tallest = 0;
        for (int i = 0; i < count(); ++i) tallest = std::max(tallest, (widget(i)->*hint)().height());
        all.setHeight(all.height() - tallest + (cur->*hint)().height());
        return all;
    }
};
// Column widths from the cells that are not part of a span (title and note rows span the whole table
// and would otherwise make the first column as wide as their text).
void fitColumnsIgnoringSpans(QTableWidget* t) {
    for (int c = 0; c < t->columnCount(); ++c) {
        int w = t->horizontalHeader()->isVisible() ? t->horizontalHeader()->sectionSizeHint(c) : 30;
        for (int r = 0; r < t->rowCount(); ++r) {
            QTableWidgetItem* it = t->item(r, c);
            if (!it || t->columnSpan(r, c) > 1) continue;
            QFontMetrics fm(it->font());
            w = std::max(w, fm.horizontalAdvance(it->text()) + 16);
        }
        t->setColumnWidth(c, std::min(w, 520));
    }
}
} // namespace

GEStudioPanel::GEStudioPanel(QWidget* parent) : QWidget(parent) {
    setAcceptDrops(true);
    buildUi();
}

// Closing the application while a run is active: stop it and wait for the thread,
// otherwise QThread would be destroyed while still running.
GEStudioPanel::~GEStudioPanel() {
    if (m_worker) m_worker->requestStop();
    if (m_batchRunner) m_batchRunner->requestStop();
    for (QThread* t : {m_thread.data(), m_batchThread.data()})
        if (t && t->isRunning()) { t->quit(); t->wait(); }
}

void GEStudioPanel::dragEnterEvent(QDragEnterEvent* e) {
    if (e->mimeData()->hasUrls()) e->acceptProposedAction();
}
void GEStudioPanel::dropEvent(QDropEvent* e) {
    QStringList p;
    for (const auto& u : e->mimeData()->urls()) { const QString s = u.toLocalFile(); if (!s.isEmpty()) p << s; }
    addToQueue(p);
}

// ── UI ────────────────────────────────────────────────────────────────────────
void GEStudioPanel::buildUi() {
    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(20, 14, 20, 14);
    root->setSpacing(8);

    auto* title = new QLabel("Grammatical Evolution Studio");
    { QFont f = title->font(); f.setPointSize(14); f.setBold(true); title->setFont(f); }
    root->addWidget(title);
    auto* sub = new QLabel(
        "One <b>BNF grammar</b> derives a whole network: <b>architecture</b>, per-layer <b>bound commands</b> "
        "(SHRINK / EXPAND …) and optional <b>constructed features</b>. The chromosome never encodes the "
        "weights: for every distinct phenotype an inner optimizer trains them <i>inside the intervals the "
        "grammar produced</i>. Genotype: classic codons (GE) or learned production probabilities (PGE). "
        "The final table compares the best phenotypes against a standard network at the same and at an "
        "equal total budget, on the companion test file.");
    sub->setWordWrap(true);
    sub->setStyleSheet("color:palette(mid);font-size:12px;");
    root->addWidget(sub);

    // Setup: a single run on the queued files, or a whole experiment (datasets × folds × models)
    m_setupTabs = new CurrentPageTabs;
    m_singlePage = new QWidget;
    {
        auto* sl = new QVBoxLayout(m_singlePage);
        sl->setContentsMargins(6, 6, 6, 6);
        auto* r1 = new QHBoxLayout; r1->setSpacing(10);
        r1->addWidget(buildQueueGroup(), 2);
        r1->addWidget(buildModelsGroup(), 3);
        r1->addWidget(buildGrammarGroup(), 4);
        sl->addLayout(r1);
        auto* r2 = new QHBoxLayout; r2->setSpacing(10);
        r2->addWidget(buildEvolutionGroup(), 3);
        r2->addWidget(buildTrainingGroup(), 2);
        r2->addWidget(buildFinalGroup(), 2);
        sl->addLayout(r2);
    }
    m_setupTabs->addTab(m_singlePage, "Single run (queued files)");

    // one Run / Stop / Export / results-folder row for both pages
    auto* runRow = new QHBoxLayout;
    m_runBtn = mkBtn("▶  Run", kAccent);
    m_runBtn->setMinimumWidth(170);
    m_stopBtn = mkBtn("■  Stop", "#8B3030");
    m_stopBtn->setEnabled(false);
    m_exportBtn = mkBtn("Export (.xlsx)");
    m_exportBtn->setEnabled(false);
    m_batchOpenBtn = mkBtn("Open results folder");
    runRow->addWidget(m_runBtn);
    runRow->addWidget(m_stopBtn);
    runRow->addWidget(m_exportBtn);
    runRow->addWidget(m_batchOpenBtn);
    m_bar = new QProgressBar; m_bar->setRange(0, 1000); m_bar->setValue(0); m_bar->setTextVisible(true);
    runRow->addWidget(m_bar, 1);
    m_statusLbl = new QLabel("Single run: add datasets (.train/.test pairs). Experiment: New / Open, Edit…, Run.");
    m_statusLbl->setStyleSheet("color:palette(mid);");
    QWidget* results = buildResultTabs();          // before the experiment page: it writes to m_batchView

    m_expPage = buildExperimentPage();
    m_setupTabs->addTab(m_expPage, "Experiment (datasets × folds × models)");
    root->addWidget(m_setupTabs);
    root->addLayout(runRow);
    root->addWidget(m_statusLbl);
    root->addWidget(results, 1);

    connect(m_runBtn, &QPushButton::clicked, this, &GEStudioPanel::onRunClicked);
    connect(m_stopBtn, &QPushButton::clicked, this, &GEStudioPanel::onStop);
    connect(m_exportBtn, &QPushButton::clicked, this, &GEStudioPanel::onExportClicked);
    connect(m_batchOpenBtn, &QPushButton::clicked, this, &GEStudioPanel::onBatchOpenResults);
    connect(m_setupTabs, &QTabWidget::currentChanged, this, &GEStudioPanel::onSetupTabChanged);
    {
        QSettings st("NeuralStudio", "NeuralStudio");
        m_setupTabs->setCurrentIndex(std::clamp(st.value("GEStudio/setupTab", 0).toInt(), 0, 1));
    }
    onSetupTabChanged(m_setupTabs->currentIndex());
    // the overall tables of the current experiment, if it has results
    if (!experimentPrefix().isEmpty() && QFileInfo::exists(experimentPrefix() + "_raw.csv"))
        loadOverallFromRaw(experimentPrefix() + "_raw.csv", true);
    onPresetChanged(0);
    onGenotypeChanged();
}

QGroupBox* GEStudioPanel::buildQueueGroup() {
    auto* g = new QGroupBox("Dataset Queue  (drag && drop files here)");
    auto* l = new QVBoxLayout(g);
    auto* br = new QHBoxLayout;
    m_addBtn = mkBtn("Add Files…");
    m_removeBtn = mkBtn("Remove");
    m_clearBtn = mkBtn("Clear", "#8B3030");
    br->addWidget(m_addBtn); br->addWidget(m_removeBtn); br->addWidget(m_clearBtn); br->addStretch();
    l->addLayout(br);
    m_queueTable = new QTableWidget(0, 1);
    m_queueTable->setHorizontalHeaderLabels({"File"});
    m_queueTable->horizontalHeader()->setStretchLastSection(true);
    m_queueTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_queueTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_queueTable->setMinimumHeight(150);
    l->addWidget(m_queueTable);
    connect(m_addBtn, &QPushButton::clicked, this, &GEStudioPanel::onAddFiles);
    connect(m_removeBtn, &QPushButton::clicked, this, &GEStudioPanel::onRemoveSelected);
    connect(m_clearBtn, &QPushButton::clicked, this, &GEStudioPanel::onClearQueue);
    return g;
}

QGroupBox* GEStudioPanel::buildGrammarGroup() {
    auto* g = new QGroupBox("BNF Grammar");
    auto* l = new QVBoxLayout(g);
    auto* top = new QHBoxLayout;
    m_presetCombo = new QComboBox;
    m_presetCombo->addItems({"Unified: architecture + bounds + features",
                             "Architecture + bounds", "Architecture only", "Custom",
                             "Unified + periodic input embedding"});
    m_maxLayersSpin = mkSpin(1, 8, 4);
    m_maxFeaturesSpin = mkSpin(0, 8, 3);
    m_featSizeSpin = mkSpin(3, 60, 15);
    m_featSizeSpin->setToolTip("Max operations + operands of one feature after constant folding");
    auto* vb = mkBtn("Validate");
    m_presetCombo->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
    m_presetCombo->setMinimumContentsLength(22);
    top->addWidget(new QLabel("Preset:")); top->addWidget(m_presetCombo, 1);
    top->addWidget(vb);
    l->addLayout(top);
    auto* lim = new QHBoxLayout;
    lim->addWidget(new QLabel("max layers")); lim->addWidget(m_maxLayersSpin);
    lim->addSpacing(8);
    lim->addWidget(new QLabel("max features")); lim->addWidget(m_maxFeaturesSpin);
    lim->addSpacing(8);
    lim->addWidget(new QLabel("feature size")); lim->addWidget(m_featSizeSpin);
    lim->addStretch();
    l->addLayout(lim);
    m_grammarEdit = new QPlainTextEdit;
    m_grammarEdit->setFont(QFontDatabase::systemFont(QFontDatabase::FixedFont));
    m_grammarEdit->setMinimumHeight(150);
    l->addWidget(m_grammarEdit);
    m_grammarStatus = new QLabel;
    m_grammarStatus->setWordWrap(true);
    l->addWidget(m_grammarStatus);
    auto* help = new QLabel(
        "The interpreter reads meaning from rule <b>names</b>: &lt;layer&gt; (first &lt;size&gt;, &lt;act&gt;, "
        "two &lt;cmd&gt;), &lt;out&gt; (two &lt;cmd&gt;), &lt;feat&gt; (its &lt;expr&gt;). Commands: NOTHING, SHRINK "
        "(×0.5), EXPAND (×1.5), SHRINK_EXPAND, EXPAND_SHRINK (×0.75), SHRINK_SHRINK (×0.25), "
        "EXPAND_EXPAND (×2.25) or a plain number. <code>@VARS@</code> expands to x1 | … | xN per dataset.");
    help->setWordWrap(true);
    help->setStyleSheet("color:palette(mid);font-size:11px;");
    l->addWidget(help);
    connect(m_presetCombo, &QComboBox::currentIndexChanged, this, &GEStudioPanel::onPresetChanged);
    connect(vb, &QPushButton::clicked, this, &GEStudioPanel::onValidateGrammar);
    connect(m_grammarEdit, &QPlainTextEdit::textChanged, this, [this] {
        if (m_grammarEdit->hasFocus() && m_presetCombo->currentIndex() != 3)
            m_presetCombo->setCurrentIndex(3);
    });
    return g;
}

QGroupBox* GEStudioPanel::buildEvolutionGroup() {
    m_evoGroup = new QGroupBox("Outer evolution (GE)");
    auto* l = new QVBoxLayout(m_evoGroup);
    auto* f = new QFormLayout; f->setLabelAlignment(Qt::AlignRight);
    m_genotypeCombo = new QComboBox;
    m_genotypeCombo->addItem("GE — integer codons", "ge");
    m_genotypeCombo->addItem("PGE — learned production probabilities", "pge");
    m_popSpin = mkSpin(8, 2000, 40);
    m_gensSpin = mkSpin(1, 5000, 40);
    m_replaceSpin = mkDSpin(0.05, 1.0, 0.30, 0.05);
    f->addRow("Genotype", m_genotypeCombo);
    f->addRow("Population", m_popSpin);
    f->addRow("Generations", m_gensSpin);
    f->addRow("Replaced / generation", m_replaceSpin);
    l->addLayout(f);

    m_codonBox = new QWidget;
    auto* cf = new QFormLayout(m_codonBox); cf->setContentsMargins(0, 0, 0, 0);
    m_mutSpin = mkDSpin(0, 1, 0.05, 0.01);
    m_tournSpin = mkSpin(2, 20, 4);
    m_genomeSpin = mkSpin(20, 1000, 120);
    m_crossCombo = new QComboBox;
    m_crossCombo->addItem("One-point", "onepoint");
    m_crossCombo->addItem("One-point + Local Crossover", "local");
    m_pcrSpin = mkDSpin(0, 1, 0.10, 0.05);
    m_ncrSpin = mkSpin(1, 20, 3);
    cf->addRow("Mutation rate", m_mutSpin);
    cf->addRow("Tournament size", m_tournSpin);
    cf->addRow("Chromosome length", m_genomeSpin);
    cf->addRow("Crossover", m_crossCombo);
    cf->addRow("Local: rate pcr / partners Ncr", [&] {
        auto* w = new QWidget; auto* h = new QHBoxLayout(w); h->setContentsMargins(0, 0, 0, 0);
        h->addWidget(m_pcrSpin); h->addWidget(m_ncrSpin); return w; }());
    l->addWidget(m_codonBox);

    m_pgeBox = new QWidget;
    auto* pf = new QFormLayout(m_pgeBox); pf->setContentsMargins(0, 0, 0, 0);
    m_alphaSpin = mkDSpin(0.01, 1, 0.30, 0.05);
    m_eliteSpin = mkDSpin(0.05, 1, 0.25, 0.05);
    m_floorSpin = mkDSpin(0, 0.2, 0.02, 0.01);
    m_transferChk = new QCheckBox("carry probabilities to the next dataset");
    pf->addRow("Learning rate α", m_alphaSpin);
    pf->addRow("Elite fraction", m_eliteSpin);
    pf->addRow("Probability floor", m_floorSpin);
    pf->addRow("", m_transferChk);
    l->addWidget(m_pgeBox);

    auto* sf = new QFormLayout;
    m_patienceSpin = mkSpin(0, 500, 8);
    m_patienceSpin->setSpecialValueText("off");
    m_patienceSpin->setToolTip("Stop when the best fitness has not improved for this many generations");
    m_seedSpin = mkSpin(0, 2000000000, 0);
    m_seedSpin->setSpecialValueText("random");
    sf->addRow("Early stop (patience)", m_patienceSpin);
    sf->addRow("Seed", m_seedSpin);
    l->addLayout(sf);
    connect(m_genotypeCombo, &QComboBox::currentIndexChanged, this, &GEStudioPanel::onGenotypeChanged);
    return m_evoGroup;
}

QGroupBox* GEStudioPanel::buildTrainingGroup() {
    m_trainGroup = new QGroupBox("Inner training of each phenotype");
    auto* f = new QFormLayout(m_trainGroup);
    m_innerCombo = new QComboBox;
    for (const char* n : {"de", "pso", "cmaes", "jso", "ga", "lmcmaes", "acor", "clpso", "mlshaderl", "arq3", "sa"})
        m_innerCombo->addItem(QString(n).toUpper(), n);
    m_metricCombo = new QComboBox;
    m_metricCombo->addItem("Validation loss (continuous)", "loss");
    m_metricCombo->addItem("Validation error % (can saturate)", "error");
    m_innerPopSpin = mkSpin(8, 2000, 30);
    m_innerItersSpin = mkSpin(1, 20000, 60);
    m_boundSpin = mkDSpin(0.1, 1000, 10, 1, 1);
    m_valSpin = mkDSpin(0, 0.6, 0.2, 0.05);
    m_parsSpin = mkDSpin(0, 10, 0, 0.1);
    f->addRow("Optimizer", m_innerCombo);
    f->addRow("Population", m_innerPopSpin);
    f->addRow("Iterations", m_innerItersSpin);
    f->addRow("Base bound ±B", m_boundSpin);
    f->addRow("Fitness", m_metricCombo);
    f->addRow("Validation split", m_valSpin);
    f->addRow("Parsimony λ", m_parsSpin);
    auto* note = new QLabel("fitness = validation loss (or error %) × (1 + λ·params/1000). The base bound is what "
                            "SHRINK / EXPAND scale; the chromosome never sees the weights.");
    note->setWordWrap(true); note->setStyleSheet("color:palette(mid);font-size:11px;");
    f->addRow(note);
    return m_trainGroup;
}

// Models of a single run: rows of the GE comparison (equal-budget mode) and reference models.
QGroupBox* GEStudioPanel::buildModelsGroup() {
    auto* g = new QGroupBox("Models  (click to select)");
    auto* l = new QVBoxLayout(g);
    m_singleModels = new GEModelPicker;
    m_singleModels->setCompact(true);
    QVector<GEModelPicker::Entry> entries;
    for (const QString& id : {"standard", "standard_eq", "grid_eq", "ge1"}) entries += GEModelPicker::catalogEntries({id});
    entries.push_back({kGeRest, "GE #2 … #K", true, "The other finalists (Top-K of the Final comparison), each trained with the final budget"});
    entries += GEModelPicker::catalogEntries({"ge1_nobounds", "ge1_eq", "ge1_nobounds_eq"});
    entries += GEModelPicker::catalogEntries(geBaselineIds());
    m_singleModels->setEntries(entries);
    const QStringList def{"standard", "standard_eq", "grid_eq", "ge1", kGeRest, "ge1_nobounds", "ge1_eq", "ge1_nobounds_eq"};
    QSettings st("NeuralStudio", "NeuralStudio");
    m_singleModels->setChecked(st.contains("GEStudio/singleModels") ? st.value("GEStudio/singleModels").toStringList() : def);
    m_singleModels->setToolTip("Reference models (kNN, SVM, …) are trained on the same train file and measured on the same test file.");
    connect(m_singleModels, &GEModelPicker::changed, this, [this] {
        QSettings("NeuralStudio", "NeuralStudio").setValue("GEStudio/singleModels", m_singleModels->checked());
    });
    l->addWidget(m_singleModels);
    return g;
}

QGroupBox* GEStudioPanel::buildFinalGroup() {
    m_finalGroup = new QGroupBox("Final comparison");
    auto* f = new QFormLayout(m_finalGroup);
    m_topKSpin = mkSpin(1, 20, 5);
    m_finalPopSpin = mkSpin(8, 5000, 80);
    m_finalItersSpin = mkSpin(1, 50000, 200);
    m_polishChk = new QCheckBox("L-BFGS polish, projected to bounds (all rows)"); m_polishChk->setChecked(true);
    m_baseHiddenEdit = new QLineEdit("16");
    m_baseActCombo = new QComboBox; m_baseActCombo->addItems({"relu", "tanh", "sigmoid"});
    f->addRow("Top-K to retrain", m_topKSpin);
    f->addRow("Population", m_finalPopSpin);
    f->addRow("Iterations", m_finalItersSpin);
    f->addRow("", m_polishChk);
    f->addRow("Baseline hidden", m_baseHiddenEdit);
    f->addRow("Baseline activation", m_baseActCombo);
    return m_finalGroup;
}

QWidget* GEStudioPanel::buildResultTabs() {
    m_tabs = new QTabWidget;
    m_chartView = new QChartView(new QChart);
    m_chartView->setRenderHint(QPainter::Antialiasing);
    m_chartView->setMinimumHeight(260);
    m_tabs->addTab(m_chartView, "Convergence");

    m_cmpTable = new QTableWidget(0, 8);
    m_cmpTable->setHorizontalHeaderLabels({"Dataset", "Method", "Train err", "Test err", "Val err", "Params", "Evals", "Time (s)"});
    m_cmpTable->horizontalHeader()->setStretchLastSection(true);
    m_cmpTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_tabs->addTab(m_cmpTable, "Comparison");

    m_topTable = new QTableWidget(0, 8);
    m_topTable->setHorizontalHeaderLabels({"Dataset", "Rank", "Fitness", "Train err", "Val err", "Params", "Inputs", "Phenotype"});
    m_topTable->horizontalHeader()->setStretchLastSection(true);
    m_topTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_tabs->addTab(m_topTable, "Phenotypes");

    m_probsView = new QPlainTextEdit; m_probsView->setReadOnly(true);
    m_probsView->setFont(QFontDatabase::systemFont(QFontDatabase::FixedFont));
    m_tabs->addTab(m_probsView, "Grammar statistics");

    // ONE log for both pages: the timestamped events of a single run / experiment and the experiment's
    // own report (progress lines, per-dataset tables, notes) used to be split over the tabs "Log" and
    // "Batch"; m_batchView is the same widget, so every writer ends up here.
    m_logView = new QPlainTextEdit; m_logView->setReadOnly(true);
    m_logView->setFont(QFontDatabase::systemFont(QFontDatabase::FixedFont));
    m_logView->setLineWrapMode(QPlainTextEdit::NoWrap);
    m_logView->setMaximumBlockCount(20000);
    m_batchView = m_logView;
    m_tabs->addTab(m_logView, "Log");

    for (QTableWidget** t : {&m_clsTable, &m_regTable}) {
        *t = new QTableWidget;
        (*t)->setEditTriggers(QAbstractItemView::NoEditTriggers);
        (*t)->horizontalHeader()->setVisible(false);
        (*t)->verticalHeader()->setVisible(false);
        (*t)->setSelectionMode(QAbstractItemView::ContiguousSelection);
    }
    m_tabs->addTab(m_clsTable, "Classification");
    m_tabs->addTab(m_regTable, "Regression");
    m_hypTable = new QTableWidget;
    m_hypTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_hypTable->verticalHeader()->setVisible(false);
    m_hypTable->setSelectionMode(QAbstractItemView::ContiguousSelection);
    m_tabs->addTab(m_hypTable, "Hypotheses");
    m_tabs->setMinimumHeight(300);
    return m_tabs;
}

// ── Queue ─────────────────────────────────────────────────────────────────────
void GEStudioPanel::addToQueue(const QStringList& paths) {
    if (m_running) return;
    for (const QString& p : paths)
        if (!m_queue.contains(p) && QFileInfo::exists(p)) m_queue << p;
    refreshQueueTable();
}
void GEStudioPanel::setQueuePaths(const QStringList& paths) {
    if (m_running) return;
    m_queue.clear();
    addToQueue(paths);
}
void GEStudioPanel::refreshQueueTable() {
    m_queueTable->setRowCount(m_queue.size());
    for (int i = 0; i < m_queue.size(); ++i)
        m_queueTable->setItem(i, 0, cell(QFileInfo(m_queue[i]).fileName()));
}
void GEStudioPanel::onAddFiles() {
    addToQueue(QFileDialog::getOpenFileNames(this, "Add datasets", {},
        "Datasets (*.train *.test *.data *.csv *.tsv *.txt);;All files (*)"));
}
void GEStudioPanel::onRemoveSelected() {
    if (m_running) return;
    QList<int> rows;
    for (const auto& idx : m_queueTable->selectionModel()->selectedRows()) rows << idx.row();
    std::sort(rows.begin(), rows.end(), std::greater<int>());
    for (int r : rows) if (r >= 0 && r < m_queue.size()) m_queue.removeAt(r);
    refreshQueueTable();
}
void GEStudioPanel::onClearQueue() {
    if (m_running) return;
    m_queue.clear();
    refreshQueueTable();
}

// ── Grammar ───────────────────────────────────────────────────────────────────
void GEStudioPanel::onPresetChanged(int idx) {
    if (idx == 3) return;                       // Custom: leave the text alone
    ge::GrammarPreset p = idx == 1 ? ge::GrammarPreset::ArchAndBounds
                        : idx == 2 ? ge::GrammarPreset::ArchOnly
                        : idx == 4 ? ge::GrammarPreset::UnifiedEmbedding
                                   : ge::GrammarPreset::Unified;
    QSignalBlocker b(m_grammarEdit);
    m_grammarEdit->setPlainText(QString::fromStdString(ge::presetGrammar(p)));
    m_maxFeaturesSpin->setEnabled(idx == 0 || idx == 4);
    onValidateGrammar();
}

void GEStudioPanel::onValidateGrammar() {
    try {
        const std::string txt = ge::substituteVars(m_grammarEdit->toPlainText().toStdString(), 4);
        const ge::Grammar g = ge::Grammar::parse(txt);
        QStringList warn;
        if (g.ntIndex("layer") < 0) warn << "no &lt;layer&gt; rule: networks will be invalid";
        if (g.ntIndex("size") < 0 || g.ntIndex("act") < 0) warn << "&lt;size&gt; or &lt;act&gt; missing";
        m_grammarStatus->setStyleSheet(warn.isEmpty() ? "color:#3FB950;" : "color:#E3B341;");
        m_grammarStatus->setText(QString("OK — %1 rules, %2 productions%3")
            .arg(g.rules().size()).arg(g.totalProductions())
            .arg(warn.isEmpty() ? QString() : "  ⚠ " + warn.join("; ")));
    } catch (const std::exception& ex) {
        m_grammarStatus->setStyleSheet("color:#F85149;");
        m_grammarStatus->setText(QString("Grammar error: ") + QString::fromStdString(ex.what()));
    }
}

void GEStudioPanel::onGenotypeChanged() {
    const bool pge = m_genotypeCombo->currentData().toString() == "pge";
    m_codonBox->setVisible(!pge);
    m_pgeBox->setVisible(pge);
}

// ── Run ───────────────────────────────────────────────────────────────────────
GEStudioConfig GEStudioPanel::collectConfig() const {
    GEStudioConfig c;
    c.datasetPaths = m_queue;
    c.grammarText = m_grammarEdit->toPlainText();
    c.maxLayers = m_maxLayersSpin->value();
    c.maxFeatures = m_presetCombo->currentIndex() == 0 || m_presetCombo->currentIndex() == 3 || m_presetCombo->currentIndex() == 4
                    ? m_maxFeaturesSpin->value() : 0;
    c.genotype = m_genotypeCombo->currentData().toString();
    c.pop = m_popSpin->value();
    c.gens = m_gensSpin->value();
    c.replaceRate = m_replaceSpin->value();
    c.mutationRate = m_mutSpin->value();
    c.tournament = m_tournSpin->value();
    c.crossover = m_crossCombo->currentData().toString();
    c.localRate = m_pcrSpin->value();
    c.localNcr = m_ncrSpin->value();
    c.genomeLen = m_genomeSpin->value();
    c.pgeAlpha = m_alphaSpin->value();
    c.pgeElite = m_eliteSpin->value();
    c.pgeFloor = m_floorSpin->value();
    c.transferPrior = m_transferChk->isChecked();
    c.seed = static_cast<quint32>(m_seedSpin->value());
    c.baseBound = m_boundSpin->value();
    c.innerOpt = m_innerCombo->currentData().toString();
    c.innerPop = m_innerPopSpin->value();
    c.innerIters = m_innerItersSpin->value();
    c.valSplit = m_valSpin->value();
    c.parsimony = m_parsSpin->value();
    c.fitnessMetric = m_metricCombo->currentData().toString();
    c.patience = m_patienceSpin->value();
    c.maxFeatureSize = m_featSizeSpin->value();
    c.topK = m_topKSpin->value();
    c.finalPop = m_finalPopSpin->value();
    c.finalIters = m_finalItersSpin->value();
    c.finalPolish = m_polishChk->isChecked();
    // models chosen on the Models list: GE rows of the comparison + reference models
    const QStringList picked = m_singleModels->checked();
    const QStringList ref = geBaselineIds();
    QStringList geRows, refRows;
    for (const QString& id : picked) {
        if (ref.contains(id)) refRows << id;
        else if (id == kGeRest) for (int k = 2; k <= c.topK; ++k) geRows << QString("ge%1").arg(k);
        else geRows << id;
    }
    c.equalBudgetBaseline = geRows.contains("standard_eq") || geRows.contains("ge1_eq") || geRows.contains("ge1_nobounds_eq");
    c.gridBaseline = geRows.contains("grid_eq");
    if (geRows.isEmpty()) { c.mode = "baselines"; c.methods = refRows; }
    else c.methods = geRows + refRows;
    c.baselineHidden = m_baseHiddenEdit->text();
    c.baselineAct = m_baseActCombo->currentText();
    return c;
}

void GEStudioPanel::setRunning(bool r) {
    m_running = r;
    m_runBtn->setEnabled(!r);
    m_stopBtn->setEnabled(r);
    for (QWidget* w : {static_cast<QWidget*>(m_evoGroup), static_cast<QWidget*>(m_trainGroup),
                       static_cast<QWidget*>(m_finalGroup), static_cast<QWidget*>(m_addBtn),
                       static_cast<QWidget*>(m_removeBtn), static_cast<QWidget*>(m_clearBtn),
                       static_cast<QWidget*>(m_presetCombo), static_cast<QWidget*>(m_grammarEdit),
                       static_cast<QWidget*>(m_maxLayersSpin), static_cast<QWidget*>(m_maxFeaturesSpin),
                       static_cast<QWidget*>(m_featSizeSpin), static_cast<QWidget*>(m_batchCmdEdit),
                       static_cast<QWidget*>(m_batchDirEdit), static_cast<QWidget*>(m_batchBrowseBtn),
                       static_cast<QWidget*>(m_batchCheckBtn), static_cast<QWidget*>(m_newBtn),
                       static_cast<QWidget*>(m_openBtn), static_cast<QWidget*>(m_editBtn),
                       static_cast<QWidget*>(m_saveBtn), static_cast<QWidget*>(m_saveAsBtn),
                       static_cast<QWidget*>(m_singleModels), static_cast<QWidget*>(m_expModels)})
        if (w) w->setEnabled(!r);
    // the page that is running cannot be left; the other page cannot start anything anyway
    if (m_setupTabs) m_setupTabs->tabBar()->setEnabled(!r);
    updateActionStates();
    emit runningStateChanged(r);
}

void GEStudioPanel::log(const QString& line) {
    m_logView->appendPlainText(QDateTime::currentDateTime().toString("HH:mm:ss  ") + line);
}

void GEStudioPanel::onRun() {
    if (m_running) return;
    if (m_queue.isEmpty()) {
        QMessageBox::information(this, "GE Studio", "Add at least one dataset first.");
        return;
    }
    if (m_singleModels->checked().isEmpty()) {
        QMessageBox::information(this, "GE Studio", "Select at least one model in the Models list.");
        return;
    }
    if (collectConfig().mode == "ge") {
        onValidateGrammar();
        if (m_grammarStatus->text().startsWith("Grammar error")) {
            QMessageBox::warning(this, "GE Studio", "Fix the grammar first.");
            return;
        }
    }
    m_results.clear();
    m_cmpTable->setRowCount(0);
    m_topTable->setRowCount(0);
    m_probsView->clear();
    m_logView->clear();
    rebuildChart();
    m_exportBtn->setEnabled(false);
    m_dsIdx = 0; m_totalDs = m_queue.size(); m_inFinal = false; m_lastGenFrac = 0;
    m_bar->setValue(0);

    auto* worker = new GEStudioWorker;
    worker->setConfig(collectConfig());
    m_worker = worker;
    m_thread = new QThread(this);
    worker->moveToThread(m_thread);
    connect(m_thread, &QThread::started, worker, &GEStudioWorker::run);
    connect(worker, &GEStudioWorker::datasetStarted, this, &GEStudioPanel::onDatasetStarted);
    connect(worker, &GEStudioWorker::generationDone, this, &GEStudioPanel::onGenerationDone);
    connect(worker, &GEStudioWorker::finalStageStarted, this, &GEStudioPanel::onFinalStageStarted);
    connect(worker, &GEStudioWorker::finalRowDone, this, &GEStudioPanel::onFinalRowDone);
    connect(worker, &GEStudioWorker::datasetFinished, this, &GEStudioPanel::onDatasetFinished);
    connect(worker, &GEStudioWorker::allDone, this, &GEStudioPanel::onAllDone);
    connect(worker, &GEStudioWorker::errorOccurred, this, &GEStudioPanel::onError);
    connect(m_stopBtn, &QPushButton::clicked, worker, &GEStudioWorker::requestStop, Qt::DirectConnection);
    connect(worker, &GEStudioWorker::allDone, m_thread, &QThread::quit);
    connect(worker, &GEStudioWorker::errorOccurred, m_thread, &QThread::quit);
    connect(m_thread, &QThread::finished, worker, &QObject::deleteLater);
    connect(m_thread, &QThread::finished, m_thread, &QObject::deleteLater);
    setRunning(true);
    log("Run started.");
    m_thread->start();
}

void GEStudioPanel::onStop() {
    if (!m_running) return;
    if (m_batchMode) {
        m_statusLbl->setText("Stopping the experiment… the unfinished run is discarded (Run again continues with the missing ones).");
    } else {
        m_statusLbl->setText("Stopping after the current network…");
    }
    log("Stop requested.");
}

void GEStudioPanel::updateProgress(double within) {
    const double v = (m_dsIdx + std::min(1.0, std::max(0.0, within))) / std::max(1, m_totalDs);
    m_bar->setValue(static_cast<int>(v * 1000));
}

void GEStudioPanel::onDatasetStarted(int dsIdx, int totalDs, QString dsName, QString info) {
    m_dsIdx = dsIdx; m_totalDs = totalDs; m_inFinal = false;
    m_statusLbl->setText(QString("[%1/%2] %3 — %4").arg(dsIdx + 1).arg(totalDs).arg(dsName, info));
    log(QString("Dataset %1: %2").arg(dsName, info));
}

void GEStudioPanel::onGenerationDone(int gen, int totalGens, double best, QString desc,
                                     double mean, double div, int uniq, int hits, qint64 calls) {
    updateProgress(0.75 * gen / std::max(1, totalGens));
    m_statusLbl->setText(QString("[%1/%2] generation %3/%4 — best %5 — %6 distinct networks trained, "
                                 "%7 cache hits, %8 loss evaluations")
        .arg(m_dsIdx + 1).arg(m_totalDs).arg(gen).arg(totalGens)
        .arg(best, 0, 'f', 3).arg(uniq).arg(hits).arg(calls));
    log(QString("gen %1/%2  best=%3  mean=%4  diversity=%5  %6")
        .arg(gen).arg(totalGens).arg(best, 0, 'f', 3).arg(mean, 0, 'f', 2).arg(div, 0, 'f', 2).arg(desc));
}

void GEStudioPanel::onFinalStageStarted(int totalRows) {
    m_inFinal = true; m_finalRows = std::max(1, totalRows); m_finalDone = 0;
    m_statusLbl->setText(QString("[%1/%2] final comparison (%3 networks)…")
                         .arg(m_dsIdx + 1).arg(m_totalDs).arg(totalRows));
}
void GEStudioPanel::onFinalRowDone(int row, int totalRows, QString label) {
    m_finalDone = row;
    updateProgress(0.75 + 0.25 * row / std::max(1, totalRows));
    log(QString("final %1/%2: %3").arg(row).arg(totalRows).arg(label));
}

void GEStudioPanel::onDatasetFinished(GEDatasetResult r) {
    // reference models: "knn (k=7)" -> "kNN (k=7)"; they are not GE rows, but GNN uses the BNF grammar
    const QStringList refIds = geBaselineIds();
    for (GEFinalRow& x : r.rows)
        if (refIds.contains(x.id)) {
            const GEMethodInfo mi = geMethodInfo(x.id);
            if (x.label.startsWith(x.id)) x.label = mi.label + x.label.mid(x.id.size());
            x.isGE = mi.bnf;
        }
    m_results.append(r);
    updateProgress(1.0);
    if (r.gensRun < r.gensPlanned)
        log(QString("Dataset %1: stopped early after %2 of %3 generations (no improvement).").arg(r.dsName).arg(r.gensRun).arg(r.gensPlanned));
    log(QString("Dataset %1 done: %2 distinct networks, %3 cache hits, %4 search evaluations, %5 s")
        .arg(r.dsName).arg(r.uniqueEvals).arg(r.cacheHits).arg(r.searchCalls).arg(r.searchSeconds, 0, 'f', 1));

    for (const GEFinalRow& x : r.rows) {
        const int row = m_cmpTable->rowCount();
        m_cmpTable->insertRow(row);
        m_cmpTable->setItem(row, 0, cell(r.dsName));
        m_cmpTable->setItem(row, 1, cell(x.label));
        m_cmpTable->setItem(row, 2, cell(fmtErr(x.trainErr, r.isCls), true));
        m_cmpTable->setItem(row, 3, cell(fmtErr(x.testErr, r.isCls), true));
        m_cmpTable->setItem(row, 4, cell(fmtErr(x.valErr, r.isCls), true));
        m_cmpTable->setItem(row, 5, cell(QString::number(x.params), true));
        m_cmpTable->setItem(row, 6, cell(QString::number(x.evals), true));
        m_cmpTable->setItem(row, 7, cell(QString::number(x.seconds, 'f', 1), true));
        if (x.isGE)
            for (int c = 0; c < 8; ++c)
                m_cmpTable->item(row, c)->setForeground(QColor(kAccent));
    }
    // cost line: what the search itself spent (none when only reference models ran)
    if (r.searchCalls > 0 || r.uniqueEvals > 0) {
        const int row = m_cmpTable->rowCount();
        m_cmpTable->insertRow(row);
        m_cmpTable->setItem(row, 0, cell(r.dsName));
        m_cmpTable->setItem(row, 1, cell("GE search cost (all generations)"));
        for (int c = 2; c < 6; ++c) m_cmpTable->setItem(row, c, cell("", true));
        m_cmpTable->setItem(row, 6, cell(QString::number(r.searchCalls), true));
        m_cmpTable->setItem(row, 7, cell(QString::number(r.searchSeconds, 'f', 1), true));
    }
    for (const GEPhenotype& p : r.top) {
        const int row = m_topTable->rowCount();
        m_topTable->insertRow(row);
        m_topTable->setItem(row, 0, cell(r.dsName));
        m_topTable->setItem(row, 1, cell(QString::number(p.rank), true));
        m_topTable->setItem(row, 2, cell(QString::number(p.fitness, 'g', 5), true));
        m_topTable->setItem(row, 3, cell(fmtErr(p.trainErr, r.isCls), true));
        m_topTable->setItem(row, 4, cell(fmtErr(p.valErr, r.isCls), true));
        m_topTable->setItem(row, 5, cell(QString::number(p.params), true));
        m_topTable->setItem(row, 6, cell(QString::number(p.inputs), true));
        m_topTable->setItem(row, 7, cell(p.desc));
    }
    m_cmpTable->resizeColumnsToContents();
    m_topTable->resizeColumnsToContents();
    m_probsView->appendPlainText(QString("=== %1 ===\nBest program: %2\n\n%3\n")
                                 .arg(r.dsName, r.bestProgram, r.probsReport));
    rebuildChart();
    m_exportBtn->setEnabled(true);
}

void GEStudioPanel::rebuildChart() {
    auto* chart = new QChart;
    chart->setTitle("Best validation fitness per generation");
    auto* ax = new QValueAxis; ax->setTitleText("generation"); ax->setLabelFormat("%d");
    auto* ay = new QValueAxis; ay->setTitleText("fitness (lower is better)");
    chart->addAxis(ax, Qt::AlignBottom);
    chart->addAxis(ay, Qt::AlignLeft);
    double ymax = 0; int xmax = 1;
    for (const GEDatasetResult& r : m_results) {
        auto* s = new QLineSeries; s->setName(r.dsName);
        for (int i = 0; i < r.convergence.size(); ++i) {
            s->append(i, r.convergence[i]);
            ymax = std::max(ymax, r.convergence[i]);
        }
        xmax = std::max(xmax, static_cast<int>(r.convergence.size()) - 1);
        chart->addSeries(s);
        s->attachAxis(ax); s->attachAxis(ay);
    }
    ax->setRange(0, xmax);
    ax->setTickCount(std::min(xmax + 1, 11));
    ay->setRange(0, ymax > 0 ? ymax * 1.05 : 1);
    chart->legend()->setVisible(!m_results.isEmpty());
    auto* old = m_chartView->chart();
    m_chartView->setChart(chart);
    delete old;
}

void GEStudioPanel::onAllDone(std::shared_ptr<NeuralNetwork> net, std::shared_ptr<Dataset> ds) {
    setRunning(false);
    m_thread = nullptr;
    m_bar->setValue(1000);
    m_statusLbl->setText(QString("Finished — %1 dataset(s).").arg(m_results.size()));
    log("Run finished.");
    if (net && ds) emit trainingCompleted(net, ds);
}

void GEStudioPanel::onError(QString msg) {
    setRunning(false);
    m_thread = nullptr;
    m_statusLbl->setText("Error: " + msg);
    log("ERROR: " + msg);
    QMessageBox::critical(this, "GE Studio", msg);
}

// ── Export ────────────────────────────────────────────────────────────────────
void GEStudioPanel::onExportXlsx() {
    if (m_results.isEmpty()) return;
    const QString path = QFileDialog::getSaveFileName(this, "Export results", "ge_studio_results.xlsx",
                                                      XlsxWriter::fileFilter());
    if (path.isEmpty()) return;
    try {
        XlsxWriter wb;
        auto* cmp = wb.addSheet("Comparison");
        cmp->setHeader({"Dataset", "Method", "Task", "Train err", "Test err", "Val err", "Params", "Evals", "Time (s)"});
        auto* top = wb.addSheet("Phenotypes");
        top->setHeader({"Dataset", "Rank", "Fitness", "Train err", "Val err", "Params", "Inputs", "Phenotype"});
        auto* conv = wb.addSheet("Convergence");
        conv->setHeader({"Dataset", "Generation", "Best fitness"});
        auto* stat = wb.addSheet("Grammar statistics");
        stat->setHeader({"Dataset", "Line"});
        auto* sum = wb.addSheet("Search cost");
        sum->setHeader({"Dataset", "Distinct networks", "Cache hits", "Loss evaluations", "Search time (s)"});
        for (const GEDatasetResult& r : m_results) {
            const QString task = r.isCls ? "classification (error %)" : "regression (MSE)";
            for (const GEFinalRow& x : r.rows)
                cmp->addRow({r.dsName, x.label, task, x.trainErr, x.testErr < 0 ? XlsxCell() : XlsxCell(x.testErr),
                             x.valErr < 0 ? XlsxCell() : XlsxCell(x.valErr), x.params, x.evals, x.seconds});
            for (const GEPhenotype& p : r.top)
                top->addRow({r.dsName, p.rank, p.fitness, p.trainErr, p.valErr < 0 ? XlsxCell() : XlsxCell(p.valErr),
                             p.params, p.inputs, p.desc});
            for (int i = 0; i < r.convergence.size(); ++i) conv->addRow({r.dsName, i, r.convergence[i]});
            for (const QString& line : r.probsReport.split('\n')) stat->addRow({r.dsName, line});
            sum->addRow({r.dsName, r.uniqueEvals, r.cacheHits, r.searchCalls, r.searchSeconds});
        }
        wb.save(path);
        m_statusLbl->setText("Exported: " + path);
    } catch (const std::exception& ex) {
        QMessageBox::critical(this, "Export", QString::fromStdString(ex.what()));
    }
}

// ── Experiment page ───────────────────────────────────────────────────────────
namespace {
QSettings batchSettings() { return QSettings("NeuralStudio", "NeuralStudio"); }
QString cmdPath(const QString& p) {
    const QString n = QDir::toNativeSeparators(p);
    return n.contains(' ') ? "\"" + n + "\"" : n;
}
GEExperiment emptyExperiment() {
    GEExperiment e = geBenchmarkExperiment();
    e.name = "experiment";
    e.description.clear();
    e.output = "results/experiment";
    e.classification.clear();
    e.regression.clear();
    return e;
}
} // namespace

QWidget* GEStudioPanel::buildExperimentPage() {
    auto* page = new QWidget;
    auto* l = new QVBoxLayout(page);
    l->setContentsMargins(8, 8, 8, 8);
    l->setSpacing(6);

    auto* row = new QHBoxLayout;
    m_newBtn = new QToolButton;
    m_newBtn->setText("New");
    m_newBtn->setPopupMode(QToolButton::InstantPopup);
    m_newBtn->setMinimumHeight(28);
    auto* menu = new QMenu(m_newBtn);
    menu->addAction("Empty experiment…", this, &GEStudioPanel::onNewEmpty);
    menu->addAction("64-dataset benchmark of the BNF paper", this, &GEStudioPanel::onNewBenchmark);
    m_newBtn->setMenu(menu);
    m_openBtn = mkBtn("Open…");
    m_openBtn->setToolTip("Open an experiment file (.json) or the results of an earlier experiment (<prefix>_raw.csv).\n"
                          "Its datasets and models can then be changed with Edit…: only the missing runs are computed.");
    m_editBtn = mkBtn("Edit…");
    m_editBtn->setToolTip("Datasets (classification / regression), models (with / without BNF), configurations, folds, options");
    m_saveBtn = mkBtn("Save");
    m_saveAsBtn = mkBtn("Save as…");
    row->addWidget(m_newBtn); row->addWidget(m_openBtn); row->addWidget(m_editBtn);
    row->addWidget(m_saveBtn); row->addWidget(m_saveAsBtn);
    row->addSpacing(16);
    row->addWidget(new QLabel("Working folder:"));
    m_batchDirEdit = new QLineEdit;
    m_batchDirEdit->setToolTip("Relative paths (data folder, results prefix) are resolved against this folder.");
    row->addWidget(m_batchDirEdit, 1);
    m_batchBrowseBtn = mkBtn("…");
    m_batchBrowseBtn->setFixedWidth(34);
    row->addWidget(m_batchBrowseBtn);
    l->addLayout(row);

    m_expSummary = new QTextBrowser;
    m_expSummary->setOpenLinks(false);
    m_expSummary->setMinimumHeight(150);
    auto* mid = new QHBoxLayout;
    mid->addWidget(m_expSummary, 5);
    auto* mbox = new QGroupBox("Models  (click to select; datasets, folds and options: Edit…)");
    auto* mbl = new QVBoxLayout(mbox);
    mbl->setContentsMargins(6, 4, 6, 6);
    m_expModels = new GEModelPicker;
    m_expModels->setMinimumHeight(250);
    mbl->addWidget(m_expModels);
    mid->addWidget(mbox, 6);
    l->addLayout(mid);
    connect(m_expModels, &GEModelPicker::changed, this, &GEStudioPanel::onExperimentModelsPicked);

    // advanced: the equivalent ge_batch command (a pasted command replaces the experiment)
    m_cmdToggle = new QToolButton;
    m_cmdToggle->setCheckable(true);
    m_cmdToggle->setAutoRaise(true);
    m_cmdToggle->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    m_cmdToggle->setText("Command line (the same as ge_batch; paste a command here to use it instead)");
    l->addWidget(m_cmdToggle);
    m_cmdBox = new QWidget;
    auto* cl = new QVBoxLayout(m_cmdBox); cl->setContentsMargins(0, 0, 0, 0);
    m_batchCmdEdit = new QPlainTextEdit;
    m_batchCmdEdit->setFont(QFontDatabase::systemFont(QFontDatabase::FixedFont));
    m_batchCmdEdit->setPlaceholderText("ge_batch --dir sample_data/tenfolding --datasets pima,airfoil --folds 1-10 "
                                       "--configs unified-ge,baselines --total-budget 600000 --out results/run1");
    { const QFontMetrics fm(m_batchCmdEdit->font()); m_batchCmdEdit->setFixedHeight(fm.lineSpacing() * 3 + 14); }
    cl->addWidget(m_batchCmdEdit);
    auto* crow = new QHBoxLayout;
    m_batchCheckBtn = mkBtn("Check / apply");
    m_batchCheckBtn->setToolTip("Validate the command; an edited or pasted command becomes the current experiment.");
    m_batchUsageBtn = mkBtn("Options");
    crow->addWidget(m_batchCheckBtn); crow->addWidget(m_batchUsageBtn); crow->addStretch();
    cl->addLayout(crow);
    l->addWidget(m_cmdBox);
    l->addStretch(1);

    // restore the session
    QSettings st = batchSettings();
    const QString wd = st.value("GEStudio/batchWorkDir").toString();
    m_batchDirEdit->setText(!wd.isEmpty() && QFileInfo(wd).isDir() ? wd : QDir::toNativeSeparators(defaultWorkDir()));
    const bool showCmd = st.value("GEStudio/batchExpanded", false).toBool();
    m_cmdToggle->setChecked(showCmd);
    m_cmdToggle->setArrowType(showCmd ? Qt::DownArrow : Qt::RightArrow);
    m_cmdBox->setVisible(showCmd);
    {
        GEExperiment e; QString err; QStringList w;
        const QByteArray js = st.value("GEStudio/currentExperiment").toByteArray();
        if (js.isEmpty() || !geExperimentFromJson(js, "experiment", e, err, w)) e = emptyExperiment();
        setExperiment(e, st.value("GEStudio/currentExperimentPath").toString(), st.value("GEStudio/currentExperimentDirty").toBool());
        const QString cmd = st.value("GEStudio/batchCommand").toString();
        if (!cmd.trimmed().isEmpty() && cmd != m_cmdGenerated) m_batchCmdEdit->setPlainText(cmd);   // keep an unapplied command
    }

    connect(m_cmdToggle, &QToolButton::toggled, this, [this](bool on) {
        m_cmdToggle->setArrowType(on ? Qt::DownArrow : Qt::RightArrow);
        m_cmdBox->setVisible(on);
        saveBatchSettings();
    });
    connect(m_openBtn, &QPushButton::clicked, this, &GEStudioPanel::onOpen);
    connect(m_editBtn, &QPushButton::clicked, this, &GEStudioPanel::onEdit);
    connect(m_saveBtn, &QPushButton::clicked, this, &GEStudioPanel::onSave);
    connect(m_saveAsBtn, &QPushButton::clicked, this, &GEStudioPanel::onSaveAs);
    connect(m_batchCheckBtn, &QPushButton::clicked, this, &GEStudioPanel::onBatchCheck);
    connect(m_batchBrowseBtn, &QPushButton::clicked, this, &GEStudioPanel::onBatchBrowseDir);
    connect(m_batchUsageBtn, &QPushButton::clicked, this, &GEStudioPanel::onBatchUsage);
    connect(m_batchDirEdit, &QLineEdit::editingFinished, this, [this] { saveBatchSettings(); refreshExperimentSummary(); });
    return page;
}

// The folder that holds sample_data (searching upwards from the executable), so that relative
// paths written for the project root work when the app is started from build/Release.
QString GEStudioPanel::defaultWorkDir() const {
    QDir d(QCoreApplication::applicationDirPath());
    for (int up = 0; up < 6; ++up) {
        if (d.exists("sample_data")) return d.absolutePath();
        if (!d.cdUp()) break;
    }
    return QDir::currentPath();
}

QString GEStudioPanel::workDir() const {
    QString wd = m_batchDirEdit ? m_batchDirEdit->text().trimmed() : QString();
    if (wd.isEmpty()) wd = defaultWorkDir();
    wd = QDir::fromNativeSeparators(wd);
    wd.replace('\\', '/');
    return wd;
}

void GEStudioPanel::saveBatchSettings() const {
    QSettings st = batchSettings();
    st.setValue("GEStudio/batchCommand", m_batchCmdEdit ? m_batchCmdEdit->toPlainText() : QString());
    st.setValue("GEStudio/batchWorkDir", m_batchDirEdit ? m_batchDirEdit->text() : QString());
    st.setValue("GEStudio/batchExpanded", m_cmdToggle && m_cmdToggle->isChecked());
    st.setValue("GEStudio/currentExperiment", geExperimentToJson(m_exp));
    st.setValue("GEStudio/currentExperimentPath", m_expPath);
    st.setValue("GEStudio/currentExperimentDirty", m_expDirty);
    if (m_setupTabs) st.setValue("GEStudio/setupTab", m_setupTabs->currentIndex());
}

QString GEStudioPanel::experimentPrefix() const {
    QString p = m_exp.output.trimmed(); p.replace('\\', '/');
    if (p.isEmpty()) return QString();
    return QDir::isAbsolutePath(p) ? QDir::cleanPath(p) : QDir::cleanPath(QDir(workDir()).absoluteFilePath(p));
}

GEReportFilter GEStudioPanel::reportFilter() const {
    GEReportFilter f;
    f.datasets = m_exp.datasets();
    f.methods = m_exp.runModels();
    f.excludeFromTests = m_exp.development;
    const QStringList base = geBaselineIds();
    bool anyBase = false, anyGe = false;
    for (const auto& m : m_exp.models) if (m.run) (base.contains(m.id) ? anyBase : anyGe) = true;
    if (anyGe) f.configs << m_exp.configs;
    if (anyBase) f.configs << "baselines";
    return f;
}

void GEStudioPanel::setExperiment(const GEExperiment& e, const QString& path, bool dirty) {
    m_exp = e;
    m_expPath = path;
    m_expDirty = dirty;
    m_cmdGenerated = geExperimentToCommand(m_exp);
    if (m_batchCmdEdit) m_batchCmdEdit->setPlainText(m_cmdGenerated);
    if (m_expModels) {
        QStringList ids;
        for (const auto& m : m_exp.models) ids << m.id;
        m_expModels->setEntries(GEModelPicker::catalogEntries(ids));
        m_expModels->setChecked(m_exp.runModels());
    }
    refreshExperimentSummary();
    saveBatchSettings();
}

// A tick on the Models list of the Experiment page changes the experiment at once
// (added models are computed by the next Run, removed ones leave the tables).
void GEStudioPanel::onExperimentModelsPicked() {
    if (m_running) return;
    QString err;
    if (!applyCommandIfEdited(err)) { QMessageBox::warning(this, "Experiment", err); setExperiment(m_exp, m_expPath, m_expDirty); return; }
    const QStringList on = m_expModels->checked();
    bool changed = false;
    for (auto& m : m_exp.models) {
        const bool run = on.contains(m.id);
        if (m.run != run) { m.run = run; changed = true; }
    }
    if (!changed) return;
    setExperiment(m_exp, m_expPath, true);
    const QString raw = experimentPrefix() + "_raw.csv";
    if (QFileInfo::exists(raw)) {
        QWidget* cur = m_tabs->currentWidget();
        loadOverallFromRaw(raw, true);
        m_tabs->setCurrentWidget(cur);
    }
}

void GEStudioPanel::refreshExperimentSummary() {
    if (!m_expSummary) return;
    const GEExperiment& e = m_exp;
    auto esc = [](const QString& s) { return s.toHtmlEscaped(); };
    QString h;
    h += QString("<b style='font-size:13px'>%1</b>%2 &nbsp; <span style='color:gray'>%3</span><br>")
             .arg(esc(e.name), m_expDirty ? QString(" <span style='color:#c0392b'>(modified — not saved)</span>") : QString(),
                  m_expPath.isEmpty() ? QString("not saved to a file") : esc(QDir::toNativeSeparators(m_expPath)));
    if (!e.description.isEmpty()) h += "<i>" + esc(e.description) + "</i><br>";
    h += QString("Data: <code>%1</code> &nbsp; Results: <code>%2</code><br>")
             .arg(esc(e.dataDir), esc(e.output));
    h += QString("Folds %1 &nbsp; Seeds %2 &nbsp; Grammar configurations: %3 &nbsp; %4<br>")
             .arg(esc(e.folds)).arg(e.seeds).arg(esc(e.configs.join(", ")))
             .arg(e.fixedBudget() ? "total budget T = " + e.options.value("total-budget") : QString("equal-budget mode"));
    h += QString("<b>Datasets:</b> %1 classification, %2 regression")
             .arg(e.classification.size()).arg(e.regression.size());
    if (!e.development.isEmpty())
        h += QString(" &nbsp; <span style='color:gray'>development (not in the hypothesis tests): %1</span>")
                 .arg(esc(e.development.join(", ")));
    if (!e.datasets().isEmpty()) {
        QStringList shown = e.classification.mid(0, 12);
        QString cl = esc(shown.join(", ")) + (e.classification.size() > 12 ? " …" : "");
        QString rg = esc(e.regression.mid(0, 12).join(", ")) + (e.regression.size() > 12 ? " …" : "");
        h += QString("<br><span style='color:gray'>classification: %1<br>regression: %2</span>").arg(cl, rg);
    }
    QStringList withBnf, without, skipped;
    for (const auto& m : e.models) {
        const QString label = geMethodInfo(m.id).label;
        if (!m.run) skipped << label; else (m.bnf ? withBnf : without) << label;
    }
    h += QString("<br><b>Models with BNF:</b> %1<br><b>Models without BNF:</b> %2")
             .arg(withBnf.isEmpty() ? QString("—") : esc(withBnf.join(", ")),
                  without.isEmpty() ? QString("—") : esc(without.join(", ")));
    if (!skipped.isEmpty()) h += QString("<br><span style='color:gray'>not run: %1</span>").arg(esc(skipped.join(", ")));
    // progress from the results file
    const QString raw = experimentPrefix() + "_raw.csv";
    if (!experimentPrefix().isEmpty() && QFileInfo::exists(raw)) {
        QVector<GERawRow> rows; QString err;
        geLoadRawCsv(raw, rows, err);
        int total = 0; const int done = geExperimentDone(e, rows, total);
        if (m_expModels) {
            QMap<QString, int> n;
            const QStringList ds = e.datasets();
            for (const GERawRow& r : rows) if (ds.contains(r.ds)) ++n[r.method];
            QMap<QString, QString> txt;
            for (auto it = n.begin(); it != n.end(); ++it) txt[it.key()] = QString::number(it.value());
            m_expModels->setInfoColumn("Result rows", txt);
        }
        h += QString("<br><b>Results:</b> %1 of %2 runs complete%3").arg(done).arg(total)
                 .arg(done < total ? QString(" — Run computes only the missing ones") : QString(" ✓"));
    } else {
        if (m_expModels) m_expModels->setInfoColumn(QString(), {});
        int total = 0; geExperimentDone(e, {}, total);
        h += QString("<br><b>Results:</b> none yet — %1 runs to compute").arg(total);
    }
    m_expSummary->setHtml(h);
    updateActionStates();
}

bool GEStudioPanel::experimentMode() const {
    return m_setupTabs && m_setupTabs->currentWidget() == m_expPage;
}

void GEStudioPanel::updateActionStates() {
    if (!m_exportBtn) return;
    if (experimentMode()) {
        m_exportBtn->setEnabled(!m_running && !m_overall.isEmpty());
        m_exportBtn->setToolTip("Overall tables (Classification / Regression, with and without BNF) to .xlsx");
    } else {
        m_exportBtn->setEnabled(!m_running && !m_results.isEmpty());
        m_exportBtn->setToolTip("Results of the single run to .xlsx");
    }
    if (m_batchOpenBtn) m_batchOpenBtn->setVisible(experimentMode());
    if (m_batchOpenBtn) m_batchOpenBtn->setEnabled(!experimentPrefix().isEmpty() && QFileInfo(QFileInfo(experimentPrefix()).absolutePath()).isDir());
    if (m_saveBtn) m_saveBtn->setEnabled(!m_running);
}

void GEStudioPanel::onSetupTabChanged(int idx) {
    // the setup area takes the height of the visible page only (the results get the rest)
    for (int i = 0; i < m_setupTabs->count(); ++i)
        m_setupTabs->widget(i)->setSizePolicy(QSizePolicy::Preferred, i == idx ? QSizePolicy::Preferred : QSizePolicy::Ignored);
    m_setupTabs->updateGeometry();
    m_runBtn->setText(experimentMode() ? "▶  Run experiment" : "▶  Run");
    showResultTabsFor(experimentMode());
    updateActionStates();
    saveBatchSettings();
}

void GEStudioPanel::onRunClicked() {
    if (experimentMode()) onBatchRun(); else onRun();
}

void GEStudioPanel::onExportClicked() {
    if (experimentMode()) onExportTables(); else onExportXlsx();
}

// ── New / Open / Edit / Save ──────────────────────────────────────────────────
void GEStudioPanel::onNewEmpty() {
    GEExperimentDialog dlg(emptyExperiment(), workDir(), this);
    if (dlg.exec() != QDialog::Accepted) return;
    setExperiment(dlg.experiment(), QString(), true);
    m_overall = GEOverall(); m_overallSource.clear();
    refreshOverallViews();
    const QString raw = experimentPrefix() + "_raw.csv";
    if (QFileInfo::exists(raw)) loadOverallFromRaw(raw, true);
}

void GEStudioPanel::onNewBenchmark() {
    setExperiment(geBenchmarkExperiment(), QString(), true);
    m_tabs->setCurrentWidget(m_batchView);
    m_batchView->appendPlainText("\n" + geExperimentSummary(m_exp));
    const QString raw = experimentPrefix() + "_raw.csv";
    if (QFileInfo::exists(raw)) loadOverallFromRaw(raw, true);
}

void GEStudioPanel::onOpen() {
    QString start = QDir(workDir()).exists("experiments") ? workDir() + "/experiments" : workDir();
    const QString path = QFileDialog::getOpenFileName(this, "Open experiment or results", start,
        "Experiments and results (*.json *_raw.csv);;Experiment (*.json);;Results (*_raw.csv);;All files (*)");
    if (path.isEmpty()) return;
    GEExperiment e; QString err; QStringList warns; QString expPath;
    if (path.endsWith(".csv", Qt::CaseInsensitive)) {
        if (!geExperimentFromRaw(path, e, err, warns)) { QMessageBox::warning(this, "Open", err); return; }
        QString prefix = path; if (prefix.endsWith("_raw.csv")) prefix.chop(8);
        if (QFileInfo::exists(prefix + "_experiment.json")) expPath = prefix + "_experiment.json";
        // keep the results where they are: the prefix is the opened file's, relative to the working folder if possible
        QString rel = QDir(workDir()).relativeFilePath(prefix);
        e.output = rel.startsWith("..") ? prefix : rel;
    } else {
        if (!geLoadExperiment(path, e, err, warns)) { QMessageBox::warning(this, "Open", err); return; }
        expPath = path;
    }
    setExperiment(e, expPath, expPath.isEmpty());
    m_setupTabs->setCurrentWidget(m_expPage);
    m_batchView->appendPlainText("\nOpened " + QDir::toNativeSeparators(path) + "\n" + geExperimentSummary(m_exp));
    for (const QString& w : warns) m_batchView->appendPlainText("note: " + w);
    const QString raw = experimentPrefix() + "_raw.csv";
    m_overall = GEOverall(); m_overallSource.clear();
    if (QFileInfo::exists(raw)) loadOverallFromRaw(raw, true);
    else { refreshOverallViews(); m_tabs->setCurrentWidget(m_batchView); }
    if (!warns.isEmpty()) QMessageBox::information(this, "Open", warns.mid(0, 6).join("\n\n"));
    updateActionStates();
}

void GEStudioPanel::onEdit() {
    QString err;
    if (!applyCommandIfEdited(err)) { QMessageBox::warning(this, "Edit", err); return; }
    GEExperimentDialog dlg(m_exp, workDir(), this);
    if (dlg.exec() != QDialog::Accepted) return;
    const GEExperiment e = dlg.experiment();
    const bool optsChanged = e.options != m_exp.options || e.folds != m_exp.folds || e.seeds != m_exp.seeds;
    const QString raw = QFileInfo::exists(experimentPrefix() + "_raw.csv") ? experimentPrefix() + "_raw.csv" : QString();
    setExperiment(e, m_expPath, true);
    if (optsChanged && !raw.isEmpty())
        QMessageBox::information(this, "Edit", "Options, folds or seeds changed while results already exist for this prefix. "
                                 "New GE runs will use the new settings; consider a new results prefix to keep the "
                                 "comparison consistent.");
    const QString raw2 = experimentPrefix() + "_raw.csv";
    if (QFileInfo::exists(raw2)) loadOverallFromRaw(raw2, true);   // removed models / datasets leave the tables at once
    else { m_overall = GEOverall(); refreshOverallViews(); }
    updateActionStates();
}

void GEStudioPanel::onSave() {
    if (m_expPath.isEmpty()) { onSaveAs(); return; }
    QString err;
    if (!applyCommandIfEdited(err)) { QMessageBox::warning(this, "Save", err); return; }
    if (!geSaveExperiment(m_expPath, m_exp, err)) { QMessageBox::critical(this, "Save", err); return; }
    m_expDirty = false;
    refreshExperimentSummary();
    saveBatchSettings();
    m_statusLbl->setText("Saved " + QDir::toNativeSeparators(m_expPath));
}

void GEStudioPanel::onSaveAs() {
    QString err;
    if (!applyCommandIfEdited(err)) { QMessageBox::warning(this, "Save", err); return; }
    QDir(workDir()).mkpath("experiments");
    const QString def = m_expPath.isEmpty() ? workDir() + "/experiments/" + (m_exp.name.isEmpty() ? QString("experiment") : m_exp.name) + ".json"
                                            : m_expPath;
    const QString path = QFileDialog::getSaveFileName(this, "Save experiment", def, "Experiment (*.json)");
    if (path.isEmpty()) return;
    if (!geSaveExperiment(path, m_exp, err)) { QMessageBox::critical(this, "Save", err); return; }
    m_expPath = path; m_expDirty = false;
    refreshExperimentSummary();
    saveBatchSettings();
    m_statusLbl->setText("Saved " + QDir::toNativeSeparators(path));
}

// A command typed or pasted in the command box replaces the experiment.
bool GEStudioPanel::applyCommandIfEdited(QString& error) {
    const QString cmd = m_batchCmdEdit->toPlainText().trimmed();
    if (cmd.isEmpty() || cmd == m_cmdGenerated.trimmed()) return true;
    GEBatchOptions o;
    if (!parseBatchArgs(splitBatchCommandLine(cmd), workDir(), o, error)) return false;
    if (o.help) { onBatchUsage(); error = "--help only shows the options."; return false; }
    GEExperiment e;
    if (!o.experimentPath.isEmpty()) {
        QStringList w;
        if (!geLoadExperiment(o.experimentPath, e, error, w)) return false;
        // options given on the command line next to --experiment override the file
        GEExperiment fromCmd; QStringList bad;
        QGuiApplication::setOverrideCursor(Qt::WaitCursor);
        const bool ok = geExperimentFromBatchOptions(o, fromCmd, bad);
        QGuiApplication::restoreOverrideCursor();
        if (ok) { fromCmd.name = e.name; fromCmd.description = e.description; e = fromCmd; }
        setExperiment(e, o.experimentPath, ok);
        return true;
    }
    QStringList bad;
    QGuiApplication::setOverrideCursor(Qt::WaitCursor);
    const bool ok = geExperimentFromBatchOptions(o, e, bad);
    QGuiApplication::restoreOverrideCursor();
    if (!ok) { error = "None of the datasets of the command can be read."; return false; }
    if (!bad.isEmpty()) m_batchView->appendPlainText("note: left out (cannot be read): " + bad.join(", "));
    setExperiment(e, QString(), true);
    return true;
}

void GEStudioPanel::onBatchBrowseDir() {
    const QString d = QFileDialog::getExistingDirectory(this, "Working folder", m_batchDirEdit->text());
    if (!d.isEmpty()) { m_batchDirEdit->setText(QDir::toNativeSeparators(d)); saveBatchSettings(); refreshExperimentSummary(); }
}

void GEStudioPanel::onBatchUsage() {
    m_tabs->setCurrentWidget(m_batchView);
    m_batchView->appendPlainText(batchUsage());
}

// Check / apply: an edited command becomes the experiment; then the run plan is shown.
void GEStudioPanel::onBatchCheck() {
    QString err;
    if (!applyCommandIfEdited(err)) { QMessageBox::warning(this, "Command", err); return; }
    const GEExperiment& e = m_exp;
    QString dir = e.dataDir; dir.replace('\\', '/');
    if (!QDir::isAbsolutePath(dir)) dir = QDir(workDir()).absoluteFilePath(dir);
    int files = 0; QStringList missing;
    QList<int> folds;
    for (const QString& p : e.folds.split(',', Qt::SkipEmptyParts)) {
        const QStringList ab = p.split('-');
        if (ab.size() == 2) for (int i = ab[0].toInt(); i <= ab[1].toInt(); ++i) folds << i; else folds << p.toInt();
    }
    for (const QString& ds : e.datasets()) for (int f : folds) {
        const QString p = QString("%1/%2_%3.train").arg(dir, ds).arg(f);
        if (QFileInfo::exists(p) && QFileInfo(p).size() > 0) ++files;
        else if (missing.size() < 6) missing << QString("%1_%2.train").arg(ds).arg(f);
    }
    int total = 0, done = 0;
    QVector<GERawRow> rows; QString e2;
    if (QFileInfo::exists(experimentPrefix() + "_raw.csv") && geLoadRawCsv(experimentPrefix() + "_raw.csv", rows, e2))
        done = geExperimentDone(e, rows, total);
    else geExperimentDone(e, {}, total);
    QMessageBox::information(this, "Experiment",
        QString("%1\n\nTrain files found: %2%3\nRuns: %4 in total, %5 complete, %6 to run.")
            .arg(geExperimentSummary(e)).arg(files)
            .arg(missing.isEmpty() ? QString() : QString(" (missing: %1%2)").arg(missing.join(", "), missing.size() >= 6 ? ", …" : ""))
            .arg(total).arg(done).arg(std::max(0, total - done)));
}

void GEStudioPanel::onBatchOpenResults() {
    const QString prefix = experimentPrefix();
    if (prefix.isEmpty()) return;
    const QString folder = QFileInfo(prefix).absolutePath();
    if (!QDesktopServices::openUrl(QUrl::fromLocalFile(folder)))
        QMessageBox::information(this, "Results", "Results are in:\n" + QDir::toNativeSeparators(folder));
}

// ── Run an experiment ─────────────────────────────────────────────────────────
void GEStudioPanel::onBatchRun() {
    if (m_running) return;
    QString err;
    if (!applyCommandIfEdited(err)) { QMessageBox::warning(this, "Experiment", err); return; }
    if (m_exp.datasets().isEmpty()) { QMessageBox::information(this, "Experiment", "The experiment has no datasets: use Edit… or New."); return; }
    if (m_exp.runModels().isEmpty()) { QMessageBox::information(this, "Experiment", "No model is selected: use Edit…"); return; }
    const QString prefix = experimentPrefix();
    if (prefix.isEmpty()) { QMessageBox::information(this, "Experiment", "Set the results prefix in Edit…"); return; }
    QDir().mkpath(QFileInfo(prefix).absolutePath());
    // the settings travel with the results: <prefix>_experiment.json
    const QString expFile = prefix + "_experiment.json";
    if (!geSaveExperiment(expFile, m_exp, err)) { QMessageBox::critical(this, "Experiment", err); return; }
    if (QFileInfo(m_expPath).absoluteFilePath() == QFileInfo(expFile).absoluteFilePath()) m_expDirty = false;
    else if (m_expPath.isEmpty()) { m_expPath = expFile; m_expDirty = false; }   // the settings now live next to the results
    GEBatchOptions o;
    if (!parseBatchArgs({"--experiment", expFile}, workDir(), o, err)) { QMessageBox::warning(this, "Experiment", err); return; }
    saveBatchSettings();

    m_batchMode = true;
    m_batchJobs = 0;
    m_batchPrefix = o.prefix;
    m_batchLastLine.clear();
    m_batchView->clear();
    m_batchView->appendPlainText(QString("Experiment: %1\nWorking folder: %2\nResults: %3\n")
        .arg(m_exp.name, QDir::toNativeSeparators(workDir()), QDir::toNativeSeparators(o.prefix)));
    for (const QString& w : o.warnings) m_batchView->appendPlainText("note: " + w);
    m_tabs->setCurrentWidget(m_batchView);
    m_bar->setValue(0);
    m_statusLbl->setText("Experiment: preparing…");

    auto* runner = new GEBatchRunner;
    runner->setOptions(o);
    m_batchRunner = runner;
    m_batchThread = new QThread(this);
    runner->moveToThread(m_batchThread);
    connect(m_batchThread, &QThread::started, runner, &GEBatchRunner::run);
    connect(runner, &GEBatchRunner::message, this, &GEStudioPanel::onBatchMessage);
    connect(runner, &GEBatchRunner::batchStarted, this, &GEStudioPanel::onBatchStarted);
    connect(runner, &GEBatchRunner::progress, this, &GEStudioPanel::onBatchProgress);
    connect(runner, &GEBatchRunner::jobFinished, this, &GEStudioPanel::onBatchJobFinished);
    connect(runner, &GEBatchRunner::finished, this, &GEStudioPanel::onBatchFinished);
    connect(runner, &GEBatchRunner::errorOccurred, this, &GEStudioPanel::onBatchError);
    // requestStop is thread-safe; a direct call reaches the runner while its thread is busy
    connect(m_stopBtn, &QPushButton::clicked, runner, &GEBatchRunner::requestStop, Qt::DirectConnection);
    connect(runner, &GEBatchRunner::finished, m_batchThread, &QThread::quit);
    connect(runner, &GEBatchRunner::errorOccurred, m_batchThread, &QThread::quit);
    connect(m_batchThread, &QThread::finished, runner, &QObject::deleteLater);
    connect(m_batchThread, &QThread::finished, m_batchThread, &QObject::deleteLater);
    setRunning(true);
    log("Experiment started: " + QDir::toNativeSeparators(o.prefix));
    m_batchThread->start();
}

void GEStudioPanel::onBatchStarted(int jobsToDo, int alreadyDone) {
    m_batchJobs = jobsToDo;
    m_statusLbl->setText(QString("Experiment: %1 runs to do (%2 already complete).").arg(jobsToDo).arg(alreadyDone));
    if (jobsToDo == 0) m_bar->setValue(1000);
}

void GEStudioPanel::onBatchProgress(int jobIdx, int jobsToDo, double within, QString label) {
    const int n = std::max(1, jobsToDo);
    const double v = (jobIdx + std::min(1.0, std::max(0.0, within))) / n;
    m_bar->setValue(static_cast<int>(std::lround(v * 1000)));
    QString s = QString("Experiment run %1/%2: %3").arg(jobIdx + 1).arg(jobsToDo).arg(label);
    if (!m_batchLastLine.isEmpty()) {
        const int eta = m_batchLastLine.indexOf("(ETA");
        if (eta >= 0) s += "   " + m_batchLastLine.mid(eta);
    }
    m_statusLbl->setText(s);
}

void GEStudioPanel::onBatchJobFinished(int jobsDone, int jobsToDo, QString line, bool ok) {
    m_batchLastLine = line;
    m_bar->setValue(static_cast<int>(std::lround(1000.0 * jobsDone / std::max(1, jobsToDo))));
    if (!ok) log("Experiment: " + line);
}

void GEStudioPanel::onBatchMessage(QString text) {
    m_batchView->appendPlainText(text);
}

void GEStudioPanel::onBatchFinished(bool stopped, QString prefix) {
    m_batchThread = nullptr;
    m_batchMode = false;
    setRunning(false);
    if (!prefix.isEmpty()) m_batchPrefix = prefix;
    saveBatchSettings();
    if (stopped) {
        m_statusLbl->setText("Stopped. Finished runs are saved; Run continues with the missing ones.");
    } else {
        m_bar->setValue(1000);
        m_statusLbl->setText("Experiment finished — " + QDir::toNativeSeparators(m_batchPrefix) + "_overall.xlsx");
    }
    log(stopped ? "Experiment stopped." : "Experiment finished.");
    if (!m_batchPrefix.isEmpty() && QFileInfo::exists(m_batchPrefix + "_raw.csv"))
        loadOverallFromRaw(m_batchPrefix + "_raw.csv", true);
    refreshExperimentSummary();
}

void GEStudioPanel::onBatchError(QString msg) {
    m_batchThread = nullptr;
    m_batchMode = false;
    setRunning(false);
    m_statusLbl->setText("Experiment error: " + msg);
    log("Experiment ERROR: " + msg);
    QMessageBox::critical(this, "Experiment", msg);
}

// ── Overall tables ────────────────────────────────────────────────────────────
bool GEStudioPanel::loadOverallFromRaw(const QString& rawPath, bool quiet) {
    QVector<GERawRow> rows;
    QString err;
    if (!geLoadRawCsv(rawPath, rows, err)) {
        if (!quiet) QMessageBox::warning(this, "Results", err);
        return false;
    }
    const GEReportFilter f = reportFilter();
    int kept = 0; for (const GERawRow& r : rows) if (f.keeps(r)) ++kept;
    m_overall = geBuildOverall(rows, f);
    m_overallSource = rawPath;
    refreshOverallViews();
    bool hasCls = false;
    for (const auto& t : m_overall.tables) hasCls = hasCls || t.task == "classification";
    if (!m_overall.isEmpty() && experimentMode()) m_tabs->setCurrentWidget(hasCls ? m_clsTable : m_regTable);
    m_statusLbl->setText(QString("Tables from %1: %2 of %3 result rows belong to this experiment%4")
                             .arg(QDir::toNativeSeparators(rawPath)).arg(kept).arg(rows.size())
                             .arg(err.isEmpty() ? QString() : " — " + err));
    if (!m_overall.warnings.isEmpty()) m_batchView->appendPlainText("notes: " + m_overall.warnings.join("\nnotes: "));
    updateActionStates();
    return true;
}

void GEStudioPanel::fillOverallTable(QTableWidget* t, const QString& task) {
    t->clear();
    t->setRowCount(0);
    QVector<const GEOverallTable*> tabs;
    for (const auto& tb : m_overall.tables) if (tb.task == task) tabs << &tb;
    if (tabs.isEmpty()) {
        t->setColumnCount(1); t->setRowCount(1);
        t->setItem(0, 0, cell(QString("No %1 datasets in %2.").arg(task, m_overallSource.isEmpty() ? QString("the results")
                                                                   : QFileInfo(m_overallSource).fileName())));
        t->resizeColumnsToContents();
        return;
    }
    const bool cls = task == "classification";
    int cols = 2;
    for (const auto* tb : tabs) cols = std::max(cols, 2 + static_cast<int>(tb->methods.size()));
    t->setColumnCount(cols);
    const QColor bnfColor(kAccent);
    QFont bold = t->font(); bold.setBold(true);
    auto addRow = [&]() { const int r = t->rowCount(); t->insertRow(r); return r; };
    auto put = [&](int r, int c, const QString& s, bool b = false, bool right = false) {
        auto* it = cell(s, right);
        if (b) it->setFont(bold);
        t->setItem(r, c, it);
        return it;
    };
    auto num = [&](double v) {
        if (std::isnan(v)) return QString("-");
        return cls ? QString::number(v, 'f', 2) : QString::number(v, 'g', 5);
    };
    for (const auto* tb : tabs) {
        const int M = tb->methods.size();
        if (t->rowCount() > 0) { addRow(); }
        int r = addRow();
        put(r, 0, QString("%1 — configuration %2 — %3 (mean over folds), %4 datasets")
                      .arg(cls ? "Classification" : "Regression", tb->config, cls ? "test error %" : "test MSE")
                      .arg(tb->datasets.size()), true);
        t->setSpan(r, 0, 1, cols);
        // group row
        r = addRow();
        int nNo = 0; for (bool b : tb->bnf) if (!b) ++nNo;
        if (nNo > 0) { put(r, 2, "Without BNF", true)->setTextAlignment(Qt::AlignCenter); if (nNo > 1) t->setSpan(r, 2, 1, nNo); }
        if (M - nNo > 0) {
            auto* it = put(r, 2 + nNo, "With BNF", true);
            it->setTextAlignment(Qt::AlignCenter); it->setForeground(bnfColor);
            if (M - nNo > 1) t->setSpan(r, 2 + nNo, 1, M - nNo);
        }
        r = addRow();
        put(r, 0, "Dataset", true); put(r, 1, "Folds", true, true);
        for (int j = 0; j < M; ++j) {
            auto* it = put(r, 2 + j, geMethodInfo(tb->methods[j]).label, true, true);
            it->setToolTip(geMethodInfo(tb->methods[j]).description);
            if (tb->bnf[j]) it->setForeground(bnfColor);
        }
        for (int i = 0; i < tb->datasets.size(); ++i) {
            r = addRow();
            put(r, 0, tb->datasets[i]); put(r, 1, QString::number(tb->folds[i]), false, true);
            double best = std::numeric_limits<double>::infinity();
            for (int j = 0; j < M; ++j) if (!std::isnan(tb->value[i][j])) best = std::min(best, tb->value[i][j]);
            for (int j = 0; j < M; ++j) {
                const double v = tb->value[i][j];
                auto* it = put(r, 2 + j, num(v), !std::isnan(v) && v <= best + 1e-12 * std::max(1.0, std::fabs(best)), true);
                it->setToolTip(QString("std over folds: %1").arg(num(tb->sd[i][j])));
            }
        }
        r = addRow();
        put(r, 0, "AVERAGE", true); put(r, 1, QString::number(tb->complete), false, true);
        for (int j = 0; j < M; ++j) put(r, 2 + j, num(tb->average[j]), true, true);
        r = addRow();
        put(r, 0, "Average rank (Friedman)", true);
        for (int j = 0; j < M; ++j) put(r, 2 + j, std::isnan(tb->avgRank[j]) ? "-" : QString::number(tb->avgRank[j], 'f', 2), true, true);
        if (!tb->reference.isEmpty()) {
            const QString refL = geMethodInfo(tb->reference).label;
            r = addRow();
            put(r, 0, refL + ": wins/ties/losses");
            for (int j = 0; j < M; ++j)
                put(r, 2 + j, tb->methods[j] == tb->reference ? QString("reference")
                              : QString("%1/%2/%3").arg(tb->wins[j]).arg(tb->ties[j]).arg(tb->losses[j]), false, true);
            r = addRow();
            put(r, 0, "Wilcoxon p (" + refL + " vs column)");
            for (int j = 0; j < M; ++j) {
                const double p = tb->wilcoxonP[j];
                put(r, 2 + j, tb->methods[j] == tb->reference || std::isnan(p) ? QString("-")
                              : (p < 0.001 ? QString("<0.001") : QString::number(p, 'f', 3)), !std::isnan(p) && p < 0.05, true);
            }
            // scale-free comparison: relative differences, Holm over all columns
            r = addRow();
            put(r, 0, "Median relative difference % (> 0: " + refL + " better)")
                ->setToolTip("per dataset (column − reference) / mean of the two; scale-free, so datasets with a large MSE do not dominate");
            for (int j = 0; j < M; ++j) {
                const double v = tb->relDiffMedian[j];
                put(r, 2 + j, tb->methods[j] == tb->reference || std::isnan(v) ? QString("-") : QString::number(v, 'f', 1), false, true);
            }
            auto pRow = [&](const QString& name, const QVector<double>& pv) {
                const int rr = addRow();
                put(rr, 0, name);
                for (int j = 0; j < M; ++j) {
                    const double p = pv[j];
                    put(rr, 2 + j, tb->methods[j] == tb->reference || std::isnan(p) ? QString("-")
                                   : (p < 0.001 ? QString("<0.001") : QString::number(p, 'f', 3)), !std::isnan(p) && p < 0.05, true);
                }
            };
            pRow("Wilcoxon p on relative differences", tb->wilcoxonRelP);
            pRow("Holm-adjusted p (all columns vs " + refL + ")", tb->holmP);
        }
        r = addRow();
        put(r, 0, QString("Friedman p = %1    Nemenyi CD(0.05) = %2    datasets with every method: %3")
                      .arg(std::isnan(tb->friedmanP) ? QString("-") : QString::number(tb->friedmanP, 'g', 3))
                      .arg(std::isnan(tb->cd05) ? QString("-") : QString::number(tb->cd05, 'f', 2))
                      .arg(tb->complete));
        t->setSpan(r, 0, 1, cols);
    }
    fitColumnsIgnoringSpans(t);
    t->setColumnWidth(0, std::max(t->columnWidth(0), 120));
}

void GEStudioPanel::refreshOverallViews() {
    fillOverallTable(m_clsTable, "classification");
    fillOverallTable(m_regTable, "regression");
    fillHypothesesTable();
}

// Pre-declared hypotheses: reference < method, per task and over both tasks, Holm-adjusted.
void GEStudioPanel::fillHypothesesTable() {
    QTableWidget* t = m_hypTable;
    if (!t) return;
    t->clear();
    const QStringList head{"Scope", "Hypothesis", "Compared with", "Meaning", "Datasets", "Ref. better / tie / worse",
                           "Median rel. diff. %", "Ratio (geo. mean)", "Wilcoxon p", "Holm p", "Result"};
    t->setColumnCount(head.size());
    t->setHorizontalHeaderLabels(head);
    t->setRowCount(0);
    QFont bold = t->font(); bold.setBold(true);
    auto fp = [](double p) { return std::isnan(p) ? QString("-") : p < 0.001 ? QString("<0.001") : QString::number(p, 'f', 3); };
    QString lastKey;
    for (const GEHypothesisTest& h : m_overall.hypotheses) {
        const QString key = h.config + "|" + h.scope;
        if (key != lastKey) {
            const int r = t->rowCount(); t->insertRow(r);
            auto* it = cell(QString("%1 — %2 — %3 < method%4").arg(h.config, h.scope, geMethodInfo(h.reference).label,
                            h.excluded.isEmpty() ? QString() : QString("   (left out: %1)").arg(h.excluded.join(", "))));
            it->setFont(bold);
            t->setItem(r, 0, it);
            t->setSpan(r, 0, 1, head.size());
            lastKey = key;
        }
        const int r = t->rowCount(); t->insertRow(r);
        const bool sig = !std::isnan(h.holmP) && h.holmP < 0.05;
        const QStringList v{h.scope, h.id, geMethodInfo(h.method).label, h.meaning, QString::number(h.n),
                            QString("%1 / %2 / %3").arg(h.better).arg(h.ties).arg(h.worse),
                            std::isnan(h.relDiffMedian) ? QString("-") : QString::number(h.relDiffMedian, 'f', 1),
                            std::isnan(h.geoRatio) ? QString("-") : QString::number(h.geoRatio, 'f', 3),
                            fp(h.p), fp(h.holmP),
                            sig ? QString("supported") : QString("not significant")};
        for (int c = 0; c < v.size(); ++c) {
            auto* it = cell(v[c], c >= 4 && c <= 9);
            if (c == 10 && sig) it->setFont(bold);
            t->setItem(r, c, it);
        }
    }
    if (m_overall.hypotheses.isEmpty()) {
        t->insertRow(0);
        t->setItem(0, 0, cell("No hypothesis tests: they need GE #1 and the models it is compared with on complete datasets."));
        t->setSpan(0, 0, 1, head.size());
    } else {
        const int r = t->rowCount(); t->insertRow(r);
        t->setItem(r, 0, cell("Relative difference = (method − reference) / mean of the two, per dataset (> 0: the reference is "
                              "better). Wilcoxon signed-rank test on it, Holm adjustment over the hypotheses of the same scope. "
                              "Development datasets and datasets without every fold are left out."));
        t->setSpan(r, 0, 1, head.size());
    }
    fitColumnsIgnoringSpans(t);
}

// The result tabs that belong to the current page: a single run fills Convergence … Grammar statistics,
// an experiment fills Classification, Regression and Hypotheses; the Log (with the experiment report)
// belongs to both and is always visible.
void GEStudioPanel::showResultTabsFor(bool experiment) {
    const QList<QWidget*> single{m_chartView, m_cmpTable, m_topTable, m_probsView};
    const QList<QWidget*> exp{m_clsTable, m_regTable, m_hypTable};
    for (QWidget* w : single) m_tabs->setTabVisible(m_tabs->indexOf(w), !experiment);
    for (QWidget* w : exp) m_tabs->setTabVisible(m_tabs->indexOf(w), experiment);
    const QList<QWidget*>& mine = experiment ? exp : single;
    if (!mine.contains(m_tabs->currentWidget()) && m_tabs->currentWidget() != m_logView)
        m_tabs->setCurrentWidget(experiment ? (m_overall.isEmpty() ? m_batchView : static_cast<QWidget*>(m_clsTable))
                                            : static_cast<QWidget*>(m_chartView));
}

void GEStudioPanel::onExportTables() {
    if (m_overall.isEmpty()) return;
    QString def = workDir() + "/results/overall.xlsx";
    if (m_overallSource.endsWith("_raw.csv")) def = m_overallSource.left(m_overallSource.size() - 8) + "_overall.xlsx";
    const QString path = QFileDialog::getSaveFileName(this, "Export overall tables", def, XlsxWriter::fileFilter());
    if (path.isEmpty()) return;
    try {
        geWriteOverallXlsx(m_overall, path);
        m_statusLbl->setText("Exported: " + QDir::toNativeSeparators(path));
    } catch (const std::exception& ex) {
        QMessageBox::critical(this, "Export", QString::fromStdString(ex.what()));
    }
}

} // namespace NeuralStudio
