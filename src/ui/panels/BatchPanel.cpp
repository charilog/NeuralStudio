#include "BatchPanel.h"
#include "WilcoxonTab.h"
#include "FriedmanTab.h"
#include "core/dataset/DatasetLoader.h"
#include "core/export/XlsxWriter.h"
#include "core/nn/ActivationFunction.h"
#include "core/nn/Trainer.h"
#include "utils/Logger.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QHeaderView>
#include <QFileDialog>
#include <QFileInfo>
#include <QMessageBox>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QMimeData>
#include <QTableWidgetItem>
#include <QColor>
#include <QFont>
#include <QScrollArea>
#include <QFrame>
#include <stdexcept>
#include <limits>
#include <cmath>

namespace NeuralStudio {

// ─── NumericItem ─────────────────────────────────────────────────────────────
namespace {
class NumericItem : public QTableWidgetItem {
public:
    static constexpr double kNA = -1e18;
    NumericItem(double sortVal, const QString& display, const QColor& bg = QColor())
        : QTableWidgetItem(display), m_sort(sortVal) {
        setTextAlignment(Qt::AlignCenter);
        if (bg.isValid()) {
            const QBrush b(bg), f(QColor(20,30,50));
            setBackground(b); setData(Qt::BackgroundRole,b);
            setForeground(f); setData(Qt::ForegroundRole,f);
            QFont fnt=font(); fnt.setBold(true); setFont(fnt);
        } else if (sortVal <= kNA+1.0) {
            const QBrush d(QColor(80,85,100));
            setForeground(d); setData(Qt::ForegroundRole,d);
        }
    }
    bool operator<(const QTableWidgetItem& o) const override {
        const auto* n=dynamic_cast<const NumericItem*>(&o);
        return n ? m_sort < n->m_sort : QTableWidgetItem::operator<(o);
    }
private:
    double m_sort;
};
} // anonymous

// ─── Constructor ─────────────────────────────────────────────────────────────
BatchPanel::BatchPanel(QWidget* parent) : QWidget(parent) {
    setAcceptDrops(true);
    buildUi();
}

// ─── buildUi ─────────────────────────────────────────────────────────────────
void BatchPanel::buildUi() {
    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(20, 16, 20, 16);
    root->setSpacing(8);

    // Title
    auto* title = new QLabel("Batch Run");
    { QFont f=title->font(); f.setPointSize(14); f.setBold(true); title->setFont(f); }
    root->addWidget(title);

    auto* hint = new QLabel(
        "Load datasets, select optimizers, and train all combinations. "
        "Results: one row per (dataset \303\227 optimizer). Drag &amp; drop supported.");
    hint->setStyleSheet("color:#C8C8D8;font-size:11px;"); hint->setWordWrap(true);
    root->addWidget(hint);

    // ── Top buttons ──────────────────────────────────────────────────────────
    auto* btnRow = new QHBoxLayout; btnRow->setSpacing(8);
    auto mkBtn = [&](const QString& t, const QString& bg="", int minW=0) -> QPushButton* {
        auto* b = new QPushButton(t); b->setMinimumHeight(30);
        if (minW) b->setMinimumWidth(minW);
        if (!bg.isEmpty())
            b->setStyleSheet(QString("QPushButton{background:%1;color:white;"
                "border-radius:5px;font-weight:600;padding:0 14px;}"
                "QPushButton:disabled{background:palette(midlight);color:palette(mid);}").arg(bg));
        return b;
    };
    m_addBtn   = mkBtn("Add Files...");    m_addBtn->setMinimumWidth(110);
    m_clearBtn = mkBtn("Clear Queue");     m_clearBtn->setMinimumWidth(110);
    m_runBtn   = mkBtn("\u25B6  Run Batch","#1D9E75",140);
    m_stopBtn  = mkBtn("\u25A0  Stop","#C0392B",100); m_stopBtn->setEnabled(false);

    btnRow->addWidget(m_addBtn); btnRow->addWidget(m_clearBtn);
    btnRow->addWidget(m_runBtn); btnRow->addWidget(m_stopBtn);
    btnRow->addStretch();
    root->addLayout(btnRow);

    // ── Architecture & training config ───────────────────────────────────────
    auto* cfgGroup = new QGroupBox("Architecture & Training Config  (shared across all runs)");
    auto* cfgLay   = new QHBoxLayout(cfgGroup); cfgLay->setSpacing(12);

    auto addCol = [&](const QString& lbl, QWidget* w) {
        auto* col = new QVBoxLayout;
        auto* l = new QLabel(lbl);
        l->setStyleSheet("color:#E8E8F0;font-size:10px;font-weight:600;");
        col->addWidget(l); col->addWidget(w); cfgLay->addLayout(col);
    };

    m_hiddenEdit = new QLineEdit("64,32"); m_hiddenEdit->setFixedWidth(100);
    m_hiddenEdit->setToolTip("Comma-separated hidden layer sizes, e.g. \"64,32\"");
    addCol("Hidden Layers", m_hiddenEdit);

    m_activCombo = new QComboBox; m_activCombo->addItems({"ReLU","Tanh","Sigmoid"}); m_activCombo->setFixedWidth(80);
    addCol("Activation", m_activCombo);

    m_dropoutSpin = new QDoubleSpinBox; m_dropoutSpin->setRange(0,0.9); m_dropoutSpin->setDecimals(2);
    m_dropoutSpin->setSingleStep(0.05); m_dropoutSpin->setValue(0); m_dropoutSpin->setFixedWidth(65);
    addCol("Dropout", m_dropoutSpin);

    m_lrSpin = new QDoubleSpinBox; m_lrSpin->setRange(1e-6,1); m_lrSpin->setDecimals(6);
    m_lrSpin->setSingleStep(0.0001); m_lrSpin->setValue(0.001); m_lrSpin->setFixedWidth(95);
    m_lrSpin->setToolTip("Used by Adam / SGD. Ignored by evolutionary methods.");
    addCol("LR (gradient)", m_lrSpin);

    m_epochsSpin = new QSpinBox; m_epochsSpin->setRange(1,10000); m_epochsSpin->setValue(200); m_epochsSpin->setFixedWidth(70);
    addCol("Epochs / Iters", m_epochsSpin);

    m_batchSpin = new QSpinBox; m_batchSpin->setRange(1,2000); m_batchSpin->setValue(32); m_batchSpin->setFixedWidth(65);
    m_batchSpin->setToolTip("Batch size (Adam/SGD) or Population size (evolutionary).");
    addCol("Batch / Pop.", m_batchSpin);

    m_valSpin = new QDoubleSpinBox; m_valSpin->setRange(0,0.5); m_valSpin->setValue(0.20);
    m_valSpin->setDecimals(2); m_valSpin->setSingleStep(0.05); m_valSpin->setFixedWidth(65);
    addCol("Val Split", m_valSpin);

    m_earlySpin = new QSpinBox; m_earlySpin->setRange(0,500); m_earlySpin->setValue(20); m_earlySpin->setFixedWidth(65);
    m_earlySpin->setToolTip("Early stopping patience (0 = disabled).");
    addCol("Early Stop", m_earlySpin);

    m_wBoundSpin = new QDoubleSpinBox; m_wBoundSpin->setRange(0.5,50); m_wBoundSpin->setValue(5.0);
    m_wBoundSpin->setDecimals(1); m_wBoundSpin->setSingleStep(0.5); m_wBoundSpin->setFixedWidth(65);
    m_wBoundSpin->setToolTip("Weight search bound [-b,+b] for evolutionary methods.");
    addCol("W. Bound", m_wBoundSpin);

    cfgLay->addStretch();
    root->addWidget(cfgGroup);

    // ── Optimizer multi-select ────────────────────────────────────────────────
    auto* optimGroup = new QGroupBox("Optimizers to compare  (each checked optimizer runs on ALL datasets)");
    auto* optimVLay  = new QVBoxLayout(optimGroup);

    // Selection shortcuts
    auto* shortRow = new QHBoxLayout;
    auto mkShort = [&](const QString& t) -> QPushButton* {
        auto* b = new QPushButton(t); b->setFixedHeight(22);
        b->setStyleSheet("QPushButton{border:1px solid palette(mid);border-radius:3px;"
                         "padding:0 8px;font-size:10px;background:palette(button);}");
        return b;
    };
    auto* selAll  = mkShort("All");
    auto* selNone = mkShort("None");
    auto* selGrad = mkShort("Gradient only");
    auto* selEvo  = mkShort("Evolutionary only");
    shortRow->addWidget(new QLabel("Select:"));
    shortRow->addWidget(selAll); shortRow->addWidget(selNone);
    shortRow->addWidget(selGrad); shortRow->addWidget(selEvo);
    shortRow->addStretch();
    optimVLay->addLayout(shortRow);

    // Checkbox grid
    auto* checkGrid = new QGridLayout; checkGrid->setSpacing(4);

    m_optimEntries = {
        // Gradient Descent (gradient=true)
        {"adam",      "Adam",         true,  false, false, nullptr},
        {"sgd",       "SGD",          true,  false, false, nullptr},
        {"adamw",     "AdamW",        true,  false, false, nullptr},
        {"rmsprop",   "RMSProp",      true,  false, false, nullptr},
        {"nadam",     "Nadam",        true,  false, false, nullptr},
        {"adagrad",   "AdaGrad",      true,  false, false, nullptr},
        // Quasi-Newton
        {"lbfgs",     "L-BFGS",       false, true,  false, nullptr},
        {"bfgs",      "BFGS",         false, true,  false, nullptr},
        {"cg",        "CG",           false, true,  false, nullptr},
        {"lm",        "LM",           false, true,  false, nullptr},
        // Single-Point derivative-free
        {"sa",        "SA",           false, false, true,  nullptr},
        {"neldermead","Nelder-Mead",  false, false, true,  nullptr},
        // Evolutionary / Swarm
        {"ga",        "GA",           false, false, false, nullptr},
        {"de",        "DE",           false, false, false, nullptr},
        {"pso",       "PSO",          false, false, false, nullptr},
        {"clpso",     "CLPSO",        false, false, false, nullptr},
        {"acor",      "ACOR",         false, false, false, nullptr},
        {"cmaes",     "CMA-ES",       false, false, false, nullptr},
        {"lmcmaes",   "LM-CMA-ES",    false, false, false, nullptr},
        {"jso",       "JSO",          false, false, false, nullptr},
        {"mlshaderl", "ML-SHADE-RL",  false, false, false, nullptr},
        {"arq3",      "ARQ3",         false, false, false, nullptr},
        // EDA
        {"pbil",      "PBIL",         false, false, false, nullptr},
        {"umda",      "UMDA",         false, false, false, nullptr},
    };

    const QStringList tips = {
        "Adam — gradient descent (recommended default)",
        "SGD — Stochastic Gradient Descent",
        "AdamW — Adam with decoupled weight decay",
        "RMSProp — adaptive LR via exp. moving average of squared gradients",
        "Nadam — Nesterov + Adam",
        "AdaGrad — per-parameter LR that decreases over time",
        "L-BFGS — Limited-memory BFGS quasi-Newton (m=7), uses backprop gradient",
        "BFGS — L-BFGS with m=20, closer to full BFGS",
        "CG — Conjugate Gradient (Polak-Ribière), gradient-based",
        "LM — Levenberg-Marquardt diagonal approximation",
        "SA — Simulated Annealing, single-point, gradient-free",
        "Nelder-Mead — Downhill Simplex, gradient-free",
        "GA — Genetic Algorithm (tournament + uniform crossover + Gaussian mutation)",
        "DE — Differential Evolution",
        "PSO — Particle Swarm Optimization",
        "CLPSO — Comprehensive Learning PSO",
        "ACOR — Ant Colony (continuous)",
        "CMA-ES — Covariance Matrix Adaptation",
        "LM-CMA-ES — Limited Memory CMA-ES",
        "JSO — jSO / L-SHADE Success-History",
        "ML-SHADE-RL — Multi-operator SHADE+RL",
        "ARQ3 — NLPSR+SHADE+jSO+Eigen+TS+OBL",
        "PBIL — Population-Based Incremental Learning (Gaussian)",
        "UMDA — Univariate Marginal Distribution Algorithm",
    };

    auto* gradLabel  = new QLabel("Gradient:");
    gradLabel->setStyleSheet("color:#1D9E75;font-size:10px;font-weight:600;");
    auto* qnLabel    = new QLabel("Quasi-Newton:");
    qnLabel->setStyleSheet("color:#4A9FD8;font-size:10px;font-weight:600;");
    auto* spLabel    = new QLabel("Single-Point:");
    spLabel->setStyleSheet("color:#E0C840;font-size:10px;font-weight:600;");
    auto* evoLabel   = new QLabel("Evolutionary:");
    evoLabel->setStyleSheet("color:#E8700A;font-size:10px;font-weight:600;");
    auto* edaLabel   = new QLabel("EDA:");
    edaLabel->setStyleSheet("color:#C060E0;font-size:10px;font-weight:600;");

    checkGrid->addWidget(gradLabel, 0, 0);
    checkGrid->addWidget(qnLabel,   1, 0);
    checkGrid->addWidget(spLabel,   2, 0);
    checkGrid->addWidget(evoLabel,  3, 0);
    checkGrid->addWidget(edaLabel,  5, 0);

    int gradCol=1, qnCol=1, spCol=1, evoCol=1, evoRow=3, edaCol=1;
    for (int i=0; i<(int)m_optimEntries.size(); ++i) {
        auto& e = m_optimEntries[i];
        e.check = new QCheckBox(e.label);
        e.check->setChecked(e.gradient);  // only Adam+SGD checked by default
        e.check->setStyleSheet("font-size:11px;");
        if (i<(int)tips.size()) e.check->setToolTip(tips[i]);
        if      (e.gradient)    checkGrid->addWidget(e.check, 0, gradCol++);
        else if (e.quasiNewton) checkGrid->addWidget(e.check, 1, qnCol++);
        else if (e.singlePoint) checkGrid->addWidget(e.check, 2, spCol++);
        else if (e.id=="pbil"||e.id=="umda") checkGrid->addWidget(e.check, 5, edaCol++);
        else {
            if (evoCol > 6) { ++evoRow; evoCol=1; }
            checkGrid->addWidget(e.check, evoRow, evoCol++);
        }
    }
    optimVLay->addLayout(checkGrid);
    root->addWidget(optimGroup);

    // Shortcut connections
    connect(selAll,  &QPushButton::clicked, this, &BatchPanel::onSelectAll);
    connect(selNone, &QPushButton::clicked, this, &BatchPanel::onSelectNone);
    connect(selGrad, &QPushButton::clicked, this, &BatchPanel::onSelectGradient);
    connect(selEvo,  &QPushButton::clicked, this, &BatchPanel::onSelectEvolutionary);

    // ── Status + progress ────────────────────────────────────────────────────
    m_statusLbl = new QLabel("Queue is empty. Add datasets to begin.");
    m_statusLbl->setStyleSheet("font-size:12px;color:#C8C8D8;");

    m_timeLbl = new QLabel;
    m_timeLbl->setStyleSheet("QLabel{font-family:monospace;font-size:11px;color:#1D9E75;"
                             "background:palette(midlight);border-radius:4px;padding:2px 8px;}");
    m_timeLbl->hide();

    m_overall = new QProgressBar; m_overall->setFixedHeight(18);
    m_overall->setRange(0,100); m_overall->setValue(0);
    m_overall->setFormat("  %v / %m jobs  (%p%)");
    m_overall->setStyleSheet(
        "QProgressBar{border-radius:6px;background:#1e2030;text-align:center;"
        "color:#E8E8F0;font-size:11px;font-weight:600;}"
        "QProgressBar::chunk{border-radius:6px;background:qlineargradient("
        "x1:0,y1:0,x2:1,y2:0,stop:0 #1D9E75,stop:1 #14C98C);}");
    m_overall->hide();

    auto* stRow = new QHBoxLayout;
    stRow->addWidget(m_statusLbl,1); stRow->addWidget(m_timeLbl);
    root->addLayout(stRow);
    root->addWidget(m_overall);

    m_tickTimer = new QTimer(this); m_tickTimer->setInterval(500);
    connect(m_tickTimer, &QTimer::timeout, this, &BatchPanel::onTimerTick);

    // ── Queue + Results tables ────────────────────────────────────────────────
    auto* tablesRow = new QHBoxLayout; tablesRow->setSpacing(10);

    auto* qGroup = new QGroupBox("Queue");
    auto* qLay   = new QVBoxLayout(qGroup);
    m_queueTable = new QTableWidget;
    m_queueTable->setColumnCount(4);
    m_queueTable->setHorizontalHeaderLabels({"#", "Dataset", "Optimizer", "Status"});
    m_queueTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_queueTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_queueTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    m_queueTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    m_queueTable->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    m_queueTable->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    m_queueTable->verticalHeader()->hide();
    m_queueTable->verticalHeader()->setDefaultSectionSize(22);
    m_queueTable->setMinimumWidth(320);
    qLay->addWidget(m_queueTable);
    tablesRow->addWidget(qGroup, 1);

    // Results table
    // ── Results area: tabbed ──────────────────────────────────────────────────
    m_tabWidget = new QTabWidget;
    m_tabWidget->setTabPosition(QTabWidget::North);

    // ── TAB 1: Results ────────────────────────────────────────────────────────
    auto* resultsPage = new QWidget;
    auto* rpl = new QVBoxLayout(resultsPage);
    rpl->setContentsMargins(4,4,4,4); rpl->setSpacing(4);

    // Export button inside Results tab
    auto* rTopRow = new QHBoxLayout;
    rTopRow->addStretch();
    m_exportBtn = new QPushButton("Export XLSX");
    m_exportBtn->setMinimumSize(140, 28);
    m_exportBtn->setEnabled(false);
    m_exportBtn->setStyleSheet(
        "QPushButton:enabled{background:#185fa5;color:white;border-radius:5px;"
        "font-weight:600;padding:0 14px;}"
        "QPushButton:disabled{background:palette(midlight);color:palette(mid);}");
    rTopRow->addWidget(m_exportBtn);
    rpl->addLayout(rTopRow);

    // Results table
    m_resultsTable = new QTableWidget;
    const QStringList kHeaders = {
        "Dataset", "Optimizer", "Task", "N", "Inputs", "Classes",
        "Epochs", "Time(s)",
        "Train Err%", "Val Err%", "Test Err%",
        "Train MAE", "Val MAE", "Test MAE",
        "Train RMSE","Val RMSE","Test RMSE",
        "Train R²",  "Val R²",  "Test R²",
        "Best Epoch", "Status"
    };
    m_resultsTable->setColumnCount(kHeaders.size());
    m_resultsTable->setHorizontalHeaderLabels(kHeaders);
    m_resultsTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_resultsTable->setAlternatingRowColors(true);
    m_resultsTable->setStyleSheet(
        "QTableWidget{background:#1f1f2a;alternate-background-color:#262635;"
        "gridline-color:#3a3a4a;color:#E4E4EC;}"
        "QHeaderView::section{background:#2a2a38;color:#E8E8F0;padding:6px;"
        "border:0;border-right:1px solid #3a3a4a;font-weight:600;}");
    m_resultsTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Interactive);
    m_resultsTable->horizontalHeader()->setDefaultSectionSize(90);
    m_resultsTable->verticalHeader()->setDefaultSectionSize(24);
    m_resultsTable->setSortingEnabled(true);
    m_resultsTable->horizontalHeader()->setSortIndicatorShown(true);

    // Column tooltips
    const QStringList colTips = {
        "Dataset name","Optimizer used","Task type","Training samples",
        "Input features","Classes (classification only)",
        "Epochs trained","Duration in seconds",
        "Train Error % — lower is better","Val Error %","Test Error %",
        "Mean Absolute Error — lower is better","Val MAE","Test MAE",
        "Root Mean Squared Error","Val RMSE","Test RMSE",
        "R² (1=perfect, 0=baseline, <0=worse)","Val R²","Test R²",
        "Best epoch by val loss","Status"
    };
    for (int i=0; i<colTips.size() && i<m_resultsTable->columnCount(); ++i) {
        auto* h = m_resultsTable->horizontalHeaderItem(i);
        if (h) h->setToolTip(colTips[i]);
    }
    rpl->addWidget(m_resultsTable, 1);

    m_tabWidget->addTab(resultsPage, "📋  Results");

    // ── TAB 2: Wilcoxon ───────────────────────────────────────────────────────
    m_wilcoxon = new WilcoxonTab;
    m_tabWidget->addTab(m_wilcoxon, "📊  Wilcoxon");
    m_tabWidget->setTabEnabled(1, false);

    // ── TAB 3: Friedman ───────────────────────────────────────────────────────
    m_friedman = new FriedmanTab;
    m_tabWidget->addTab(m_friedman, "📈  Friedman");
    m_tabWidget->setTabEnabled(2, false);

    tablesRow->addWidget(m_tabWidget, 3);
    root->addLayout(tablesRow, 1);

    // Connections
    connect(m_addBtn,    &QPushButton::clicked, this, &BatchPanel::onAddFiles);
    connect(m_clearBtn,  &QPushButton::clicked, this, &BatchPanel::onClearQueue);
    connect(m_runBtn,    &QPushButton::clicked, this, &BatchPanel::onRunBatch);
    connect(m_stopBtn,   &QPushButton::clicked, this, &BatchPanel::onStopBatch);
    connect(m_exportBtn, &QPushButton::clicked, this, &BatchPanel::onExportXlsx);
}

// ─── Optimizer selection shortcuts ───────────────────────────────────────────
QStringList BatchPanel::selectedOptimizerIds() const {
    QStringList ids;
    for (const auto& e : m_optimEntries)
        if (e.check && e.check->isChecked()) ids << e.id;
    return ids;
}
void BatchPanel::onSelectAll()         { for (auto& e:m_optimEntries) e.check->setChecked(true);  }
void BatchPanel::onSelectNone()        { for (auto& e:m_optimEntries) e.check->setChecked(false); }
void BatchPanel::onSelectGradient()    { for (auto& e:m_optimEntries) e.check->setChecked(e.gradient||e.quasiNewton); }
void BatchPanel::onSelectEvolutionary(){ for (auto& e:m_optimEntries) e.check->setChecked(!e.gradient&&!e.quasiNewton&&!e.singlePoint); }

// ─── Queue management ────────────────────────────────────────────────────────
void BatchPanel::onAddFiles() {
    const QStringList paths = QFileDialog::getOpenFileNames(
        this, "Add datasets to queue", {}, DatasetLoader::fileDialogFilter());
    for (const QString& p : paths)
        if (!m_queue.contains(p)) m_queue << p;
    refreshQueueTable();
    m_statusLbl->setText(QString("%1 dataset(s) queued.").arg(m_queue.size()));
}

void BatchPanel::onClearQueue() {
    if (m_running) return;
    m_queue.clear(); m_results.clear();
    m_queueTable->setRowCount(0); m_resultsTable->setRowCount(0);
    m_exportBtn->setEnabled(false);
    m_tabWidget->setTabEnabled(1, false);
    m_tabWidget->setTabEnabled(2, false);
    m_tabWidget->setTabText(1, "\xF0\x9F\x93\x8A  Wilcoxon");
    m_tabWidget->setTabText(2, "\xF0\x9F\x93\x88  Friedman");
    m_tabWidget->setCurrentIndex(0);
    m_statusLbl->setText("Queue cleared.");
}

void BatchPanel::refreshQueueTable() {
    // In "pre-run" state show datasets only; during run the table is populated per-job
    m_queueTable->setRowCount(static_cast<int>(m_queue.size()));
    for (int i = 0; i < static_cast<int>(m_queue.size()); ++i) {
        auto* n = new QTableWidgetItem(QString::number(i+1)); n->setTextAlignment(Qt::AlignCenter);
        m_queueTable->setItem(i,0,n);
        auto* name = new QTableWidgetItem(QFileInfo(m_queue[i]).fileName());
        name->setToolTip(m_queue[i]);
        m_queueTable->setItem(i,1,name);
        m_queueTable->setItem(i,2, new QTableWidgetItem("—"));
        m_queueTable->setItem(i,3, new QTableWidgetItem("Queued"));
    }
}

// ─── Run ─────────────────────────────────────────────────────────────────────
void BatchPanel::onRunBatch() {
    if (m_running) return;
    if (m_queue.isEmpty()) {
        QMessageBox::information(this,"Batch Run","Queue is empty. Add files first."); return;
    }
    const QStringList optims = selectedOptimizerIds();
    if (optims.isEmpty()) {
        QMessageBox::information(this,"Batch Run","No optimizer selected. Check at least one."); return;
    }

    m_results.clear(); m_resultsTable->setRowCount(0); m_exportBtn->setEnabled(false);

    // Build all jobs: dataset × optimizer
    std::vector<BatchJobConfig> jobs;
    for (const QString& ds : m_queue)
        for (const QString& opt : optims)
            jobs.push_back(makeJobTemplate(ds, opt));

    // Populate queue table to show all planned jobs
    m_queueTable->setRowCount(static_cast<int>(jobs.size()));
    for (int i = 0; i < static_cast<int>(jobs.size()); ++i) {
        auto n = new QTableWidgetItem(QString::number(i+1)); n->setTextAlignment(Qt::AlignCenter);
        m_queueTable->setItem(i,0,n);
        m_queueTable->setItem(i,1, new QTableWidgetItem(QFileInfo(jobs[i].datasetPath).fileName()));
        m_queueTable->setItem(i,2, new QTableWidgetItem(jobs[i].optimizerName.toUpper()));
        m_queueTable->setItem(i,3, new QTableWidgetItem("Queued"));
    }

    setRunning(true);
    m_jobsDone = 0;
    m_batchClock.start(); m_tickTimer->start();
    m_timeLbl->setText("Elapsed: 0:00"); m_timeLbl->show();
    m_overall->setRange(0, static_cast<int>(jobs.size())); m_overall->setValue(0); m_overall->show();

    m_thread = new QThread(this);
    m_runner = new BatchRunner;
    m_runner->setJobs(jobs);
    m_runner->moveToThread(m_thread);

    connect(m_thread, &QThread::started,             m_runner, &BatchRunner::run);
    connect(m_runner, &BatchRunner::jobStarted,      this, &BatchPanel::onJobStarted,      Qt::QueuedConnection);
    connect(m_runner, &BatchRunner::jobProgress,     this, &BatchPanel::onJobProgress,     Qt::QueuedConnection);
    connect(m_runner, &BatchRunner::jobCompleted,    this, &BatchPanel::onJobCompleted,    Qt::QueuedConnection);
    connect(m_runner, &BatchRunner::allJobsFinished, this, &BatchPanel::onAllJobsFinished, Qt::QueuedConnection);
    connect(m_runner, &BatchRunner::batchError,      this, &BatchPanel::onBatchError,      Qt::QueuedConnection);
    connect(m_runner, &BatchRunner::allJobsFinished, m_thread, &QThread::quit);
    connect(m_runner, &BatchRunner::batchError,      m_thread, &QThread::quit);
    connect(m_thread, &QThread::finished,            m_runner, &QObject::deleteLater);
    m_thread->start();
}

void BatchPanel::onStopBatch() { if (m_runner) m_runner->requestStop(); }

BatchJobConfig BatchPanel::makeJobTemplate(const QString& dsPath,
                                            const QString& optimId) const {
    BatchJobConfig cfg;
    cfg.datasetPath   = dsPath;
    cfg.optimizerName = optimId;
    cfg.weightBound   = m_wBoundSpin->value();

    const QStringList tokens = m_hiddenEdit->text().split(',', Qt::SkipEmptyParts);
    for (const QString& t : tokens) {
        bool ok; int n = t.trimmed().toInt(&ok);
        if (ok && n > 0) cfg.hiddenSizes.push_back(n);
    }
    if (cfg.hiddenSizes.empty()) cfg.hiddenSizes = {64,32};
    const Activation act = activationFromName(m_activCombo->currentText());
    for (size_t i=0; i<cfg.hiddenSizes.size(); ++i) cfg.hiddenActivations.push_back(act);
    cfg.dropoutRate = m_dropoutSpin->value();

    const QString oid = optimId.toLower();
    cfg.trainerCfg.optimizer    = (oid=="sgd") ? OptimizerType::SGD : OptimizerType::Adam;
    cfg.trainerCfg.learningRate = m_lrSpin->value();
    cfg.trainerCfg.epochs       = m_epochsSpin->value();
    cfg.trainerCfg.batchSize    = m_batchSpin->value();
    cfg.trainerCfg.validationSplit = m_valSpin->value();
    cfg.trainerCfg.earlyStoppingPatience = m_earlySpin->value();
    return cfg;
}

// ─── Job signal handlers ─────────────────────────────────────────────────────
void BatchPanel::onJobStarted(int index, QString name) {
    if (index < m_queueTable->rowCount()) {
        auto* it = new QTableWidgetItem("Running\u2026"); it->setForeground(QColor("#E8700A"));
        m_queueTable->setItem(index, 3, it);
    }
    m_statusLbl->setText(QString("[%1/%2] %3 \u2014 %4")
        .arg(index+1).arg(m_overall->maximum()).arg(name)
        .arg(m_queueTable->item(index,2) ? m_queueTable->item(index,2)->text() : ""));
}

void BatchPanel::onJobProgress(int index, int epoch, int total, double loss) {
    if (index < m_queueTable->rowCount()) {
        auto* it = new QTableWidgetItem(QString("Iter %1/%2  loss %3")
            .arg(epoch).arg(total).arg(loss,0,'f',4));
        it->setForeground(QColor("#E8700A"));
        m_queueTable->setItem(index,3,it);
    }
}

void BatchPanel::onJobCompleted(int index, BatchJobResult r) {
    m_results.push_back(r); ++m_jobsDone;
    if (index < m_queueTable->rowCount()) {
        auto* it = new QTableWidgetItem(r.success ? "Done" : "FAILED");
        it->setForeground(r.success ? QColor("#1D9E75") : QColor("#C0392B"));
        QFont f=it->font(); f.setBold(true); it->setFont(f);
        m_queueTable->setItem(index,3,it);
    }
    appendResultRow(r); m_overall->setValue(index+1);
}

void BatchPanel::onAllJobsFinished() {
    m_tickTimer->stop(); onTimerTick();
    setRunning(false);
    m_statusLbl->setText(QString("Batch complete \u2014 %1 runs finished.").arg(m_results.size()));
    m_exportBtn->setEnabled(!m_results.empty());

    // Feed results to statistical tabs
    m_wilcoxon->refresh(m_results);
    m_friedman->refresh(m_results);

    // Enable tabs when there is enough data for meaningful analysis
    // Wilcoxon: ≥2 optimizers, ≥2 common datasets
    m_tabWidget->setTabEnabled(1, m_wilcoxon->hasValidData());
    // Friedman: ≥3 optimizers, ≥2 common datasets (checked inside FriedmanTab)
    m_tabWidget->setTabEnabled(2, m_friedman->hasValidData());

    // Suffix tab titles with data availability indicator
    m_tabWidget->setTabText(1, m_wilcoxon->hasValidData()
                             ? "\xF0\x9F\x93\x8A  Wilcoxon \xe2\x9c\x93"
                             : "\xF0\x9F\x93\x8A  Wilcoxon");
    m_tabWidget->setTabText(2, m_friedman->hasValidData()
                             ? "\xF0\x9F\x93\x88  Friedman \xe2\x9c\x93"
                             : "\xF0\x9F\x93\x88  Friedman");
}

void BatchPanel::onBatchError(QString msg) {
    m_tickTimer->stop(); setRunning(false);
    QMessageBox::critical(this,"Batch Error",msg);
    m_statusLbl->setText("Error: "+msg);
}

// ─── Timer tick ──────────────────────────────────────────────────────────────
void BatchPanel::onTimerTick() {
    if (!m_running) return;
    const int sec = static_cast<int>(m_batchClock.elapsed()/1000);
    QString eta;
    const int total = m_overall->maximum();
    if (m_jobsDone>0 && m_jobsDone<total) {
        const int etaSec = static_cast<int>(
            (double)m_batchClock.elapsed()/m_jobsDone*(total-m_jobsDone)/1000.0);
        eta = QString("  |  ETA %1m%2s").arg(etaSec/60).arg(etaSec%60,2,10,QChar('0'));
    }
    m_timeLbl->setText(QString("Elapsed  %1:%2%3")
        .arg(sec/60).arg(sec%60,2,10,QChar('0')).arg(eta));
}

// ─── setRunning ──────────────────────────────────────────────────────────────
void BatchPanel::setRunning(bool r) {
    m_running = r;
    m_runBtn->setEnabled(!r); m_stopBtn->setEnabled(r);
    m_addBtn->setEnabled(!r); m_clearBtn->setEnabled(!r);
    for (auto* w : {(QWidget*)m_hiddenEdit,(QWidget*)m_activCombo,(QWidget*)m_dropoutSpin,
                    (QWidget*)m_lrSpin,(QWidget*)m_epochsSpin,(QWidget*)m_batchSpin,
                    (QWidget*)m_valSpin,(QWidget*)m_earlySpin,(QWidget*)m_wBoundSpin})
        w->setEnabled(!r);
    for (auto& e : m_optimEntries) e.check->setEnabled(!r);
    if (!r) { m_overall->hide(); } else { m_exportBtn->setEnabled(false); }
}

void BatchPanel::setResultsSortingEnabled(bool on) { m_resultsTable->setSortingEnabled(on); }

// ─── appendResultRow ─────────────────────────────────────────────────────────
void BatchPanel::appendResultRow(const BatchJobResult& r) {
    setResultsSortingEnabled(false);
    const int row = m_resultsTable->rowCount();
    m_resultsTable->insertRow(row);
    int c = 0;

    const QColor errBg(255,245,225); const QColor regBg(225,240,255);
    const bool isCls  = (r.task != TaskType::Regression);
    const bool isRegr = (r.task == TaskType::Regression);

    auto putStr = [&](const QString& t, const QColor& bg=QColor()) {
        auto* it=new QTableWidgetItem(t); it->setTextAlignment(Qt::AlignCenter);
        if (bg.isValid()) {
            const QBrush b(bg),f(QColor(20,30,50));
            it->setBackground(b); it->setData(Qt::BackgroundRole,b);
            it->setForeground(f); it->setData(Qt::ForegroundRole,f);
            QFont fnt=it->font(); fnt.setBold(true); it->setFont(fnt);
        }
        m_resultsTable->setItem(row,c++,it);
    };
    auto putNum=[&](double v,const QString& d,const QColor& bg=QColor()){
        m_resultsTable->setItem(row,c++,new NumericItem(v,d,bg));
    };
    auto putNA=[&](){ m_resultsTable->setItem(row,c++,new NumericItem(NumericItem::kNA,"—")); };
    auto putNAC=[&](const QColor& bg){
        auto* it=new NumericItem(NumericItem::kNA,"—");
        const QBrush b(bg),f(QColor(90,100,120));
        it->setBackground(b); it->setData(Qt::BackgroundRole,b);
        it->setForeground(f); it->setData(Qt::ForegroundRole,f);
        m_resultsTable->setItem(row,c++,it);
    };

    auto errCell=[&](double v){ if(isCls&&v>=0) putNum(v,QString::number(v,'f',2)+"%",errBg); else putNAC(errBg); };
    auto maeCell=[&](double v){ if(isRegr&&v>=0) putNum(v,QString::number(v,'f',4),regBg); else putNAC(regBg); };
    auto r2Cell =[&](double v){ if(isRegr) putNum(v,QString::number(v,'f',4),regBg); else putNAC(regBg); };
    auto testErr=[&](double v){ if(r.hasTestSet) errCell(v); else putNAC(errBg); };
    auto testMAE=[&](double v){ if(r.hasTestSet) maeCell(v); else putNAC(regBg); };
    auto testR2 =[&](double v){ if(r.hasTestSet) r2Cell(v);  else putNAC(regBg); };

    putStr(r.datasetName);
    putStr(r.optimizerName.toUpper());
    putStr(taskName(r.task));
    putNum(r.samples, QString::number(r.samples));
    putNum(r.inputs,  QString::number(r.inputs));
    r.numClasses>0 ? putNum(r.numClasses,QString::number(r.numClasses)) : putNA();
    putNum(r.epochsRun, QString::number(r.epochsRun));
    putNum(r.durationMs/1000.0, QString::number(r.durationMs/1000.0,'f',1));
    errCell(r.trainErrorPct); errCell(r.valErrorPct); testErr(r.testErrorPct);
    maeCell(r.trainMAE);  maeCell(r.valMAE);  testMAE(r.testMAE);
    maeCell(r.trainRMSE); maeCell(r.valRMSE); testMAE(r.testRMSE);
    r2Cell(r.trainR2);    r2Cell(r.valR2);    testR2(r.testR2);
    putNum(r.bestEpoch, QString::number(r.bestEpoch));
    putStr(r.success ? "OK" : "FAIL: "+r.errorMsg,
           r.success ? QColor(210,245,225) : QColor(255,215,215));

    setResultsSortingEnabled(true);
}

// ─── XLSX Export ─────────────────────────────────────────────────────────────
void BatchPanel::onExportXlsx() {
    if (m_results.empty()) return;
    const QString path = QFileDialog::getSaveFileName(
        this,"Export Batch Results","batch_results.xlsx", XlsxWriter::fileFilter());
    if (path.isEmpty()) return;

    try {
        XlsxWriter wb;

        // ── Helper: empty cell for N/A ────────────────────────────────────────
        auto optD = [](double v, double sentinel=-1.0) -> XlsxCell {
            return (v <= sentinel) ? XlsxCell() : XlsxCell(v);
        };

        // ── Shared base columns ───────────────────────────────────────────────
        auto baseRow = [&](const BatchJobResult& r) -> std::vector<XlsxCell> {
            return {
                r.datasetName, r.optimizerName.toUpper(), taskName(r.task),
                r.samples, r.inputs,
                r.epochsRun, r.durationMs/1000.0,
                r.finalTrainLoss,
                r.finalValLoss >= 0 ? XlsxCell(r.finalValLoss) : XlsxCell(),
                r.bestValLoss, r.bestEpoch,
                QString(r.hasTestSet ? "yes" : "no"),
                QString(r.success ? "OK" : "FAIL")
            };
        };

        // ── SHEET 1: Regression ───────────────────────────────────────────────
        {
            auto* s = wb.addSheet("Regression");
            s->setHeader({"Dataset","Optimizer","Task","N","Inputs","Epochs","Time(s)",
                          "Train Loss","Val Loss","Best Val Loss","Best Epoch","Has Test","Status",
                          "Train MAE","Val MAE","Test MAE",
                          "Train RMSE","Val RMSE","Test RMSE",
                          "Train R\u00B2","Val R\u00B2","Test R\u00B2"});
            for (const auto& r : m_results) {
                if (r.task != TaskType::Regression) continue;
                auto row = baseRow(r);
                row.push_back(optD(r.trainMAE));
                row.push_back(r.hasTestSet ? optD(r.valMAE)  : XlsxCell());
                row.push_back(r.hasTestSet ? optD(r.testMAE) : XlsxCell());
                row.push_back(optD(r.trainRMSE));
                row.push_back(r.hasTestSet ? optD(r.valRMSE)  : XlsxCell());
                row.push_back(r.hasTestSet ? optD(r.testRMSE) : XlsxCell());
                row.push_back(optD(r.trainR2, -999.0));
                row.push_back(r.hasTestSet ? optD(r.valR2,-999.0) : XlsxCell());
                row.push_back(r.hasTestSet ? optD(r.testR2,-999.0): XlsxCell());
                s->addRow(row);
            }
        }

        // ── SHEET 2: Binary Classification ────────────────────────────────────
        {
            auto* s = wb.addSheet("Binary Classification");
            s->setHeader({"Dataset","Optimizer","Task","N","Inputs","Epochs","Time(s)",
                          "Train Loss","Val Loss","Best Val Loss","Best Epoch","Has Test","Status",
                          "Train Err%","Val Err%","Test Err%",
                          "Train Acc","Val Acc","Test Acc"});
            for (const auto& r : m_results) {
                if (r.task != TaskType::BinaryClassification) continue;
                auto row = baseRow(r);
                row.push_back(optD(r.trainErrorPct));
                row.push_back(optD(r.valErrorPct));
                row.push_back(r.hasTestSet ? optD(r.testErrorPct) : XlsxCell());
                row.push_back(r.trainAcc>=0 ? XlsxCell(r.trainAcc*100.0) : XlsxCell());
                row.push_back(r.valAcc  >=0 ? XlsxCell(r.valAcc  *100.0) : XlsxCell());
                row.push_back(r.hasTestSet && r.testAcc>=0 ? XlsxCell(r.testAcc*100.0) : XlsxCell());
                s->addRow(row);
            }
        }

        // ── SHEET 3: Multi-class Classification ───────────────────────────────
        {
            auto* s = wb.addSheet("Multi-class");
            s->setHeader({"Dataset","Optimizer","Task","N","Inputs","Classes","Epochs","Time(s)",
                          "Train Loss","Val Loss","Best Val Loss","Best Epoch","Has Test","Status",
                          "Train Err%","Val Err%","Test Err%",
                          "Train Acc","Val Acc","Test Acc"});
            for (const auto& r : m_results) {
                if (r.task != TaskType::MultiClassClassification) continue;
                auto row = baseRow(r);
                // Insert Classes after Inputs (position 5)
                row.insert(row.begin()+5, XlsxCell(r.numClasses));
                row.push_back(optD(r.trainErrorPct));
                row.push_back(optD(r.valErrorPct));
                row.push_back(r.hasTestSet ? optD(r.testErrorPct) : XlsxCell());
                row.push_back(r.trainAcc>=0 ? XlsxCell(r.trainAcc*100.0) : XlsxCell());
                row.push_back(r.valAcc  >=0 ? XlsxCell(r.valAcc  *100.0) : XlsxCell());
                row.push_back(r.hasTestSet && r.testAcc>=0 ? XlsxCell(r.testAcc*100.0) : XlsxCell());
                s->addRow(row);
            }
        }

        // ── SHEET 4: All Results (unified) ────────────────────────────────────
        {
            auto* s = wb.addSheet("All Results");
            s->setHeader({"Dataset","Optimizer","Task","N","Inputs","Classes",
                          "Epochs","Time(s)","Train Loss","Val Loss","Best Val Loss","Best Epoch",
                          "Has Test","Status",
                          "Train Err%","Val Err%","Test Err%",
                          "Train MAE","Val MAE","Test MAE",
                          "Train RMSE","Val RMSE","Test RMSE",
                          "Train R\u00B2","Val R\u00B2","Test R\u00B2"});
            for (const auto& r : m_results) {
                auto row = baseRow(r);
                // Insert Classes after Inputs
                row.insert(row.begin()+5,
                    r.numClasses>0 ? XlsxCell(r.numClasses) : XlsxCell());
                row.push_back(optD(r.trainErrorPct));
                row.push_back(optD(r.valErrorPct));
                row.push_back(r.hasTestSet ? optD(r.testErrorPct) : XlsxCell());
                row.push_back(optD(r.trainMAE));
                row.push_back(optD(r.valMAE));
                row.push_back(r.hasTestSet ? optD(r.testMAE)  : XlsxCell());
                row.push_back(optD(r.trainRMSE));
                row.push_back(optD(r.valRMSE));
                row.push_back(r.hasTestSet ? optD(r.testRMSE) : XlsxCell());
                row.push_back(optD(r.trainR2,-999.0));
                row.push_back(optD(r.valR2,  -999.0));
                row.push_back(r.hasTestSet ? optD(r.testR2,-999.0) : XlsxCell());
                s->addRow(row);
            }
        }

        // ── SHEET 5: Config ───────────────────────────────────────────────────
        {
            auto* s = wb.addSheet("Config");
            s->setHeader({"Parameter","Value"});
            s->addRow({QString("Hidden Layers"),  QString(m_hiddenEdit->text())});
            s->addRow({QString("Activation"),     QString(m_activCombo->currentText())});
            s->addRow({QString("Dropout"),        m_dropoutSpin->value()});
            s->addRow({QString("Learning Rate"),  m_lrSpin->value()});
            s->addRow({QString("Epochs"),         m_epochsSpin->value()});
            s->addRow({QString("Batch / Pop."),   m_batchSpin->value()});
            s->addRow({QString("Val Split"),      m_valSpin->value()});
            s->addRow({QString("Early Stop"),     m_earlySpin->value()});
            s->addRow({QString("Weight Bound"),   m_wBoundSpin->value()});
            s->addRow({QString("Optimizers run"), QString(selectedOptimizerIds().join(", "))});
            s->addRow({QString("Total jobs"),     (int)m_results.size()});
        }

        wb.save(path);
        QMessageBox::information(this,"Export Complete",
            QString("Saved %1 results to:\n%2\n\nSheets: Regression | Binary | Multi-class | All Results | Config")
                .arg(m_results.size()).arg(path));
        m_statusLbl->setText("Exported: " + path);

    } catch (const std::exception& ex) {
        QMessageBox::critical(this,"Export Error",QString("Failed:\n%1").arg(ex.what()));
    }
}

// ─── Drag & drop ─────────────────────────────────────────────────────────────
void BatchPanel::dragEnterEvent(QDragEnterEvent* e) {
    if (e->mimeData()->hasUrls()) e->acceptProposedAction();
}
void BatchPanel::dropEvent(QDropEvent* e) {
    for (const auto& url : e->mimeData()->urls()) {
        const QString p = url.toLocalFile();
        if (!p.isEmpty() && !m_queue.contains(p)) m_queue << p;
    }
    refreshQueueTable();
    m_statusLbl->setText(QString("%1 dataset(s) queued.").arg(m_queue.size()));
}

} // namespace NeuralStudio
