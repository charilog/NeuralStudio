#include "EvolBoundsPanel.h"
#include "core/export/XlsxWriter.h"
#include "core/dataset/DatasetLoader.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QFileDialog>
#include <QMessageBox>
#include <QPainter>
#include <QHeaderView>
#include <QFileInfo>
#include <QFrame>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QMimeData>
#include <QChart>
#include <QChartView>
#include <QLineSeries>
#include <QAreaSeries>
#include <QValueAxis>
#include <QLegend>
#include <QImage>
#include <algorithm>
#include <cmath>

namespace NeuralStudio {

// ── Statistics helpers ────────────────────────────────────────────────────────
static std::pair<double,double> runStats(const QVector<double>& v) {
    if (v.isEmpty()) return {0.0, 0.0};
    double sum = 0;
    for (double x : v) sum += x;
    const double mean = sum / v.size();
    if (v.size() == 1) return {mean, 0.0};
    double var = 0;
    for (double x : v) var += (x - mean) * (x - mean);
    return {mean, std::sqrt(var / (v.size() - 1))};
}
static QString fmtMS(const QVector<double>& v, char fmt = 'f', int prec = 2) {
    if (v.isEmpty()) return QStringLiteral("-");
    auto [mean, sd] = runStats(v);
    if (v.size() == 1)
        return QString::number(mean, fmt, prec);
    return QString("%1 \u00b1 %2")
        .arg(QString::number(mean, fmt, prec))
        .arg(QString::number(sd,   fmt, prec));
}

// ── Optimizer combo helper ────────────────────────────────────────────────────
static QComboBox* makeOptCombo(bool allowNone, bool allowGE, QWidget* p = nullptr) {
    auto* c = new QComboBox(p);
    if (allowNone) { c->addItem("— None (skip phase) —","none"); c->insertSeparator(1); }
    PhaseBox::fillOptimizerCombo(c, false);
    if (allowGE) { c->insertSeparator(c->count());
                   c->addItem("GE Grammar Bounds  (BDCC 2025)","ge_grammar"); }
    return c;
}

// ── Constructor ───────────────────────────────────────────────────────────────
EvolBoundsPanel::EvolBoundsPanel(QWidget* parent) : QWidget(parent) {
    setAcceptDrops(true);
    buildUi();
}

void EvolBoundsPanel::dragEnterEvent(QDragEnterEvent* e) {
    if (e->mimeData()->hasUrls()) e->acceptProposedAction();
}
void EvolBoundsPanel::dropEvent(QDropEvent* e) {
    QStringList paths;
    for (const auto& url : e->mimeData()->urls()) {
        const QString p = url.toLocalFile();
        if (!p.isEmpty()) paths << p;
    }
    addToQueue(paths);
}

// ── buildUi ───────────────────────────────────────────────────────────────────
void EvolBoundsPanel::buildUi() {
    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(20,14,20,14);
    root->setSpacing(8);

    auto* title = new QLabel("Evolutionary Bounds Training");
    { QFont f=title->font(); f.setPointSize(14); f.setBold(true); title->setFont(f); }
    root->addWidget(title);
    auto* sub = new QLabel(
        "<b>Phase 1</b>: penalised optimizer discovers per-parameter bounds.  "
        "<b>Phase 2</b>: trains within those bounds.  "
        "<b>Phase 3</b>: final training (optional).  "
        "Each dataset also runs a <b>baseline</b> for comparison.  "
        "<b>N independent runs</b> (mean \u00b1 std) supported.  "
        "<i>Foundations 2025 / BDCC 2025</i>");
    sub->setWordWrap(true);
    sub->setStyleSheet("color:palette(mid);font-size:12px;");
    root->addWidget(sub);

    buildNetworkGroup();
    buildUi_queueAndNetwork(root);
    buildConfigArea(root);
    buildTabWidget(root);
}

// ── Dataset queue + Network & Training, side by side ────────────────────────
void EvolBoundsPanel::buildUi_queueAndNetwork(QVBoxLayout* root) {
    auto* row = new QHBoxLayout; row->setSpacing(10);
    QGroupBox* queueBox = buildQueueGroup();
    row->addWidget(queueBox, 2);      // dataset queue: a bit wider
    row->addWidget(m_netGrp, 1);      // Network & Training: to its right
    root->addLayout(row);
}

// ── Dataset queue ─────────────────────────────────────────────────────────────
QGroupBox* EvolBoundsPanel::buildQueueGroup() {
    auto* grp = new QGroupBox("Dataset Queue  (drag && drop files here)");
    auto* lay = new QVBoxLayout(grp); lay->setSpacing(6);

    auto* btnRow = new QHBoxLayout;
    auto mkBtn = [&](const QString& t, const QString& bg = "") -> QPushButton* {
        auto* b = new QPushButton(t); b->setMinimumHeight(28);
        if (!bg.isEmpty())
            b->setStyleSheet(QString("QPushButton{background:%1;color:white;border-radius:4px;"
                "font-weight:600;padding:0 12px;}QPushButton:disabled{background:palette(midlight);}").arg(bg));
        return b;
    };
    m_addBtn    = mkBtn("Add Files…");
    m_removeBtn = mkBtn("Remove Selected");
    m_clearBtn  = mkBtn("Clear Queue","#8B3030");
    btnRow->addWidget(m_addBtn); btnRow->addWidget(m_removeBtn);
    btnRow->addWidget(m_clearBtn); btnRow->addStretch();
    lay->addLayout(btnRow);
    connect(m_addBtn,    &QPushButton::clicked, this, &EvolBoundsPanel::onAddFiles);
    connect(m_removeBtn, &QPushButton::clicked, this, &EvolBoundsPanel::onRemoveSelected);
    connect(m_clearBtn,  &QPushButton::clicked, this, &EvolBoundsPanel::onClearQueue);

    m_queueTable = new QTableWidget(0,3);
    m_queueTable->setHorizontalHeaderLabels({"#","Dataset","Status"});
    m_queueTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_queueTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_queueTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    m_queueTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    m_queueTable->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    m_queueTable->verticalHeader()->hide();
    m_queueTable->verticalHeader()->setDefaultSectionSize(22);
    m_queueTable->setMinimumHeight(90);
    m_queueTable->setMaximumHeight(160);
    lay->addWidget(m_queueTable);
    return grp;
}

// ── Phase config ──────────────────────────────────────────────────────────────
void EvolBoundsPanel::buildNetworkGroup() {
    m_netGrp = new QGroupBox("Network & Training");
    auto* netF   = new QFormLayout(m_netGrp); netF->setSpacing(5);
    m_hiddenEdit = new QLineEdit("16"); m_hiddenEdit->setToolTip("e.g. 64  or  64,32");
    m_actCombo   = new QComboBox;
    m_actCombo->addItem("ReLU","relu"); m_actCombo->addItem("Sigmoid","sigmoid");
    m_actCombo->addItem("Tanh","tanh"); m_actCombo->addItem("Linear","linear");
    m_valSpin    = new QDoubleSpinBox;
    m_valSpin->setRange(0,0.5); m_valSpin->setValue(0.2); m_valSpin->setSingleStep(0.05);
    netF->addRow("Hidden layers:", m_hiddenEdit);
    netF->addRow("Activation:", m_actCombo);
    netF->addRow("Validation split:", m_valSpin);
    m_equalChk = new QCheckBox("Baseline with equal total budget");
    m_equalChk->setChecked(true);
    m_equalChk->setToolTip("On: the Standard baseline gets as many evaluations as Phases 1 + 2 + 3 (+ BFGS)\n"
                           "together, so the table compares the bounds and not the budget.\n"
                           "Off: the baseline gets only the Phase-2 budget (original protocol).");
    netF->addRow("", m_equalChk);
}

void EvolBoundsPanel::buildConfigArea(QVBoxLayout* root) {
    auto* cfgRow = new QHBoxLayout; cfgRow->setSpacing(10);

    // Phase 1
    auto* g1 = new QGroupBox("Phase 1  —  Bounds discovery");
    m_ph1Grp = g1;
    g1->setStyleSheet("QGroupBox{border:1px solid #E3B341;border-radius:6px;padding-top:14px;}"
                      "QGroupBox::title{color:#E3B341;font-weight:600;}");
    auto* f1 = new QFormLayout(g1); f1->setSpacing(4);
    m_opt1Combo = makeOptCombo(false,false);
    m_nc1Spin   = new QSpinBox; m_nc1Spin->setRange(10,5000); m_nc1Spin->setValue(500);
    m_ng1Spin   = new QSpinBox; m_ng1Spin->setRange(10,5000); m_ng1Spin->setValue(200);
    m_lamSpin   = new QDoubleSpinBox; m_lamSpin->setRange(1.01,20); m_lamSpin->setValue(2.0); m_lamSpin->setSingleStep(0.1);
    m_lamSpin->setToolTip("λ: penalty multiplier (>1)");
    m_aSpin     = new QDoubleSpinBox; m_aSpin->setRange(0.5,20); m_aSpin->setValue(5.0); m_aSpin->setSingleStep(0.5);
    m_aSpin->setToolTip("a: saturation threshold");
    m_fSpin     = new QDoubleSpinBox; m_fSpin->setRange(1.01,10); m_fSpin->setValue(3.0); m_fSpin->setSingleStep(0.1);
    m_fSpin->setToolTip("f: bound factor — lb_i = −f·|g*_i|");
    m_ibSpin    = new QDoubleSpinBox; m_ibSpin->setRange(0.5,50); m_ibSpin->setValue(10.0); m_ibSpin->setSingleStep(0.5);
    m_ibSpin->setToolTip("Initial ±bound for Phase 1 search and baseline training");
    f1->addRow("Optimizer:",m_opt1Combo); f1->addRow("Population:",m_nc1Spin);
    f1->addRow("Iterations:",m_ng1Spin); f1->addRow("Penalty λ:",m_lamSpin);
    f1->addRow("Threshold a:",m_aSpin); f1->addRow("Bound factor f:",m_fSpin);
    f1->addRow("Initial bound ±:",m_ibSpin);
    cfgRow->addWidget(g1);

    // Phase 2
    auto* g2 = new QGroupBox("Phase 2  —  Training in bounds");
    m_ph2Grp = g2;
    g2->setStyleSheet("QGroupBox{border:1px solid #79C0FF;border-radius:6px;padding-top:14px;}"
                      "QGroupBox::title{color:#79C0FF;font-weight:600;}");
    auto* f2 = new QFormLayout(g2); f2->setSpacing(4);
    m_opt2Combo = makeOptCombo(true,true); m_opt2Combo->setCurrentIndex(2);
    m_nc2Spin   = new QSpinBox; m_nc2Spin->setRange(10,5000); m_nc2Spin->setValue(200);
    m_ng2Spin   = new QSpinBox; m_ng2Spin->setRange(10,5000); m_ng2Spin->setValue(200);
    auto* geE = new QWidget; auto* geP = new QWidget;
    auto* geF = new QFormLayout(geP); geF->setContentsMargins(0,0,0,0);
    m_nsGeSpin = new QSpinBox; m_nsGeSpin->setRange(5,500);  m_nsGeSpin->setValue(30);
    m_nlGeSpin = new QSpinBox; m_nlGeSpin->setRange(1,100);  m_nlGeSpin->setValue(10);
    m_plGeSpin = new QDoubleSpinBox; m_plGeSpin->setRange(0,1); m_plGeSpin->setValue(0.05); m_plGeSpin->setSingleStep(0.01);
    geF->addRow("Samples Ns:",m_nsGeSpin); geF->addRow("Local Nl:",m_nlGeSpin); geF->addRow("Rate pl:",m_plGeSpin);
    m_geStack = new QStackedWidget; m_geStack->addWidget(geE); m_geStack->addWidget(geP);
    connect(m_opt2Combo, &QComboBox::currentIndexChanged, this, [this](int idx){
        const bool skip=(m_opt2Combo->itemData(idx).toString()=="none");
        m_nc2Spin->setEnabled(!skip); m_ng2Spin->setEnabled(!skip); m_geStack->setEnabled(!skip);
        onOpt2Changed(idx);
    });
    f2->addRow("Optimizer:",m_opt2Combo); f2->addRow("Population:",m_nc2Spin);
    f2->addRow("Iterations:",m_ng2Spin); f2->addRow("",m_geStack);
    cfgRow->addWidget(g2);

    // Phase 3
    auto* g3 = new QGroupBox("Phase 3  —  Final training");
    m_ph3Grp = g3;
    g3->setStyleSheet("QGroupBox{border:1px solid #56D364;border-radius:6px;padding-top:14px;}"
                      "QGroupBox::title{color:#56D364;font-weight:600;}");
    auto* f3 = new QFormLayout(g3); f3->setSpacing(4);
    m_opt3Combo = makeOptCombo(true,false);
    m_nc3Spin   = new QSpinBox; m_nc3Spin->setRange(10,5000); m_nc3Spin->setValue(500);
    m_ng3Spin   = new QSpinBox; m_ng3Spin->setRange(10,5000); m_ng3Spin->setValue(200);
    m_bfgsChk   = new QCheckBox("BFGS polish"); m_bfgsChk->setChecked(true);
    connect(m_opt3Combo, &QComboBox::currentIndexChanged, this, [this](int idx){
        const bool skip=(m_opt3Combo->itemData(idx).toString()=="none");
        m_nc3Spin->setEnabled(!skip); m_ng3Spin->setEnabled(!skip); m_bfgsChk->setEnabled(!skip);
    });
    f3->addRow("Optimizer:",m_opt3Combo); f3->addRow("Population:",m_nc3Spin);
    f3->addRow("Iterations:",m_ng3Spin); f3->addRow("",m_bfgsChk);
    cfgRow->addWidget(g3);
    root->addLayout(cfgRow);

    // Run / Stop / N-runs row
    auto* bRow = new QHBoxLayout;
    m_runBtn  = new QPushButton("▶  Run Evolutionary Bounds");
    m_stopBtn = new QPushButton("■  Stop"); m_stopBtn->setEnabled(false);
    m_runBtn->setStyleSheet(
        "QPushButton{background:#1D9E75;color:white;border-radius:6px;padding:6px 18px;"
        "font-weight:600;font-size:13px;}QPushButton:disabled{background:#555;color:#999;}");
    m_stopBtn->setStyleSheet(
        "QPushButton{background:#B03030;color:white;border-radius:6px;padding:6px 18px;"
        "font-weight:600;font-size:13px;}QPushButton:disabled{background:#555;color:#999;}");

    m_nRunsSpin = new QSpinBox;
    m_nRunsSpin->setRange(1, 100);
    m_nRunsSpin->setValue(1);
    m_nRunsSpin->setToolTip(
        "Number of independent runs per configuration.\n"
        "When N > 1 the Comparison table shows mean \u00b1 std.\n"
        "Bounds chart shows results from the last run.");
    auto* nRunsLbl = new QLabel("Runs:");
    nRunsLbl->setToolTip(m_nRunsSpin->toolTip());

    bRow->addWidget(m_runBtn);
    bRow->addWidget(m_stopBtn);
    bRow->addSpacing(16);
    bRow->addWidget(nRunsLbl);
    bRow->addWidget(m_nRunsSpin);
    bRow->addStretch();
    root->addLayout(bRow);
    connect(m_runBtn, &QPushButton::clicked, this, &EvolBoundsPanel::onRunClicked);
    connect(m_stopBtn,&QPushButton::clicked, this, &EvolBoundsPanel::onStopClicked);

    m_phaseNameLbl = new QLabel(" ");
    m_phaseNameLbl->setStyleSheet("color:#79C0FF;font-size:12px;font-style:italic;");
    root->addWidget(m_phaseNameLbl);
    m_progressContainer = new QWidget;
    m_progressContainer->setLayout(new QVBoxLayout);
    m_progressContainer->layout()->setContentsMargins(0,0,0,0);
    root->addWidget(m_progressContainer);
    { auto* stRow=new QHBoxLayout;
      m_statusLbl=new QLabel("Add dataset files and click Run.");
      m_statusLbl->setStyleSheet("color:palette(mid);font-size:12px;");
      m_timeLbl=new QLabel; m_timeLbl->hide();
      m_timeLbl->setStyleSheet("QLabel{font-family:monospace;font-size:11px;color:#1D9E75;"
                               "background:transparent;}");
      stRow->addWidget(m_statusLbl,1); stRow->addWidget(m_timeLbl);
      root->addLayout(stRow); }
    m_tickTimer=new QTimer(this); m_tickTimer->setInterval(500);
    connect(m_tickTimer,&QTimer::timeout,this,&EvolBoundsPanel::onTimerTick);
}

// ── Scroll tab helper ─────────────────────────────────────────────────────────
static QPushButton* makeExportBtn(const QString& label) {
    auto* btn = new QPushButton(label);
    btn->setStyleSheet("QPushButton{background:#3D5A80;color:white;border-radius:5px;"
                       "padding:5px 14px;font-size:12px;}");
    return btn;
}
static QWidget* makeScrollTab(QVBoxLayout*& outLayout, QPushButton* exportBtn) {
    auto* scrollContent = new QWidget;
    outLayout = new QVBoxLayout(scrollContent);
    outLayout->setContentsMargins(8,8,8,8); outLayout->setSpacing(16);
    outLayout->addStretch();
    auto* scroll = new QScrollArea;
    scroll->setWidget(scrollContent);
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    auto* tab = new QWidget;
    auto* tl  = new QVBoxLayout(tab); tl->setContentsMargins(4,4,4,4);
    tl->addWidget(scroll);
    auto* er = new QHBoxLayout; er->addWidget(exportBtn); er->addStretch();
    tl->addLayout(er);
    return tab;
}

// ── Tab widget ─────────────────────────────────────────────────────────────────
void EvolBoundsPanel::buildTabWidget(QVBoxLayout* root) {
    m_tabs = new QTabWidget; m_tabs->setMinimumHeight(360);

    // Tab 0: Loss
    {
        auto* expBtn = makeExportBtn("💾  Export all Loss PNG (300 DPI)");
        connect(expBtn, &QPushButton::clicked, this, &EvolBoundsPanel::onExportLossPng);
        m_tabs->addTab(makeScrollTab(m_lossLayout, expBtn), "📈  Loss per iteration");
    }

    // Tab 1: Bounds
    {
        auto* expBtn = makeExportBtn("💾  Export all Bounds PNG (300 DPI)");
        connect(expBtn, &QPushButton::clicked, this, &EvolBoundsPanel::onExportBoundsPng);
        m_tabs->addTab(makeScrollTab(m_boundsLayout, expBtn), "📊  Parameter bounds");
    }

    // Tab 2: Evaluation
    {
        auto* tab = new QWidget; auto* tl = new QVBoxLayout(tab); tl->setContentsMargins(8,8,8,8);
        m_evalTable = new QTableWidget(0,4);
        m_evalTable->setHorizontalHeaderLabels({"Dataset","Task","Metric","Value"});
        m_evalTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
        m_evalTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
        m_evalTable->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
        m_evalTable->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
        m_evalTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
        m_evalTable->setSelectionBehavior(QAbstractItemView::SelectRows);
        tl->addWidget(m_evalTable);
        auto* er = new QHBoxLayout;
        auto* eb = makeExportBtn("💾  Export Evaluation XLSX");
        connect(eb, &QPushButton::clicked, this, &EvolBoundsPanel::onExportEvalXlsx);
        er->addWidget(eb); er->addStretch(); tl->addLayout(er);
        m_tabs->addTab(tab, "📋  Evaluation");
    }

    // Tab 3: Comparison
    {
        auto* tab = new QWidget; auto* tl = new QVBoxLayout(tab); tl->setContentsMargins(8,8,8,8);
        m_compTable = new QTableWidget(0,2);
        m_compTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
        m_compTable->setSelectionBehavior(QAbstractItemView::SelectRows);
        tl->addWidget(m_compTable);
        auto* er = new QHBoxLayout;
        auto* eb = makeExportBtn("💾  Export Comparison XLSX");
        connect(eb, &QPushButton::clicked, this, &EvolBoundsPanel::onExportCompXlsx);
        er->addWidget(eb); er->addStretch(); tl->addLayout(er);
        m_tabs->addTab(tab, "🔬  Comparison");
    }

    root->addWidget(m_tabs);
}

// ── Chart styling ─────────────────────────────────────────────────────────────
void EvolBoundsPanel::styleChart(QChart* c, bool dark) {
    if (dark) {
        c->setBackgroundBrush(QBrush(QColor("#1e1e2a")));
        c->setPlotAreaBackgroundBrush(QBrush(QColor("#272735")));
        c->setTitleBrush(QBrush(QColor("#E4E4EC")));
    } else {
        c->setBackgroundBrush(QBrush(Qt::white));
        c->setPlotAreaBackgroundBrush(QBrush(QColor("#F8F8FB")));
        c->setTitleBrush(QBrush(Qt::black));
        QFont f=c->titleFont(); f.setPointSize(11); f.setBold(true); c->setTitleFont(f);
    }
    c->setBackgroundRoundness(6); c->setPlotAreaBackgroundVisible(true);
    c->setMargins(QMargins(10,10,10,10));
    c->legend()->setVisible(true); c->legend()->setAlignment(Qt::AlignRight);
    c->legend()->setLabelBrush(QBrush(dark ? QColor("#E4E4EC") : Qt::black));
    c->legend()->setBackgroundVisible(!dark);
}

// ── Queue management ──────────────────────────────────────────────────────────
void EvolBoundsPanel::addToQueue(const QStringList& paths) {
    for (const QString& p : paths)
        if (!m_queue.contains(p)) m_queue << p;
    refreshQueueTable();
    m_statusLbl->setText(QString("%1 dataset(s) queued.").arg(m_queue.size()));
}
void EvolBoundsPanel::refreshQueueTable() {
    m_queueTable->setRowCount(m_queue.size());
    for (int i = 0; i < m_queue.size(); ++i) {
        auto* n = new QTableWidgetItem(QString::number(i+1));
        n->setTextAlignment(Qt::AlignCenter); m_queueTable->setItem(i,0,n);
        auto* nm = new QTableWidgetItem(QFileInfo(m_queue[i]).fileName());
        nm->setToolTip(m_queue[i]); m_queueTable->setItem(i,1,nm);
        m_queueTable->setItem(i,2, new QTableWidgetItem("Queued"));
    }
}
void EvolBoundsPanel::onAddFiles() {
    const QStringList p = QFileDialog::getOpenFileNames(
        this,"Add datasets",{},DatasetLoader::fileDialogFilter());
    if (!p.isEmpty()) addToQueue(p);
}
void EvolBoundsPanel::onRemoveSelected() {
    if (m_running) return;
    QSet<int> toRm;
    for (auto* item : m_queueTable->selectedItems()) toRm.insert(item->row());
    QList<int> sorted(toRm.begin(), toRm.end());
    std::sort(sorted.begin(), sorted.end(), std::greater<int>());
    for (int r : sorted) m_queue.removeAt(r);
    refreshQueueTable();
}
void EvolBoundsPanel::onClearQueue() {
    if (m_running) return;
    m_queue.clear(); refreshQueueTable();
    m_statusLbl->setText("Queue cleared.");
}
void EvolBoundsPanel::onOpt2Changed(int idx) {
    m_geStack->setCurrentIndex(m_opt2Combo->itemData(idx).toString()=="ge_grammar"?1:0);
}

// ── Progress ──────────────────────────────────────────────────────────────────
void EvolBoundsPanel::rebuildProgress(int n) {
    for (auto* b : m_bars) b->deleteLater();
    for (auto* l : m_barLbls) l->deleteLater();
    m_bars.clear(); m_barLbls.clear();
    static const QStringList cols={"#888888","#E3B341","#79C0FF","#56D364"};
    static const QStringList lbls={"Baseline:","Phase 1  Bounds:","Phase 2  Training:","Phase 3  Final:"};
    auto* lay = qobject_cast<QVBoxLayout*>(m_progressContainer->layout());
    while (lay->count()) { auto* i=lay->takeAt(0); delete i->widget(); delete i; }
    for (int ph=0; ph<n; ++ph) {
        auto* row = new QHBoxLayout;
        auto* l   = new QLabel(ph<lbls.size()?lbls[ph]:QString("Phase %1:").arg(ph+1));
        l->setFixedWidth(180);
        auto* bar = new QProgressBar; bar->setRange(0,100); bar->setValue(0);
        bar->setStyleSheet(QString("QProgressBar::chunk{background:%1;}").arg(cols[ph%cols.size()]));
        row->addWidget(l); row->addWidget(bar);
        lay->addLayout(row); m_bars.push_back(bar); m_barLbls.push_back(l);
    }
}

// ── Loss chart management ─────────────────────────────────────────────────────
void EvolBoundsPanel::clearLossCharts() {
    m_curLossSeries.clear();
    m_curLossAxisX = nullptr;
    m_curLossAxisY = nullptr;
    m_curLossChart = nullptr;
    m_lossCharts.clear();
    m_lossViews.clear();
    while (m_lossLayout->count() > 1) {
        auto* item = m_lossLayout->takeAt(0);
        if (item->widget()) item->widget()->deleteLater();
        delete item;
    }
}

void EvolBoundsPanel::startNewLossChart(const QString& dsName, int nPhases) {
    auto* chart = new QChart;
    chart->setTitle(QString("Loss per iteration  —  %1").arg(dsName));
    styleChart(chart, true);

    auto* axX = new QValueAxis; axX->setTitleText("Iteration");
    auto* axY = new QValueAxis; axY->setTitleText("Loss");
    chart->addAxis(axX, Qt::AlignBottom);
    chart->addAxis(axY, Qt::AlignLeft);
    axX->setRange(0,10); axY->setRange(0,1);

    static const QList<QColor> cols={QColor("#888888"),QColor("#E3B341"),QColor("#79C0FF"),QColor("#56D364")};
    static const QStringList   names={"Baseline","Phase 1  Bounds","Phase 2  Training","Phase 3  Final"};
    m_curLossSeries.clear();
    for (int ph=0; ph<nPhases; ++ph) {
        auto* s = new QLineSeries;
        s->setName(ph<names.size()?names[ph]:QString("Phase %1").arg(ph+1));
        s->setColor(cols[ph%cols.size()]);
        chart->addSeries(s);
        s->attachAxis(axX); s->attachAxis(axY);
        m_curLossSeries.append(s);
    }

    auto* view = new QChartView(chart);
    view->setRenderHint(QPainter::Antialiasing);
    view->setFixedHeight(280);
    const int pos = std::max(0, m_lossLayout->count()-1);
    m_lossLayout->insertWidget(pos, view);
    m_lossCharts.append(chart);
    m_lossViews.append(view);
    m_curLossChart  = chart;
    m_curLossAxisX  = axX;
    m_curLossAxisY  = axY;
}

// ── makeBoundsChart ───────────────────────────────────────────────────────────
QChart* EvolBoundsPanel::makeBoundsChart(const QString& dsName,
                                          const QVector<double>& lb,
                                          const QVector<double>& rb,
                                          double initBound) {
    auto* chart = new QChart;
    chart->setTitle(dsName);
    styleChart(chart, false);

    const int N = lb.size();
    auto* lbS = new QLineSeries; lbS->setName("lb");
    { QPen p(QColor("#2CA58D")); p.setWidthF(1.2); lbS->setPen(p); }
    auto* rbS = new QLineSeries; rbS->setName("rb");
    { QPen p(QColor("#E07B39")); p.setWidthF(1.2); rbS->setPen(p); }
    double yMin=1e9, yMax=-1e9;
    for (int i=0; i<N; ++i) {
        lbS->append(i,lb[i]); rbS->append(i,rb[i]);
        yMin=std::min({yMin,lb[i],rb[i]}); yMax=std::max({yMax,lb[i],rb[i]});
    }
    auto* rbA = new QLineSeries; auto* lbA = new QLineSeries;
    for (int i=0; i<N; ++i) { rbA->append(i,rb[i]); lbA->append(i,lb[i]); }
    auto* area = new QAreaSeries(rbA,lbA); area->setName("bounds");
    area->setColor(QColor(100,140,210,55)); area->setBorderColor(Qt::transparent);
    auto* rP = new QLineSeries; auto* rN = new QLineSeries;
    { QPen p(QColor("#5B6FD6")); p.setWidthF(1.5); rP->setPen(p); rN->setPen(p); }
    rP->setName(QString("+ %1").arg(initBound)); rN->setName(QString("- %1").arg(initBound));
    rP->append(0,+initBound); rP->append(N-1,+initBound);
    rN->append(0,-initBound); rN->append(N-1,-initBound);
    yMin=std::min(yMin,-initBound*1.1); yMax=std::max(yMax,+initBound*1.1);
    chart->addSeries(area); chart->addSeries(lbS); chart->addSeries(rbS);
    chart->addSeries(rP); chart->addSeries(rN);
    auto* axX = new QValueAxis; axX->setTitleText("Parameter index");
    axX->setRange(0,N-1); axX->setLabelFormat("%d");
    auto* axY = new QValueAxis; axY->setTitleText("Bound value");
    const double pad=(yMax-yMin)*0.08; axY->setRange(yMin-pad,yMax+pad);
    chart->addAxis(axX,Qt::AlignBottom); chart->addAxis(axY,Qt::AlignLeft);
    for (auto* s : {static_cast<QAbstractSeries*>(area),
                    static_cast<QAbstractSeries*>(lbS),
                    static_cast<QAbstractSeries*>(rbS),
                    static_cast<QAbstractSeries*>(rP),
                    static_cast<QAbstractSeries*>(rN)})
        { s->attachAxis(axX); s->attachAxis(axY); }
    return chart;
}

// ── PNG export ────────────────────────────────────────────────────────────────
bool EvolBoundsPanel::exportStackedPng(const QList<QChart*>& charts,
                                        QWidget* parent,
                                        const QString& defaultName) {
    if (charts.isEmpty()) {
        QMessageBox::information(parent,"Export","No charts to export yet.");
        return false;
    }
    const QString path = QFileDialog::getSaveFileName(
        parent,"Export chart",defaultName,"PNG images (*.png)");
    if (path.isEmpty()) return false;
    const int W=2400, perH=800, H=perH*charts.size();
    const int dpm=qRound(300.0/0.0254);
    QImage img(W,H,QImage::Format_RGB32);
    img.setDotsPerMeterX(dpm); img.setDotsPerMeterY(dpm);
    // Dark fill to match the dark-themed charts — white background made the
    // light-colored axis text and gridlines unreadable in the exported PNG
    // (and was inconsistent with PreTrainNNCPanel's export).
    img.fill(QColor("#1e1e2a"));
    QPainter painter(&img);
    painter.setRenderHint(QPainter::Antialiasing);
    for (int i=0; i<charts.size(); ++i)
        if (charts[i]->scene())
            charts[i]->scene()->render(&painter, QRectF(0, i*perH, W, perH));
    painter.end();
    if (img.save(path,"PNG",100)) {
        QMessageBox::information(parent,"Export OK",
            QString("Saved %1 dataset(s) stacked, 300 DPI (%2×%3 px):\n%4")
                .arg(charts.size()).arg(W).arg(H).arg(path));
        return true;
    }
    QMessageBox::critical(parent,"Export Error","Failed to save PNG.");
    return false;
}

// ── refreshInfoLabel: always shows run / dataset / phase on one line ──────────
void EvolBoundsPanel::refreshInfoLabel(const QString& phaseInfo) {
    QString s;
    if (m_totalRuns > 1)
        s += QString("Run %1/%2   ").arg(m_currentRun+1).arg(m_totalRuns);
    if (m_totalDs > 0)
        s += QString("Dataset %1/%2: %3   ").arg(m_currentDsIdx+1).arg(m_totalDs).arg(m_currentDsName);
    s += phaseInfo;
    m_statusLbl->setText(s);
}

// ── buildConfig / launchWorker ────────────────────────────────────────────────
EvolBoundsConfig EvolBoundsPanel::buildConfig() const {
    EvolBoundsConfig cfg;
    cfg.datasetPaths  = m_queue;
    cfg.hiddenLayout  = m_hiddenEdit->text().trimmed();
    cfg.activation    = m_actCombo->currentData().toString();
    cfg.valSplit      = m_valSpin->value();
    cfg.opt1=m_opt1Combo->currentData().toString(); cfg.Nc1=m_nc1Spin->value(); cfg.Ng1=m_ng1Spin->value();
    cfg.lambda=m_lamSpin->value(); cfg.a=m_aSpin->value(); cfg.f=m_fSpin->value();
    cfg.initialBound=m_ibSpin->value();
    cfg.opt2=m_opt2Combo->currentData().toString(); cfg.Nc2=m_nc2Spin->value(); cfg.Ng2=m_ng2Spin->value();
    cfg.Ns_ge=m_nsGeSpin->value(); cfg.Nl_ge=m_nlGeSpin->value(); cfg.pl_ge=m_plGeSpin->value();
    cfg.opt3=m_opt3Combo->currentData().toString(); cfg.Nc3=m_nc3Spin->value(); cfg.Ng3=m_ng3Spin->value();
    cfg.bfgsEnd=m_bfgsChk->isChecked();
    cfg.equalBudgetBaseline=m_equalChk&&m_equalChk->isChecked();
    return cfg;
}

void EvolBoundsPanel::launchWorker(const EvolBoundsConfig& cfg) {
    auto* worker = new EvolBoundsWorker;
    worker->setConfig(cfg);
    auto* thread = new QThread(this);
    worker->moveToThread(thread);
    m_thread = thread;
    m_worker = worker;

    connect(thread,&QThread::started,                  worker,&EvolBoundsWorker::run);
    connect(worker,&EvolBoundsWorker::datasetStarted,   this,&EvolBoundsPanel::onDatasetStarted);
    connect(worker,&EvolBoundsWorker::phaseStarted,     this,&EvolBoundsPanel::onPhaseStarted);
    connect(worker,&EvolBoundsWorker::iterationDone,    this,&EvolBoundsPanel::onIterationDone);
    connect(worker,&EvolBoundsWorker::phaseFinished,    this,&EvolBoundsPanel::onPhaseFinished);
    connect(worker,&EvolBoundsWorker::datasetBoundsReady,this,&EvolBoundsPanel::onDatasetBoundsReady);
    connect(worker,&EvolBoundsWorker::datasetFinished,  this,&EvolBoundsPanel::onDatasetFinished);
    connect(worker,&EvolBoundsWorker::allDone,          this,&EvolBoundsPanel::onAllDone);
    connect(worker,&EvolBoundsWorker::errorOccurred,    this,&EvolBoundsPanel::onError);
    connect(m_stopBtn,&QPushButton::clicked,            worker,&EvolBoundsWorker::requestStop);
    connect(worker,&EvolBoundsWorker::allDone,          thread,&QThread::quit);
    connect(worker,&EvolBoundsWorker::errorOccurred,    thread,&QThread::quit);
    connect(thread,&QThread::finished,                  worker,&QObject::deleteLater);
    connect(thread,&QThread::finished,                  thread,&QObject::deleteLater);

    thread->start();
}

// ── Run ───────────────────────────────────────────────────────────────────────
void EvolBoundsPanel::onRunClicked() {
    if (m_queue.isEmpty()) {
        QMessageBox::information(this,"Evol. Bounds","Add at least one dataset first.");
        return;
    }

    // Stop any previous run
    if (m_worker) m_worker->requestStop();
    if (m_thread && m_thread->isRunning()) {
        m_thread->quit();
        m_thread->wait(4000);
    }
    m_worker = nullptr;
    m_thread = nullptr;

    const QString o2 = m_opt2Combo->currentData().toString();
    const QString o3 = m_opt3Combo->currentData().toString();
    m_activePhases = 2 + (o2!="none"?1:0) + (o3!="none"?1:0);

    rebuildProgress(m_activePhases);
    for (auto* b : m_bars) b->setValue(0);

    // Clear previous results
    clearLossCharts();
    m_boundsCharts.clear(); m_boundsViews.clear();
    while (m_boundsLayout->count() > 1) {
        auto* item = m_boundsLayout->takeAt(0);
        if (item->widget()) item->widget()->deleteLater();
        delete item;
    }
    m_evalTable->setRowCount(0);
    m_compTable->setRowCount(0);
    m_compTable->setColumnCount(0);

    for (int i=0; i<m_queue.size(); ++i)
        m_queueTable->setItem(i,2,new QTableWidgetItem("Queued"));

    // Phase iteration counts for x-offset
    m_phaseIters.assign(m_activePhases, 0);
    m_xOff.assign(m_activePhases+1, 0);
    {
        const EvolBoundsConfig tmp = buildConfig();
        int pi=0;
        if(pi<m_activePhases) m_phaseIters[pi++]=tmp.Ng2;
        if(pi<m_activePhases) m_phaseIters[pi++]=tmp.Ng1;
        if(o2!="none"&&pi<m_activePhases) m_phaseIters[pi++]=tmp.Ng2;
        if(o3!="none"&&pi<m_activePhases) m_phaseIters[pi++]=tmp.Ng3;
    }

    // N-runs init
    m_accumResults.clear();
    m_totalRuns     = m_nRunsSpin->value();
    m_currentRun    = 0;
    m_stopRequested = false;
    m_lastNet.reset();
    m_lastDs.reset();
    m_lastConfig = buildConfig();

    setRunning(true);
    launchWorker(m_lastConfig);
}

void EvolBoundsPanel::onStopClicked() {
    m_stopRequested = true;
    if (m_worker) m_worker->requestStop();
}

// ── Worker signal handlers ────────────────────────────────────────────────────
void EvolBoundsPanel::onDatasetStarted(int dsIdx, int totalDs, QString dsName) {
    m_currentDsName = dsName;
    m_currentDsIdx  = dsIdx;
    m_totalDs       = totalDs;
    for (auto* b : m_bars) b->setValue(0);
    std::fill(m_xOff.begin(), m_xOff.end(), 0);
    startNewLossChart(dsName, m_activePhases);
    if (dsIdx < m_queueTable->rowCount())
        m_queueTable->setItem(dsIdx, 2, new QTableWidgetItem("Running…"));
    refreshInfoLabel("—");
    m_tabs->setCurrentIndex(0);
}

void EvolBoundsPanel::onPhaseStarted(int phase, int total, QString name) {
    m_phaseNameLbl->setText(name);
    refreshInfoLabel(QString("%1  (%2 iter)").arg(name).arg(total));
}

void EvolBoundsPanel::onIterationDone(int phase, int iter, int total, double loss, double) {
    const int idx = phase - 1;
    if (idx < (int)m_bars.size())
        m_bars[idx]->setValue(static_cast<int>(100.0*iter/std::max(1,total)));
    if (idx < m_curLossSeries.size() && m_curLossAxisX && m_curLossAxisY) {
        const int xb = (idx < (int)m_xOff.size()) ? m_xOff[idx] : 0;
        m_curLossSeries[idx]->append(xb + iter, loss);
        if (iter % 10 == 0) {
            double yMin=1e9, yMax=-1e9;
            for (auto* s : m_curLossSeries)
                for (const auto& pt : s->points()) {
                    yMin=std::min(yMin,pt.y()); yMax=std::max(yMax,pt.y());
                }
            if (yMax > yMin) m_curLossAxisY->setRange(yMin*0.9, yMax*1.1);
            m_curLossAxisX->setRange(0, std::max(xb+total, 10));
        }
    }
}

void EvolBoundsPanel::onPhaseFinished(int phase, double loss) {
    const int idx = phase - 1;
    if (idx < (int)m_bars.size()) m_bars[idx]->setValue(100);
    const int iters = (idx < (int)m_phaseIters.size()) ? m_phaseIters[idx] : 200;
    if (idx+1 < (int)m_xOff.size()) m_xOff[idx+1] = m_xOff[idx] + iters;
    refreshInfoLabel(QString("Phase %1 done  (loss=%2)").arg(phase).arg(loss,0,'g',5));
}

void EvolBoundsPanel::onDatasetBoundsReady(QString dsName,
                                            QVector<double> lb,
                                            QVector<double> rb,
                                            double initBound) {
    // Bounds chart: keep only the LAST run's charts
    // Clear old charts for this dataset (if we're re-running)
    // For simplicity: clear all bounds charts on each new run (done in onAllDone)
    // Here we just add the chart for the current run
    QChart* chart = makeBoundsChart(dsName, lb, rb, initBound);
    auto*   view  = new QChartView(chart);
    view->setRenderHint(QPainter::Antialiasing);
    view->setFixedHeight(280);
    const int pos = std::max(0, m_boundsLayout->count()-1);
    m_boundsLayout->insertWidget(pos, view);
    m_boundsCharts.append(chart);
    m_boundsViews.append(view);
    m_tabs->setCurrentIndex(1);
    m_statusLbl->setText(QString("[%1]  Phase 1 done: %2 bounds computed.")
                         .arg(dsName).arg(lb.size()));
}

void EvolBoundsPanel::onDatasetFinished(QString dsName, QStringList configs,
                                         QVector<double> trainLoss,
                                         QVector<double> trainErr,
                                         QVector<double> valErr,
                                         QVector<double> testErr,
                                         bool isCls) {
    ++m_dsDone;
    for (int i=0; i<m_queue.size(); ++i)
        if (QFileInfo(m_queue[i]).baseName() == dsName)
            m_queueTable->setItem(i,2,new QTableWidgetItem(
                m_currentRun < m_totalRuns-1
                    ? QString("Run %1/%2 done").arg(m_currentRun+1).arg(m_totalRuns)
                    : "Done ✓"));

    // Accumulate
    auto& acc = m_accumResults[dsName];
    if (acc.trainErr.isEmpty()) {
        acc.configs = configs;
        acc.isCls   = isCls;
        acc.trainErr.resize(configs.size());
        acc.valErr.resize(configs.size());
        acc.trainLoss.resize(configs.size());
        acc.testErr.resize(configs.size());
    }
    // "Standard + Phase 3 (Phase 1+2 rejected)": the final phase of a run that carried the Standard
    // weights forward. The column keeps the label "Phase 1+2+3" and the runs are counted, so the
    // table can say how often it happened (the label differs from run to run and between datasets).
    if(acc.fromStd.size()<configs.size()) acc.fromStd.resize(configs.size());
    for(int c=0;c<configs.size();++c)
        if(configs[c].startsWith("Standard + Phase 3")){
            ++acc.fromStd[c];
            if(c<acc.configs.size()) acc.configs[c]="Phase 1+2+3";
        }
    for (int c=0; c<configs.size(); ++c) {
        acc.testErr[c].append  (c<testErr.size()   ? testErr[c]   : -1.0);
        acc.trainErr[c].append (c<trainErr.size()  ? trainErr[c]  : 0.0);
        acc.valErr[c].append   (c<valErr.size()    ? valErr[c]    : -1.0);
        acc.trainLoss[c].append(c<trainLoss.size() ? trainLoss[c] : 0.0);
    }
}

void EvolBoundsPanel::onAllDone(std::shared_ptr<NeuralNetwork> net,
                                 std::shared_ptr<Dataset>       ds) {
    m_lastNet = net;
    m_lastDs  = ds;

    if (!m_stopRequested && m_currentRun < m_totalRuns - 1) {
        ++m_currentRun;

        // Clear loss charts for the new run
        clearLossCharts();
        // Clear bounds charts for the new run (last run's bounds will be shown)
        m_boundsCharts.clear();
        m_boundsViews.clear();
        while (m_boundsLayout->count() > 1) {
            auto* item = m_boundsLayout->takeAt(0);
            if (item->widget()) item->widget()->deleteLater();
            delete item;
        }

        for (int i=0; i<m_queue.size(); ++i)
            m_queueTable->setItem(i,2,new QTableWidgetItem("Queued"));
        for (auto* b : m_bars) b->setValue(0);
        std::fill(m_xOff.begin(), m_xOff.end(), 0);

        m_statusLbl->setText(QString("Run %1/%2 — starting next run...")
            .arg(m_currentRun+1).arg(m_totalRuns));
        m_phaseNameLbl->setText(" ");

        launchWorker(m_lastConfig);
    } else {
        finalizeAllRuns();
    }
}

// ── finalizeAllRuns ───────────────────────────────────────────────────────────
void EvolBoundsPanel::finalizeAllRuns() {
    m_evalTable->clearContents(); m_evalTable->setRowCount(0);
    m_compTable->clearContents(); m_compTable->setRowCount(0);
    m_compTable->setColumnCount(0);

    const bool hasVal = m_valSpin->value() > 0;
    const int  N      = m_currentRun + 1;

    for (auto it = m_accumResults.begin(); it != m_accumResults.end(); ++it) {
        const QString&  dsName = it.key();
        const DsAccum&  acc    = it.value();
        const QString errLbl = acc.isCls ? "Error %" : "MSE";
        // regression errors with 4 significant digits (two decimals turned an MSE of 0.00035 into "0.00")
        auto fmtE = [&](const QVector<double>& v) { return acc.isCls ? fmtMS(v) : fmtMS(v, 'g', 4); };
        const QString valLbl = acc.isCls ? "Val Error %" : "Val MSE";
        const int lastIdx = acc.configs.size() - 1;

        // Evaluation: last configuration's stats
        auto addE=[&](const QString& metric, const QString& value){
            const int r=m_evalTable->rowCount(); m_evalTable->setRowCount(r+1);
            m_evalTable->setItem(r,0,new QTableWidgetItem(dsName));
            m_evalTable->setItem(r,1,new QTableWidgetItem(acc.isCls?"Classification":"Regression"));
            m_evalTable->setItem(r,2,new QTableWidgetItem(metric));
            m_evalTable->setItem(r,3,new QTableWidgetItem(value));
        };
        if (lastIdx >= 0) {
            addE(errLbl, fmtE(acc.trainErr[lastIdx]));
            if (hasVal) {
                QVector<double> vv;
                for (double x : acc.valErr[lastIdx]) if (x >= 0) vv.append(x);
                if (!vv.isEmpty()) addE(valLbl, fmtE(vv));
            }
            {
                QVector<double> tv;
                for (double x : acc.testErr[lastIdx]) if (x >= 0) tv.append(x);
                if (!tv.isEmpty()) addE(acc.isCls ? "Test Error %" : "Test MSE", fmtE(tv));
            }
            addE("Final Loss", fmtMS(acc.trainLoss[lastIdx],'g',4));
            if (N > 1) addE("Runs", QString::number(N));
        }

        // Comparison: all configurations with mean ± std
        if (m_compTable->columnCount() < 3) {
            const int nCols = 2 + acc.configs.size();
            m_compTable->setColumnCount(nCols);
            QStringList hdr; hdr << "Dataset" << "Metric";
            for (const QString& c : acc.configs) hdr << c;
            m_compTable->setHorizontalHeaderLabels(hdr);
            m_compTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
            m_compTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
            for (int c=2; c<nCols; ++c)
                m_compTable->horizontalHeader()->setSectionResizeMode(c, QHeaderView::Stretch);
        }
        auto addC=[&](const QString& metric, const std::function<QString(int)>& fn){
            const int r=m_compTable->rowCount(); m_compTable->setRowCount(r+1);
            m_compTable->setItem(r,0,new QTableWidgetItem(dsName));
            m_compTable->setItem(r,1,new QTableWidgetItem(metric));
            for (int c=0; c<acc.configs.size(); ++c)
                m_compTable->setItem(r,c+2,new QTableWidgetItem(fn(c)));
        };
        addC(errLbl, [&](int c){ return fmtE(acc.trainErr[c]); });
        if (hasVal) addC(valLbl, [&](int c){
            QVector<double> vv;
            for (double x : acc.valErr[c]) if (x >= 0) vv.append(x);
            return vv.isEmpty() ? QStringLiteral("-") : fmtE(vv);
        });
        {
            bool anyTest = false;
            for (const auto& v : acc.testErr) for (double x : v) anyTest = anyTest || x >= 0;
            if (anyTest) addC(acc.isCls ? "Test Error %" : "Test MSE", [&](int c){
                QVector<double> tv;
                for (double x : acc.testErr[c]) if (x >= 0) tv.append(x);
                return tv.isEmpty() ? QStringLiteral("-") : fmtE(tv);
            });
        }
        addC("Final Loss", [&](int c){ return fmtMS(acc.trainLoss[c],'g',4); });
        {
            int anyStd=0; for(int v:acc.fromStd) anyStd+=v;
            if(anyStd>0) addC("Phase 3 started from", [&](int c){
                if(c>=acc.fromStd.size()||acc.configs.value(c)!="Phase 1+2+3") return QStringLiteral("-");
                const int k=acc.fromStd[c], n=acc.trainErr[c].size();
                return k==0 ? QStringLiteral("Phase 1+2") : k==n ? QStringLiteral("Standard (Phase 1+2 rejected)")
                            : QString("Standard in %1 of %2 runs").arg(k).arg(n);
            });
        }
        if (N > 1) addC("Runs", [&](int){ return QString::number(N); });
    }

    setRunning(false);
    m_phaseNameLbl->setText(" ");
    const QString nStr = N > 1 ? QString(" (%1 runs, mean \u00b1 std)").arg(N) : "";
    m_statusLbl->setText(QString("✓  All %1 dataset(s) complete%2.  All tabs updated.")
        .arg(m_queue.size()).arg(nStr));
    m_tabs->setCurrentIndex(3);
    if (m_lastNet) emit trainingCompleted(m_lastNet, m_lastDs);
}

void EvolBoundsPanel::onError(QString msg) {
    m_stopRequested = true;
    finalizeAllRuns();
    QMessageBox::critical(this,"Evol. Bounds Error",msg);
    m_statusLbl->setText(QString("Error (run %1/%2): %3")
        .arg(m_currentRun+1).arg(m_totalRuns).arg(msg));
}

void EvolBoundsPanel::onTimerTick() {
    const int sec=static_cast<int>(m_clock.elapsed()/1000);
    const int total=m_totalRuns * m_queue.size();
    QString eta;
    if (m_dsDone>0 && m_dsDone<total) {
        const int etaSec=static_cast<int>(
            (double)m_clock.elapsed()/m_dsDone*(total-m_dsDone)/1000.0);
        eta=QString("  |  ETA %1m%2s").arg(etaSec/60).arg(etaSec%60,2,10,QChar('0'));
    }
    m_timeLbl->setText(QString("Elapsed  %1:%2%3")
        .arg(sec/60).arg(sec%60,2,10,QChar('0')).arg(eta));
}

void EvolBoundsPanel::setRunning(bool r) {
    m_running=r; m_runBtn->setEnabled(!r); m_stopBtn->setEnabled(r);
    m_nRunsSpin->setEnabled(!r);
    m_addBtn->setEnabled(!r); m_removeBtn->setEnabled(!r); m_clearBtn->setEnabled(!r);
    // Lock configuration groups while a run is in progress — changing them
    // mid-run would not affect the in-flight worker.
    if(m_netGrp) m_netGrp->setEnabled(!r);
    if(m_ph1Grp) m_ph1Grp->setEnabled(!r);
    if(m_ph2Grp) m_ph2Grp->setEnabled(!r);
    if(m_ph3Grp) m_ph3Grp->setEnabled(!r);
    if(r){
        m_dsDone=0; m_clock.start(); m_tickTimer->start();
        m_timeLbl->setText("Elapsed  0:00"); m_timeLbl->show();
    } else {
        m_tickTimer->stop(); onTimerTick();
    }
    emit runningStateChanged(r);
}

// ── Exports ───────────────────────────────────────────────────────────────────
void EvolBoundsPanel::onExportLossPng() {
    exportStackedPng(m_lossCharts, this, "loss_all.png");
}
void EvolBoundsPanel::onExportBoundsPng() {
    exportStackedPng(m_boundsCharts, this, "bounds_all.png");
}

static void tableToXlsx(QTableWidget* tbl, const QString& path, const QString& sheetName) {
    XlsxWriter xls; auto* sh = xls.addSheet(sheetName);
    QStringList hdr;
    for (int c=0; c<tbl->columnCount(); ++c)
        hdr << (tbl->horizontalHeaderItem(c)?tbl->horizontalHeaderItem(c)->text():"");
    sh->setHeader(hdr);
    for (int r=0; r<tbl->rowCount(); ++r) {
        std::vector<XlsxCell> row;
        for (int c=0; c<tbl->columnCount(); ++c) {
            const QString txt = tbl->item(r,c)?tbl->item(r,c)->text():"";
            bool ok; double val=txt.toDouble(&ok);
            row.push_back(ok?XlsxCell(val):XlsxCell(txt));
        }
        sh->addRow(row);
    }
    xls.save(path);
}
void EvolBoundsPanel::onExportEvalXlsx() {
    const QString p = QFileDialog::getSaveFileName(
        this,"Export Evaluation","evaluation.xlsx",XlsxWriter::fileFilter());
    if (p.isEmpty()) return;
    try { tableToXlsx(m_evalTable,p,"Evaluation");
          m_statusLbl->setText("Exported: "+QFileInfo(p).fileName()); }
    catch(const std::exception&ex){ QMessageBox::critical(this,"Export Error",ex.what()); }
}
void EvolBoundsPanel::onExportCompXlsx() {
    const QString p = QFileDialog::getSaveFileName(
        this,"Export Comparison","comparison.xlsx",XlsxWriter::fileFilter());
    if (p.isEmpty()) return;
    try { tableToXlsx(m_compTable,p,"Comparison");
          m_statusLbl->setText("Exported: "+QFileInfo(p).fileName()); }
    catch(const std::exception&ex){ QMessageBox::critical(this,"Export Error",ex.what()); }
}

} // namespace NeuralStudio
