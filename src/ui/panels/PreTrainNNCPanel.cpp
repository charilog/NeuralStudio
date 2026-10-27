#include "PreTrainNNCPanel.h"
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
#include <QScrollArea>
#include <QTimer>
#include <QChartView>
#include <QLineSeries>
#include <QValueAxis>
#include <QBarSeries>
#include <QBarSet>
#include <QBarCategoryAxis>
#include <QLegend>
#include <QImage>
#include <algorithm>
#include <cmath>
#include <numeric>

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

// Format a vector of run values as "mean" (N=1) or "mean ± std" (N>1)
static QString fmtMS(const QVector<double>& v, char fmt = 'f', int prec = 2) {
    if (v.isEmpty()) return QStringLiteral("-");
    auto [mean, sd] = runStats(v);
    if (v.size() == 1)
        return QString::number(mean, fmt, prec);
    return QString("%1 \u00b1 %2")
        .arg(QString::number(mean, fmt, prec))
        .arg(QString::number(sd,   fmt, prec));
}

// ── Optimizer list ────────────────────────────────────────────────────────────
static const QStringList kOptLabels = {
    "GA  (Genetic Algorithm)","DE  (Differential Evolution)","PSO  (Particle Swarm)",
    "CMA-ES","JSO","LM-CMA-ES","ACOR","CLPSO","ML-SHADE-RL",
    "L-BFGS","BFGS","Simulated Annealing","Nelder-Mead"
};
static const QStringList kOptKeys = {
    "ga","de","pso","cmaes","jso","lmcmaes","acor","clpso","mlshaderl","lbfgs","bfgs","sa","neldermead"
};

// ── PhaseBox ──────────────────────────────────────────────────────────────────
void PhaseBox::fillOptimizerCombo(QComboBox* c, bool allowNone) {
    if (allowNone) { c->addItem("— None (skip phase) —","none"); c->insertSeparator(1); }
    for (int i=0;i<kOptLabels.size();++i) c->addItem(kOptLabels[i],kOptKeys[i]);
}

PhaseBox::PhaseBox(int num, bool allowNone, QWidget* parent)
    : QGroupBox(QString("Phase %1").arg(num), parent)
{
    auto* f = new QFormLayout(this); f->setSpacing(6);
    m_optCombo = new QComboBox; fillOptimizerCombo(m_optCombo, allowNone);
    m_popSpin  = new QSpinBox; m_popSpin->setRange(10,5000); m_popSpin->setValue(200);
    m_iterSpin = new QSpinBox; m_iterSpin->setRange(10,5000); m_iterSpin->setValue(200);
    m_wbSpin   = new QDoubleSpinBox;
    m_wbSpin->setRange(0.5,50); m_wbSpin->setValue(10.0); m_wbSpin->setSingleStep(0.5);
    f->addRow("Optimizer:",   m_optCombo);
    f->addRow("Population:",  m_popSpin);
    f->addRow("Iterations:",  m_iterSpin);
    f->addRow("Weight bound:",m_wbSpin);
    if (allowNone) {
        connect(m_optCombo,&QComboBox::currentIndexChanged,this,[this](int idx){
            const bool skip=(m_optCombo->itemData(idx).toString()=="none");
            m_popSpin->setEnabled(!skip); m_iterSpin->setEnabled(!skip); m_wbSpin->setEnabled(!skip);
        });
    }
}

PhaseConfig PhaseBox::config() const {
    PhaseConfig pc;
    pc.optimizer  = m_optCombo->currentData().toString();
    pc.population = m_popSpin->value();
    pc.iterations = m_iterSpin->value();
    pc.wBound     = m_wbSpin->value();
    return pc;
}

// ── Constructor ───────────────────────────────────────────────────────────────
PreTrainNNCPanel::PreTrainNNCPanel(QWidget* parent) : QWidget(parent) {
    setAcceptDrops(true);
    buildUi();
}

void PreTrainNNCPanel::dragEnterEvent(QDragEnterEvent* e) {
    if (e->mimeData()->hasUrls()) e->acceptProposedAction();
}
void PreTrainNNCPanel::dropEvent(QDropEvent* e) {
    QStringList paths;
    for (const auto& url : e->mimeData()->urls()) {
        const QString p = url.toLocalFile();
        if (!p.isEmpty()) paths << p;
    }
    addToQueue(paths);
}

// ── buildUi ───────────────────────────────────────────────────────────────────
void PreTrainNNCPanel::buildUi() {
    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(20,14,20,14); root->setSpacing(8);

    auto* title = new QLabel("Pre-Train + NNC");
    { QFont f=title->font(); f.setPointSize(14); f.setBold(true); title->setFont(f); }
    root->addWidget(title);
    auto* sub = new QLabel(
        "<b>Phase 1</b>: any optimizer pre-trains a small network (Iw nodes).  "
        "<b>Phase 2</b>: main network fine-tuned, warm-started from Phase 1.  "
        "<b>Phase 3</b>: optional further fine-tune.  "
        "A <b>baseline</b> (Phase 2 without pre-training) is also run for comparison.  "
        "Multiple datasets and <b>N independent runs</b> (mean \u00b1 std) supported.  "
        "<i>Symmetry 2025</i>");
    sub->setWordWrap(true); sub->setStyleSheet("color:palette(mid);font-size:12px;");
    root->addWidget(sub);

    buildNetworkGroup();
    buildUi_queueAndNetwork(root);
    buildConfigArea(root);
    buildTabWidget(root);
}

// ── Dataset queue + Network & Training, side by side ────────────────────────
void PreTrainNNCPanel::buildUi_queueAndNetwork(QVBoxLayout* root) {
    auto* row = new QHBoxLayout; row->setSpacing(10);
    QGroupBox* queueBox = buildQueueGroup();
    row->addWidget(queueBox, 2);      // dataset queue: a bit wider
    row->addWidget(m_netGrp, 1);      // Network & Training: to its right
    root->addLayout(row);
}

// ── Dataset queue ─────────────────────────────────────────────────────────────
QGroupBox* PreTrainNNCPanel::buildQueueGroup() {
    auto* grp = new QGroupBox("Dataset Queue  (drag && drop files here)");
    auto* lay = new QVBoxLayout(grp); lay->setSpacing(6);
    auto* btnRow = new QHBoxLayout;
    auto mkBtn = [&](const QString& t, const QString& bg="")->QPushButton*{
        auto* b=new QPushButton(t); b->setMinimumHeight(28);
        if(!bg.isEmpty()) b->setStyleSheet(QString("QPushButton{background:%1;color:white;"
            "border-radius:4px;font-weight:600;padding:0 12px;}"
            "QPushButton:disabled{background:palette(midlight);}").arg(bg));
        return b;
    };
    m_addBtn    = mkBtn("Add Files…");
    m_removeBtn = mkBtn("Remove Selected");
    m_clearBtn  = mkBtn("Clear Queue","#8B3030");
    btnRow->addWidget(m_addBtn); btnRow->addWidget(m_removeBtn);
    btnRow->addWidget(m_clearBtn); btnRow->addStretch();
    lay->addLayout(btnRow);
    connect(m_addBtn,   &QPushButton::clicked,this,&PreTrainNNCPanel::onAddFiles);
    connect(m_removeBtn,&QPushButton::clicked,this,&PreTrainNNCPanel::onRemoveSelected);
    connect(m_clearBtn, &QPushButton::clicked,this,&PreTrainNNCPanel::onClearQueue);

    m_queueTable = new QTableWidget(0,3);
    m_queueTable->setHorizontalHeaderLabels({"#","Dataset","Status"});
    m_queueTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_queueTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_queueTable->horizontalHeader()->setSectionResizeMode(0,QHeaderView::ResizeToContents);
    m_queueTable->horizontalHeader()->setSectionResizeMode(1,QHeaderView::Stretch);
    m_queueTable->horizontalHeader()->setSectionResizeMode(2,QHeaderView::ResizeToContents);
    m_queueTable->verticalHeader()->hide();
    m_queueTable->verticalHeader()->setDefaultSectionSize(22);
    m_queueTable->setMinimumHeight(90); m_queueTable->setMaximumHeight(160);
    lay->addWidget(m_queueTable);
    return grp;
}

// ── Config area ───────────────────────────────────────────────────────────────
// ── Network & Training group (placed to the right of the dataset queue) ─────
void PreTrainNNCPanel::buildNetworkGroup() {
    m_netGrp = new QGroupBox("Network & Training");
    auto* netF   = new QFormLayout(m_netGrp); netF->setSpacing(5);
    m_hiddenEdit = new QLineEdit("16"); m_hiddenEdit->setToolTip("e.g. 64 or 64,32");
    m_actCombo   = new QComboBox;
    m_actCombo->addItem("ReLU","relu"); m_actCombo->addItem("Sigmoid","sigmoid");
    m_actCombo->addItem("Tanh","tanh"); m_actCombo->addItem("Linear","linear");
    m_valSpin    = new QDoubleSpinBox;
    m_valSpin->setRange(0,0.5); m_valSpin->setValue(0.2); m_valSpin->setSingleStep(0.05);
    m_iwSpin     = new QSpinBox; m_iwSpin->setRange(2,200); m_iwSpin->setValue(10);
    m_iwSpin->setToolTip("Iw: hidden nodes for the small Phase-1 pre-training network");
    netF->addRow("Hidden layers:", m_hiddenEdit);
    netF->addRow("Activation:", m_actCombo);
    netF->addRow("Validation split:", m_valSpin);
    netF->addRow("Pre-train nodes Iw:", m_iwSpin);
    m_equalChk = new QCheckBox("Baseline with equal total budget");
    m_equalChk->setChecked(true);
    m_equalChk->setToolTip("On: the baseline (no pre-training) gets as many evaluations as Phase 1 + 2 (+ 3)\n"
                           "together, so the comparison measures the pre-training itself.\n"
                           "Off: the baseline gets only the Phase-2 budget (original protocol).");
    netF->addRow("", m_equalChk);
}

void PreTrainNNCPanel::buildConfigArea(QVBoxLayout* root) {
    static const QStringList phColors = {
        "QGroupBox{border:1px solid #E3B341;border-radius:6px;padding-top:14px;}"
        "QGroupBox::title{color:#E3B341;font-weight:600;}",
        "QGroupBox{border:1px solid #58A6FF;border-radius:6px;padding-top:14px;}"
        "QGroupBox::title{color:#58A6FF;font-weight:600;}",
        "QGroupBox{border:1px solid #56D364;border-radius:6px;padding-top:14px;}"
        "QGroupBox::title{color:#56D364;font-weight:600;}"
    };
    auto* phRow = new QHBoxLayout; phRow->setSpacing(10);
    m_ph1 = new PhaseBox(1, false); m_ph1->setStyleSheet(phColors[0]);
    m_ph2 = new PhaseBox(2, false); m_ph2->setStyleSheet(phColors[1]);
    m_ph3 = new PhaseBox(3, true);  m_ph3->setStyleSheet(phColors[2]);
    phRow->addWidget(m_ph1); phRow->addWidget(m_ph2); phRow->addWidget(m_ph3);
    root->addLayout(phRow);

    // Run / Stop / N-runs row
    auto* bRow = new QHBoxLayout;
    m_runBtn  = new QPushButton("▶  Run Pre-Train + NNC");
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
        "When N > 1 the Comparison table shows mean \u00b1 std.");
    auto* nRunsLbl = new QLabel("Runs:");
    nRunsLbl->setToolTip(m_nRunsSpin->toolTip());

    bRow->addWidget(m_runBtn);
    bRow->addWidget(m_stopBtn);
    bRow->addSpacing(16);
    bRow->addWidget(nRunsLbl);
    bRow->addWidget(m_nRunsSpin);
    bRow->addStretch();
    root->addLayout(bRow);
    connect(m_runBtn, &QPushButton::clicked, this, &PreTrainNNCPanel::onRunClicked);
    connect(m_stopBtn,&QPushButton::clicked, this, &PreTrainNNCPanel::onStopClicked);

    m_phaseNameLbl = new QLabel(" ");
    m_phaseNameLbl->setStyleSheet("color:#58A6FF;font-size:12px;font-style:italic;");
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
    connect(m_tickTimer,&QTimer::timeout,this,&PreTrainNNCPanel::onTimerTick);
}

// ── Scroll tab helper ─────────────────────────────────────────────────────────
static QPushButton* mkExportBtn(const QString& lbl) {
    auto* b = new QPushButton(lbl);
    b->setStyleSheet("QPushButton{background:#3D5A80;color:white;border-radius:5px;"
                     "padding:5px 14px;font-size:12px;}");
    return b;
}
static QWidget* makeScrollTab(QVBoxLayout*& outLayout, QScrollArea*& outScroll, QPushButton* expBtn) {
    auto* content = new QWidget;
    outLayout = new QVBoxLayout(content);
    outLayout->setContentsMargins(8,8,8,8); outLayout->setSpacing(16);
    outLayout->addStretch();
    outScroll = new QScrollArea;
    outScroll->setWidget(content); outScroll->setWidgetResizable(true);
    outScroll->setFrameShape(QFrame::NoFrame);
    auto* tab = new QWidget; auto* tl = new QVBoxLayout(tab); tl->setContentsMargins(4,4,4,4);
    tl->addWidget(outScroll);
    auto* er = new QHBoxLayout; er->addWidget(expBtn); er->addStretch(); tl->addLayout(er);
    return tab;
}

// ── Tab widget ─────────────────────────────────────────────────────────────────
void PreTrainNNCPanel::buildTabWidget(QVBoxLayout* root) {
    m_tabs = new QTabWidget; m_tabs->setMinimumHeight(360);

    // Tab 0: Loss
    {
        auto* eb = mkExportBtn("💾  Export all Loss PNG (300 DPI)");
        connect(eb,&QPushButton::clicked,this,&PreTrainNNCPanel::onExportLossPng);
        m_tabs->addTab(makeScrollTab(m_lossLayout, m_lossScroll, eb),"📈  Loss per iteration");
    }

    // Tab 1: Evaluation
    {
        auto* tab = new QWidget; auto* tl = new QVBoxLayout(tab); tl->setContentsMargins(8,8,8,8);
        m_evalTable = new QTableWidget(0,4);
        m_evalTable->setHorizontalHeaderLabels({"Dataset","Task","Metric","Value"});
        m_evalTable->horizontalHeader()->setSectionResizeMode(0,QHeaderView::Stretch);
        m_evalTable->horizontalHeader()->setSectionResizeMode(1,QHeaderView::ResizeToContents);
        m_evalTable->horizontalHeader()->setSectionResizeMode(2,QHeaderView::ResizeToContents);
        m_evalTable->horizontalHeader()->setSectionResizeMode(3,QHeaderView::ResizeToContents);
        m_evalTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
        m_evalTable->setSelectionBehavior(QAbstractItemView::SelectRows);
        tl->addWidget(m_evalTable);
        auto* er = new QHBoxLayout;
        auto* eb = mkExportBtn("💾  Export Evaluation XLSX");
        connect(eb,&QPushButton::clicked,this,&PreTrainNNCPanel::onExportEvalXlsx);
        er->addWidget(eb); er->addStretch(); tl->addLayout(er);
        m_tabs->addTab(tab,"📋  Evaluation");
    }

    // Tab 2: Comparison
    {
        auto* tab = new QWidget; auto* tl = new QVBoxLayout(tab); tl->setContentsMargins(8,8,8,8);
        m_compTable = new QTableWidget(0,2);
        m_compTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
        m_compTable->setSelectionBehavior(QAbstractItemView::SelectRows);
        tl->addWidget(m_compTable);
        auto* er = new QHBoxLayout;
        auto* eb = mkExportBtn("💾  Export Comparison XLSX");
        connect(eb,&QPushButton::clicked,this,&PreTrainNNCPanel::onExportCompXlsx);
        er->addWidget(eb); er->addStretch(); tl->addLayout(er);
        m_tabs->addTab(tab,"🔬  Comparison");
    }

    // Tab 3: Bar chart
    {
        auto* tab = new QWidget; auto* tl = new QVBoxLayout(tab); tl->setContentsMargins(8,8,8,8);
        m_barChart = new QChart;
        m_barChart->setTitle("Error % / MSE comparison");
        styleChart(m_barChart, false);
        m_barChart->setMargins(QMargins(20, 20, 20, 20));
        m_barView = new QChartView(m_barChart);
        m_barView->setRenderHint(QPainter::Antialiasing);
        m_barView->setMinimumHeight(320);
        m_barScroll = new QScrollArea;
        m_barScroll->setWidget(m_barView);
        m_barScroll->setWidgetResizable(false);
        m_barScroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
        m_barScroll->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        m_barScroll->setFrameShape(QFrame::NoFrame);
        m_barScroll->setMinimumHeight(340);
        tl->addWidget(m_barScroll, 1);
        auto* er = new QHBoxLayout;
        auto* eb = mkExportBtn("💾  Export Bar Chart PNG (300 DPI)");
        connect(eb, &QPushButton::clicked, this, &PreTrainNNCPanel::onExportBarPng);
        er->addWidget(eb); er->addStretch(); tl->addLayout(er);
        m_tabs->addTab(tab, "📊  Bar chart");
    }

    root->addWidget(m_tabs);
}

// ── chart styling ─────────────────────────────────────────────────────────────
void PreTrainNNCPanel::styleChart(QChart* c, bool dark) {
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
    c->legend()->setLabelBrush(QBrush(dark?QColor("#E4E4EC"):Qt::black));
    c->legend()->setBackgroundVisible(!dark);
}

// ── Queue management ──────────────────────────────────────────────────────────
void PreTrainNNCPanel::addToQueue(const QStringList& paths) {
    for (const QString& p : paths) if(!m_queue.contains(p)) m_queue<<p;
    refreshQueueTable();
    m_statusLbl->setText(QString("%1 dataset(s) queued.").arg(m_queue.size()));
}
void PreTrainNNCPanel::refreshQueueTable() {
    m_queueTable->setRowCount(m_queue.size());
    for (int i=0;i<m_queue.size();++i) {
        auto* n=new QTableWidgetItem(QString::number(i+1)); n->setTextAlignment(Qt::AlignCenter);
        m_queueTable->setItem(i,0,n);
        auto* nm=new QTableWidgetItem(QFileInfo(m_queue[i]).fileName()); nm->setToolTip(m_queue[i]);
        m_queueTable->setItem(i,1,nm);
        m_queueTable->setItem(i,2,new QTableWidgetItem("Queued"));
    }
}
void PreTrainNNCPanel::onAddFiles() {
    const QStringList p=QFileDialog::getOpenFileNames(
        this,"Add datasets",{},DatasetLoader::fileDialogFilter());
    if(!p.isEmpty()) addToQueue(p);
}
void PreTrainNNCPanel::onRemoveSelected() {
    if(m_running)return;
    QSet<int> toRm; for(auto* i:m_queueTable->selectedItems()) toRm.insert(i->row());
    QList<int> s(toRm.begin(),toRm.end()); std::sort(s.begin(),s.end(),std::greater<int>());
    for(int r:s) m_queue.removeAt(r); refreshQueueTable();
}
void PreTrainNNCPanel::onClearQueue() {
    if(m_running)return; m_queue.clear(); refreshQueueTable();
    m_statusLbl->setText("Queue cleared.");
}

// ── Progress ──────────────────────────────────────────────────────────────────
void PreTrainNNCPanel::rebuildProgress(int n) {
    for(auto*b:m_bars)b->deleteLater(); for(auto*l:m_barLbls)l->deleteLater();
    m_bars.clear(); m_barLbls.clear();
    static const QStringList cols={"#888888","#E3B341","#58A6FF","#56D364"};
    static const QStringList lbls={"Baseline:","Phase 1  Pre-Train:","Phase 2  Fine-tune:","Phase 3  Fine-tune:"};
    auto* lay=qobject_cast<QVBoxLayout*>(m_progressContainer->layout());
    while(lay->count()){auto*i=lay->takeAt(0);delete i->widget();delete i;}
    for(int ph=0;ph<n;++ph){
        auto* row=new QHBoxLayout;
        auto* l=new QLabel(ph<lbls.size()?lbls[ph]:QString("Phase %1:").arg(ph+1)); l->setFixedWidth(180);
        auto* bar=new QProgressBar; bar->setRange(0,100); bar->setValue(0);
        bar->setStyleSheet(QString("QProgressBar::chunk{background:%1;}").arg(cols[ph%cols.size()]));
        row->addWidget(l); row->addWidget(bar);
        lay->addLayout(row); m_bars.push_back(bar); m_barLbls.push_back(l);
    }
}

// ── Loss chart management ─────────────────────────────────────────────────────
void PreTrainNNCPanel::clearLossCharts() {
    m_curLossSeries.clear();
    m_curLossAxisX = nullptr;
    m_curLossAxisY = nullptr;
    m_curLossChart = nullptr;
    m_lossCharts.clear();
    m_lossViews.clear();
    while(m_lossLayout->count()>1){
        auto*i=m_lossLayout->takeAt(0);
        if(i->widget()) i->widget()->deleteLater();
        delete i;
    }
}

void PreTrainNNCPanel::startNewLossChart(const QString& dsName, int nPhases) {
    auto* chart = new QChart;
    chart->setTitle(QString("Loss per iteration  —  %1").arg(dsName));
    styleChart(chart, true);
    auto* axX=new QValueAxis; axX->setTitleText("Iteration");
    auto* axY=new QValueAxis; axY->setTitleText("Loss");
    chart->addAxis(axX,Qt::AlignBottom); chart->addAxis(axY,Qt::AlignLeft);
    axX->setRange(0,10); axY->setRange(0,1);

    static const QList<QColor> cols={QColor("#888888"),QColor("#E3B341"),QColor("#58A6FF"),QColor("#56D364")};
    static const QStringList names={"Baseline","Phase 1  Pre-Train","Phase 2  Fine-tune","Phase 3  Fine-tune"};
    m_curLossSeries.clear();
    for(int ph=0;ph<nPhases;++ph){
        auto* s=new QLineSeries;
        s->setName(ph<names.size()?names[ph]:QString("Phase %1").arg(ph+1));
        s->setColor(cols[ph%cols.size()]); chart->addSeries(s);
        s->attachAxis(axX); s->attachAxis(axY); m_curLossSeries.append(s);
    }
    auto* view=new QChartView(chart); view->setRenderHint(QPainter::Antialiasing);
    view->setMinimumHeight(260); view->setFixedHeight(280);
    const int pos=std::max(0,m_lossLayout->count()-1);
    m_lossLayout->insertWidget(pos,view);
    if(auto* content=qobject_cast<QWidget*>(m_lossLayout->parent()))
        content->updateGeometry();
    if(m_lossScroll)
        QTimer::singleShot(30, this, [this, view](){
            m_lossScroll->ensureWidgetVisible(view);
        });
    m_lossCharts.append(chart); m_lossViews.append(view);
    m_curLossChart=chart; m_curLossAxisX=axX; m_curLossAxisY=axY;
}

// ── PNG export helper ─────────────────────────────────────────────────────────
bool PreTrainNNCPanel::exportStackedPng(const QList<QChart*>& charts,
                                        QWidget* parent, const QString& defaultName) {
    if(charts.isEmpty()){ QMessageBox::information(parent,"Export","No charts yet."); return false; }
    const QString path=QFileDialog::getSaveFileName(parent,"Export",defaultName,"PNG images (*.png)");
    if(path.isEmpty()) return false;
    const int W=2400, perH=800, H=perH*charts.size();
    const int dpm=qRound(300.0/0.0254);
    QImage img(W,H,QImage::Format_RGB32);
    img.setDotsPerMeterX(dpm); img.setDotsPerMeterY(dpm); img.fill(QColor("#1e1e2a"));
    QPainter painter(&img); painter.setRenderHint(QPainter::Antialiasing);
    for(int i=0;i<charts.size();++i)
        if(charts[i]->scene()) charts[i]->scene()->render(&painter,QRectF(0,i*perH,W,perH));
    painter.end();
    if(img.save(path,"PNG",100)){
        QMessageBox::information(parent,"Export OK",
            QString("Saved %1 dataset(s), 300 DPI (%2×%3 px):\n%4")
                .arg(charts.size()).arg(W).arg(H).arg(path));
        return true;
    }
    QMessageBox::critical(parent,"Export Error","Failed to save PNG."); return false;
}

// ── refreshInfoLabel: always shows run / dataset / phase on one line ──────────
void PreTrainNNCPanel::refreshInfoLabel(const QString& phaseInfo) {
    QString s;
    if (m_totalRuns > 1)
        s += QString("Run %1/%2   ").arg(m_currentRun+1).arg(m_totalRuns);
    if (m_totalDs > 0)
        s += QString("Dataset %1/%2: %3   ").arg(m_currentDsIdx+1).arg(m_totalDs).arg(m_currentDsName);
    s += phaseInfo;
    m_statusLbl->setText(s);
}

// ── buildConfig / launchWorker ────────────────────────────────────────────────
PreTrainConfig PreTrainNNCPanel::buildConfig() const {
    PreTrainConfig cfg;
    cfg.datasetPaths  = m_queue;
    cfg.hiddenLayout  = m_hiddenEdit->text().trimmed();
    cfg.activation    = m_actCombo->currentData().toString();
    cfg.valSplit      = m_valSpin->value();
    cfg.Iw            = m_iwSpin->value();
    cfg.phase1        = m_ph1->config();
    cfg.phase2        = m_ph2->config();
    cfg.phase3        = m_ph3->config();
    cfg.equalBudgetBaseline = m_equalChk && m_equalChk->isChecked();
    return cfg;
}

void PreTrainNNCPanel::launchWorker(const PreTrainConfig& cfg) {
    auto* worker = new PreTrainWorker;
    worker->setConfig(cfg);
    auto* thread = new QThread(this);
    worker->moveToThread(thread);
    m_thread = thread;
    m_worker = worker;

    connect(thread,&QThread::started,             worker,&PreTrainWorker::run);
    connect(worker,&PreTrainWorker::datasetStarted,  this,&PreTrainNNCPanel::onDatasetStarted);
    connect(worker,&PreTrainWorker::phaseStarted,    this,&PreTrainNNCPanel::onPhaseStarted);
    connect(worker,&PreTrainWorker::iterationDone,   this,&PreTrainNNCPanel::onIterationDone);
    connect(worker,&PreTrainWorker::phaseFinished,   this,&PreTrainNNCPanel::onPhaseFinished);
    connect(worker,&PreTrainWorker::datasetFinished, this,&PreTrainNNCPanel::onDatasetFinished);
    connect(worker,&PreTrainWorker::allDone,         this,&PreTrainNNCPanel::onAllDone);
    connect(worker,&PreTrainWorker::errorOccurred,   this,&PreTrainNNCPanel::onError);
    connect(m_stopBtn,&QPushButton::clicked,         worker,&PreTrainWorker::requestStop);
    connect(worker,&PreTrainWorker::allDone,         thread,&QThread::quit);
    connect(worker,&PreTrainWorker::errorOccurred,   thread,&QThread::quit);
    connect(thread,&QThread::finished,               worker,&QObject::deleteLater);
    connect(thread,&QThread::finished,               thread,&QObject::deleteLater);

    thread->start();
}

// ── Run ───────────────────────────────────────────────────────────────────────
void PreTrainNNCPanel::onRunClicked() {
    if(m_queue.isEmpty()){QMessageBox::information(this,"Pre-Train NNC","Add at least one dataset first.");return;}

    // Stop any previous run
    if(m_worker) m_worker->requestStop();
    if(m_thread && m_thread->isRunning()){
        m_thread->quit();
        m_thread->wait(4000);
    }
    m_worker = nullptr;
    m_thread = nullptr;

    // Full UI reset
    const bool hasPhase3=(m_ph3->config().optimizer!="none");
    m_activePhases=3+(hasPhase3?1:0);
    rebuildProgress(m_activePhases);
    for(auto*b:m_bars)b->setValue(0);

    clearLossCharts();

    m_evalTable->clearContents(); m_evalTable->setRowCount(0);
    m_compTable->clearContents(); m_compTable->setRowCount(0);
    m_compTable->setColumnCount(0);
    m_barEntries.clear();
    m_barChart->removeAllSeries();
    for(auto* ax : m_barChart->axes()) m_barChart->removeAxis(ax);

    for(int i=0;i<m_queue.size();++i)
        m_queueTable->setItem(i,2,new QTableWidgetItem("Queued"));

    m_phaseIters.assign(m_activePhases,0);
    m_xOff.assign(m_activePhases+1,0);
    {
        const int Ng2=m_ph2->config().iterations;
        const int Ng1=m_ph1->config().iterations;
        const int Ng3=m_ph3->config().iterations;
        int pi=0;
        if(pi<m_activePhases) m_phaseIters[pi++]=Ng2;
        if(pi<m_activePhases) m_phaseIters[pi++]=Ng1;
        if(pi<m_activePhases) m_phaseIters[pi++]=Ng2;
        if(hasPhase3&&pi<m_activePhases) m_phaseIters[pi++]=Ng3;
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

void PreTrainNNCPanel::onStopClicked() {
    m_stopRequested = true;
    if(m_worker) m_worker->requestStop();
}

// ── Worker signal handlers ────────────────────────────────────────────────────
void PreTrainNNCPanel::onDatasetStarted(int dsIdx,int totalDs,QString dsName){
    m_currentDsName=dsName;
    m_currentDsIdx =dsIdx;
    m_totalDs      =totalDs;
    for(auto*b:m_bars)b->setValue(0);
    std::fill(m_xOff.begin(),m_xOff.end(),0);
    startNewLossChart(dsName,m_activePhases);
    if(dsIdx<m_queueTable->rowCount())
        m_queueTable->setItem(dsIdx,2,new QTableWidgetItem("Running…"));
    refreshInfoLabel("—");
    m_tabs->setCurrentIndex(0);
}

void PreTrainNNCPanel::onPhaseStarted(int phase,int total,QString name){
    m_phaseNameLbl->setText(name);
    refreshInfoLabel(QString("%1  (%2 iter)").arg(name).arg(total));
}

void PreTrainNNCPanel::onIterationDone(int phase,int iter,int total,double loss,double){
    const int idx=phase-1;
    if(idx<(int)m_bars.size()) m_bars[idx]->setValue((int)(100.0*iter/std::max(1,total)));
    if(idx<m_curLossSeries.size()&&m_curLossAxisX){
        const int xb=(idx<(int)m_xOff.size())?m_xOff[idx]:0;
        m_curLossSeries[idx]->append(xb+iter,loss);
        if(iter%10==0){
            double yMin=1e9,yMax=-1e9;
            for(auto*s:m_curLossSeries) for(const auto&pt:s->points()){yMin=std::min(yMin,pt.y());yMax=std::max(yMax,pt.y());}
            if(yMax>yMin) m_curLossAxisY->setRange(yMin*0.9,yMax*1.1);
            m_curLossAxisX->setRange(0,std::max(xb+total,10));
        }
    }
}

void PreTrainNNCPanel::onPhaseFinished(int phase,double loss){
    const int idx=phase-1;
    if(idx<(int)m_bars.size())m_bars[idx]->setValue(100);
    const int iters=(idx<(int)m_phaseIters.size())?m_phaseIters[idx]:200;
    if(idx+1<(int)m_xOff.size())m_xOff[idx+1]=m_xOff[idx]+iters;
    refreshInfoLabel(QString("Phase %1 done  (loss=%2)").arg(phase).arg(loss,0,'g',5));
}

void PreTrainNNCPanel::onTimerTick(){
    const int sec=static_cast<int>(m_clock.elapsed()/1000);
    const int total=m_totalRuns * m_queue.size();
    QString eta;
    if(m_dsDone>0 && m_dsDone<total){
        const int etaSec=static_cast<int>(
            (double)m_clock.elapsed()/m_dsDone*(total-m_dsDone)/1000.0);
        eta=QString("  |  ETA %1m%2s").arg(etaSec/60).arg(etaSec%60,2,10,QChar('0'));
    }
    m_timeLbl->setText(QString("Elapsed  %1:%2%3")
        .arg(sec/60).arg(sec%60,2,10,QChar('0')).arg(eta));
}

// ── onDatasetFinished: accumulate results ─────────────────────────────────────
void PreTrainNNCPanel::onDatasetFinished(QString dsName,QStringList configs,
                                         QVector<double> trainLoss,
                                         QVector<double> trainErr,
                                         QVector<double> valErr,
                                         QVector<double> testErr,bool isCls){
    ++m_dsDone;
    for(int i=0;i<m_queue.size();++i)
        if(QFileInfo(m_queue[i]).baseName()==dsName)
            m_queueTable->setItem(i,2,new QTableWidgetItem(
                m_currentRun < m_totalRuns-1
                    ? QString("Run %1/%2 done").arg(m_currentRun+1).arg(m_totalRuns)
                    : "Done ✓"));

    // Accumulate for N-runs
    auto& acc = m_accumResults[dsName];
    if(acc.trainErr.isEmpty()){
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
    for(int c=0;c<configs.size();++c){
        acc.testErr[c].append (c<testErr.size()   ? testErr[c]   : -1.0);
        acc.trainErr[c].append(c<trainErr.size()  ? trainErr[c]  : 0.0);
        acc.valErr[c].append  (c<valErr.size()    ? valErr[c]    : -1.0);
        acc.trainLoss[c].append(c<trainLoss.size() ? trainLoss[c] : 0.0);
    }

    // Update bar chart live with running mean (visible after each run)
    BarEntry be;
    be.dsName=dsName; be.configs=acc.configs; be.isClassification=isCls;
    be.trainErr.resize(configs.size());
    for(int c=0;c<configs.size();++c){
        auto [m,s]=runStats(acc.trainErr[c]);
        be.trainErr[c]=m;
    }
    bool found=false;
    for(auto& entry:m_barEntries){if(entry.dsName==dsName){entry=be;found=true;break;}}
    if(!found) m_barEntries.append(be);
    rebuildBarChart();
}

// ── onAllDone: restart or finalize ────────────────────────────────────────────
void PreTrainNNCPanel::onAllDone(std::shared_ptr<NeuralNetwork> net,
                                  std::shared_ptr<Dataset>       ds){
    m_lastNet = net;
    m_lastDs  = ds;

    if(!m_stopRequested && m_currentRun < m_totalRuns - 1){
        // ── Intermediate run: prepare next ───────────────────────────────
        ++m_currentRun;

        // Clear loss charts for the new run (charts reflect current run's data)
        clearLossCharts();

        // Reset queue status and progress bars
        for(int i=0;i<m_queue.size();++i)
            m_queueTable->setItem(i,2,new QTableWidgetItem("Queued"));
        for(auto*b:m_bars)b->setValue(0);
        std::fill(m_xOff.begin(),m_xOff.end(),0);

        m_statusLbl->setText(QString("Run %1/%2 — starting next run...")
            .arg(m_currentRun+1).arg(m_totalRuns));
        m_phaseNameLbl->setText(" ");

        launchWorker(m_lastConfig);
    } else {
        // ── Last run (or stopped): finalize ──────────────────────────────
        finalizeAllRuns();
    }
}

// ── finalizeAllRuns: compute mean±std and populate tables ─────────────────────
void PreTrainNNCPanel::finalizeAllRuns() {
    m_evalTable->clearContents(); m_evalTable->setRowCount(0);
    m_compTable->clearContents(); m_compTable->setRowCount(0);
    m_compTable->setColumnCount(0);
    m_barEntries.clear();
    m_barChart->removeAllSeries();
    for(auto* ax : m_barChart->axes()) m_barChart->removeAxis(ax);

    const bool hasVal = m_valSpin->value() > 0;
    const int  N      = m_currentRun + 1;   // actual completed runs

    for(auto it = m_accumResults.begin(); it != m_accumResults.end(); ++it){
        const QString&  dsName = it.key();
        const DsAccum&  acc    = it.value();
        const QString errLbl = acc.isCls ? "Error %" : "MSE";
        // regression errors with 4 significant digits (two decimals turned an MSE of 0.00035 into "0.00")
        auto fmtE = [&](const QVector<double>& v) { return acc.isCls ? fmtMS(v) : fmtMS(v, 'g', 4); };
        const QString valLbl = acc.isCls ? "Val Error %" : "Val MSE";
        const int lastIdx = acc.configs.size() - 1;

        // ── Evaluation tab: last configuration's stats ────────────────────
        auto addE=[&](const QString& metric, const QString& value){
            const int r=m_evalTable->rowCount(); m_evalTable->setRowCount(r+1);
            m_evalTable->setItem(r,0,new QTableWidgetItem(dsName));
            m_evalTable->setItem(r,1,new QTableWidgetItem(acc.isCls?"Classification":"Regression"));
            m_evalTable->setItem(r,2,new QTableWidgetItem(metric));
            m_evalTable->setItem(r,3,new QTableWidgetItem(value));
        };
        if(lastIdx>=0){
            addE(errLbl, fmtE(acc.trainErr[lastIdx]));
            if(hasVal){
                // Filter out invalid (-1) val entries
                QVector<double> vv;
                for(double x:acc.valErr[lastIdx]) if(x>=0) vv.append(x);
                if(!vv.isEmpty()) addE(valLbl, fmtE(vv));
            }
            {
                QVector<double> tv;
                for(double x:acc.testErr[lastIdx]) if(x>=0) tv.append(x);
                if(!tv.isEmpty()) addE(acc.isCls ? "Test Error %" : "Test MSE", fmtE(tv));
            }
            addE("Final Loss", fmtMS(acc.trainLoss[lastIdx],'g',4));
            if(N>1) addE("Runs", QString::number(N));
        }

        // ── Comparison table ──────────────────────────────────────────────
        if(m_compTable->columnCount()<3){
            const int nCols=2+acc.configs.size();
            m_compTable->setColumnCount(nCols);
            QStringList hdr; hdr<<"Dataset"<<"Metric";
            for(const QString& c:acc.configs) hdr<<c;
            m_compTable->setHorizontalHeaderLabels(hdr);
            m_compTable->horizontalHeader()->setSectionResizeMode(0,QHeaderView::ResizeToContents);
            m_compTable->horizontalHeader()->setSectionResizeMode(1,QHeaderView::ResizeToContents);
            for(int c=2;c<nCols;++c)
                m_compTable->horizontalHeader()->setSectionResizeMode(c,QHeaderView::Stretch);
        }
        auto addC=[&](const QString& metric, const std::function<QString(int)>& fn){
            const int r=m_compTable->rowCount(); m_compTable->setRowCount(r+1);
            m_compTable->setItem(r,0,new QTableWidgetItem(dsName));
            m_compTable->setItem(r,1,new QTableWidgetItem(metric));
            for(int c=0;c<acc.configs.size();++c)
                m_compTable->setItem(r,c+2,new QTableWidgetItem(fn(c)));
        };
        addC(errLbl,   [&](int c){ return fmtE(acc.trainErr[c]); });
        if(hasVal) addC(valLbl, [&](int c){
            QVector<double> vv;
            for(double x:acc.valErr[c]) if(x>=0) vv.append(x);
            return vv.isEmpty() ? QStringLiteral("-") : fmtE(vv);
        });
        {
            bool anyTest=false;
            for(const auto& v:acc.testErr) for(double x:v) anyTest = anyTest || x>=0;
            if(anyTest) addC(acc.isCls ? "Test Error %" : "Test MSE", [&](int c){
                QVector<double> tv;
                for(double x:acc.testErr[c]) if(x>=0) tv.append(x);
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
        if(N>1) addC(QString("Runs"), [&](int){ return QString::number(N); });

        // ── Bar chart entry (mean values) ─────────────────────────────────
        BarEntry be;
        be.dsName=dsName; be.configs=acc.configs; be.isClassification=acc.isCls;
        be.trainErr.resize(acc.configs.size());
        for(int c=0;c<acc.configs.size();++c){
            auto [mean,sd]=runStats(acc.trainErr[c]);
            be.trainErr[c]=mean;
        }
        m_barEntries.append(be);
    }
    rebuildBarChart();

    setRunning(false);
    m_phaseNameLbl->setText(" ");
    const QString nStr = N > 1 ? QString(" (%1 runs, mean \u00b1 std)").arg(N) : "";
    m_statusLbl->setText(QString("✓  All %1 dataset(s) complete%2.  All tabs updated.")
        .arg(m_queue.size()).arg(nStr));
    m_tabs->setCurrentIndex(2);
    if(m_lastNet) emit trainingCompleted(m_lastNet, m_lastDs);
}

void PreTrainNNCPanel::onError(QString msg){
    m_stopRequested = true;
    finalizeAllRuns();    // show partial results from completed runs
    QMessageBox::critical(this,"Pre-Train NNC Error",msg);
    m_statusLbl->setText(QString("Error (run %1/%2): %3")
        .arg(m_currentRun+1).arg(m_totalRuns).arg(msg));
}

void PreTrainNNCPanel::setRunning(bool r){
    m_running=r; m_runBtn->setEnabled(!r); m_stopBtn->setEnabled(r);
    m_nRunsSpin->setEnabled(!r);
    // Lock the Phase 1/2/3 configuration groups while a run is in progress —
    // changing them mid-run would not affect the in-flight worker.
    if(m_ph1) m_ph1->setEnabled(!r);
    if(m_ph2) m_ph2->setEnabled(!r);
    if(m_ph3) m_ph3->setEnabled(!r);
    if(m_netGrp) m_netGrp->setEnabled(!r);   // network / validation / Iw were editable during a run
    if(r){
        m_dsDone=0; m_clock.start(); m_tickTimer->start();
        m_timeLbl->setText("Elapsed  0:00"); m_timeLbl->show();
    } else {
        m_tickTimer->stop(); onTimerTick();
    }
    emit runningStateChanged(r);
}

// ── Exports ───────────────────────────────────────────────────────────────────
void PreTrainNNCPanel::onExportLossPng(){
    exportStackedPng(m_lossCharts,this,"loss_pretrain.png");
}
static void tblToXlsx(QTableWidget* tbl,const QString& path,const QString& sheet){
    XlsxWriter xls;auto*sh=xls.addSheet(sheet);
    QStringList hdr; for(int c=0;c<tbl->columnCount();++c)
        hdr<<(tbl->horizontalHeaderItem(c)?tbl->horizontalHeaderItem(c)->text():"");
    sh->setHeader(hdr);
    for(int r=0;r<tbl->rowCount();++r){
        std::vector<XlsxCell> row;
        for(int c=0;c<tbl->columnCount();++c){
            const QString txt=tbl->item(r,c)?tbl->item(r,c)->text():"";
            bool ok;double val=txt.toDouble(&ok);
            row.push_back(ok?XlsxCell(val):XlsxCell(txt));
        }
        sh->addRow(row);
    }
    xls.save(path);
}
void PreTrainNNCPanel::onExportEvalXlsx(){
    const QString p=QFileDialog::getSaveFileName(this,"Export Evaluation",
        "evaluation_pretrain.xlsx",XlsxWriter::fileFilter());
    if(p.isEmpty())return;
    try{tblToXlsx(m_evalTable,p,"Evaluation");m_statusLbl->setText("Exported: "+QFileInfo(p).fileName());}
    catch(const std::exception&ex){QMessageBox::critical(this,"Export Error",ex.what());}
}
void PreTrainNNCPanel::onExportCompXlsx(){
    const QString p=QFileDialog::getSaveFileName(this,"Export Comparison",
        "comparison_pretrain.xlsx",XlsxWriter::fileFilter());
    if(p.isEmpty())return;
    try{tblToXlsx(m_compTable,p,"Comparison");m_statusLbl->setText("Exported: "+QFileInfo(p).fileName());}
    catch(const std::exception&ex){QMessageBox::critical(this,"Export Error",ex.what());}
}

// ── Bar chart ─────────────────────────────────────────────────────────────────
void PreTrainNNCPanel::rebuildBarChart() {
    m_barChart->removeAllSeries();
    const auto oldAxes = m_barChart->axes();
    for (auto* ax : oldAxes) { m_barChart->removeAxis(ax); delete ax; }
    if (m_barEntries.isEmpty()) return;

    QStringList allConfigs = m_barEntries.first().configs;
    const bool isCls = m_barEntries.first().isClassification;
    const QString yLabel = isCls ? "Error %" : "MSE";

    static const QList<QColor> cfgColors = {
        QColor("#888888"), QColor("#58A6FF"), QColor("#56D364")
    };

    QList<QBarSet*> barSets;
    for (int ci = 0; ci < allConfigs.size(); ++ci) {
        auto* bs = new QBarSet(allConfigs[ci]);
        bs->setColor(cfgColors[ci % cfgColors.size()]);
        bs->setBorderColor(cfgColors[ci % cfgColors.size()].darker(130));
        barSets.append(bs);
    }

    QStringList categories;
    double yMax = 0;
    for (const auto& be : m_barEntries) {
        categories << be.dsName;
        for (int ci = 0; ci < allConfigs.size(); ++ci) {
            const double v = (ci < be.trainErr.size()) ? be.trainErr[ci] : 0.0;
            *barSets[ci] << v;
            yMax = std::max(yMax, v);
        }
    }

    auto* series = new QBarSeries;
    series->setBarWidth(0.75);
    for (int ci = 0; ci < allConfigs.size(); ++ci) {
        series->append(barSets[ci]);
        if (ci < allConfigs.size() - 1) {
            auto* spacer = new QBarSet("");
            spacer->setColor(Qt::transparent);
            spacer->setBorderColor(Qt::transparent);
            for (int di = 0; di < m_barEntries.size(); ++di) *spacer << 0.0;
            series->append(spacer);
        }
    }
    series->setLabelsVisible(true);
    series->setLabelsFormat("@value");
    series->setLabelsPrecision(2);
    series->setLabelsPosition(QAbstractBarSeries::LabelsOutsideEnd);
    m_barChart->addSeries(series);

    auto* axX = new QBarCategoryAxis;
    axX->append(categories);
    { QFont f = axX->labelsFont(); f.setPointSize(10); f.setBold(true); axX->setLabelsFont(f); }
    m_barChart->addAxis(axX, Qt::AlignBottom);
    series->attachAxis(axX);

    auto* axY = new QValueAxis;
    axY->setTitleText(yLabel);
    axY->setRange(0, yMax * 1.20);
    axY->setLabelFormat("%.2f");
    m_barChart->addAxis(axY, Qt::AlignLeft);
    series->attachAxis(axY);

    const int N = m_currentRun + 1;
    m_barChart->setTitle(
        N > 1
        ? QString("Pre-Train NNC — %1 comparison  (mean of %2 runs)").arg(yLabel).arg(N)
        : QString("Pre-Train NNC — %1 comparison").arg(yLabel));
    m_barChart->legend()->setVisible(true);
    m_barChart->legend()->setAlignment(Qt::AlignBottom);

    const int nGroups = m_barEntries.size();
    const int viewW   = m_barScroll ? std::max(m_barScroll->viewport()->width(), 400) : 700;
    const int chartW  = std::max(viewW, nGroups * 120);
    m_barView->setFixedWidth(chartW);
    m_barView->setMinimumHeight(320);
}

void PreTrainNNCPanel::onExportBarPng() {
    if (m_barEntries.isEmpty()) {
        QMessageBox::information(this, "Export", "Run the analysis first.");
        return;
    }
    const QString path = QFileDialog::getSaveFileName(
        this, "Export Bar Chart", "barchart_pretrain.png", "PNG images (*.png)");
    if (path.isEmpty()) return;
    const int W = 2400, H = 1600;
    const int dpm = qRound(300.0 / 0.0254);
    QImage img(W, H, QImage::Format_RGB32);
    img.setDotsPerMeterX(dpm); img.setDotsPerMeterY(dpm);
    img.fill(Qt::white);
    QPainter painter(&img);
    painter.setRenderHint(QPainter::Antialiasing);
    if (m_barChart->scene())
        m_barChart->scene()->render(&painter, QRectF(0, 0, W, H));
    painter.end();
    if (img.save(path, "PNG", 100))
        QMessageBox::information(this, "Export OK",
            QString("Saved 300 DPI (%1×%2 px):\n%3").arg(W).arg(H).arg(path));
    else
        QMessageBox::critical(this, "Export Error", "Failed to save PNG.");
}

} // namespace NeuralStudio
