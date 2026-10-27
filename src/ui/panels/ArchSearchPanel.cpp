#include "ArchSearchPanel.h"
#include "core/export/XlsxWriter.h"
#include "core/dataset/DatasetLoader.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QDialog>
#include <QDialogButtonBox>
#include <QTableWidget>
#include <QHeaderView>
#include <QFileDialog>
#include <QMessageBox>
#include <QFileInfo>
#include <QFrame>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QMimeData>
#include <QScrollArea>
#include <QTimer>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace NeuralStudio {

// ── Statistics ────────────────────────────────────────────────────────────────
static std::pair<double,double> meanStd(const QVector<double>& v) {
    if (v.isEmpty()) return {0.0,0.0};
    double sum=0; for(double x:v)sum+=x;
    const double m=sum/v.size();
    if (v.size()==1) return {m,0.0};
    double var=0; for(double x:v)var+=(x-m)*(x-m);
    return {m, std::sqrt(var/(v.size()-1))};
}
static QString fmtMS(const QVector<double>& v, char fmt='f', int prec=2){
    if(v.isEmpty())return "-";
    auto[m,s]=meanStd(v);
    if(v.size()==1)return QString::number(m,fmt,prec);
    return QString("%1 \u00b1 %2").arg(QString::number(m,fmt,prec))
                                   .arg(QString::number(s,fmt,prec));
}

// ── Constructor ───────────────────────────────────────────────────────────────
ArchSearchPanel::ArchSearchPanel(QWidget* parent) : QWidget(parent) {
    setAcceptDrops(true);
    buildUi();
}
void ArchSearchPanel::dragEnterEvent(QDragEnterEvent* e){
    if(e->mimeData()->hasUrls())e->acceptProposedAction();
}
void ArchSearchPanel::dropEvent(QDropEvent* e){
    QStringList p; for(const auto&u:e->mimeData()->urls()){const QString s=u.toLocalFile();if(!s.isEmpty())p<<s;}
    addToQueue(p);
}

// ── buildUi ───────────────────────────────────────────────────────────────────
void ArchSearchPanel::buildUi(){
    auto* root=new QVBoxLayout(this); root->setContentsMargins(20,14,20,14); root->setSpacing(8);

    auto* title=new QLabel("Architecture Search");
    {QFont f=title->font();f.setPointSize(14);f.setBold(true);title->setFont(f);}
    root->addWidget(title);
    auto* sub=new QLabel(
        "Evolutionary outer optimizer explores the <b>architecture space</b> "
        "(layers, neurons, activations). Each candidate is trained with an inner optimizer "
        "and evaluated on the validation set. A <b>cache</b> avoids re-training identical architectures. "
        "Supports multiple datasets and <b>N independent runs</b> (mean \u00b1 std).");
    sub->setWordWrap(true);sub->setStyleSheet("color:palette(mid);font-size:12px;");
    root->addWidget(sub);

    buildUi_queueAndSearchSpace(root);
    buildOptimizerArea(root);
    buildProgressArea(root);
    buildTabWidget(root);
}

// ── Dataset queue + Search Space, side by side ───────────────────────────────
void ArchSearchPanel::buildUi_queueAndSearchSpace(QVBoxLayout* root){
    auto* row = new QHBoxLayout; row->setSpacing(10);
    QGroupBox* queueBox = buildQueueGroup();
    QGroupBox* searchBox = buildSearchSpace();
    row->addWidget(queueBox, 2);    // dataset queue: a bit wider
    row->addWidget(searchBox, 1);   // Search Space: to its right
    root->addLayout(row);
}

// ── Dataset queue ─────────────────────────────────────────────────────────────
QGroupBox* ArchSearchPanel::buildQueueGroup(){
    auto* grp=new QGroupBox("Dataset Queue  (drag && drop files here)");
    auto* lay=new QVBoxLayout(grp);lay->setSpacing(6);
    auto* btnRow=new QHBoxLayout;
    auto mkBtn=[&](const QString&t,const QString&bg="")->QPushButton*{
        auto*b=new QPushButton(t);b->setMinimumHeight(28);
        if(!bg.isEmpty())b->setStyleSheet(QString("QPushButton{background:%1;color:white;"
            "border-radius:4px;font-weight:600;padding:0 12px;}"
            "QPushButton:disabled{background:palette(midlight);}").arg(bg));
        return b;
    };
    m_addBtn=mkBtn("Add Files…");m_removeBtn=mkBtn("Remove Selected");m_clearBtn=mkBtn("Clear Queue","#8B3030");
    btnRow->addWidget(m_addBtn);btnRow->addWidget(m_removeBtn);btnRow->addWidget(m_clearBtn);btnRow->addStretch();
    lay->addLayout(btnRow);
    connect(m_addBtn,   &QPushButton::clicked,this,&ArchSearchPanel::onAddFiles);
    connect(m_removeBtn,&QPushButton::clicked,this,&ArchSearchPanel::onRemoveSelected);
    connect(m_clearBtn, &QPushButton::clicked,this,&ArchSearchPanel::onClearQueue);

    m_queueTable=new QTableWidget(0,3);
    m_queueTable->setHorizontalHeaderLabels({"#","Dataset","Status"});
    m_queueTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_queueTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_queueTable->horizontalHeader()->setSectionResizeMode(0,QHeaderView::ResizeToContents);
    m_queueTable->horizontalHeader()->setSectionResizeMode(1,QHeaderView::Stretch);
    m_queueTable->horizontalHeader()->setSectionResizeMode(2,QHeaderView::ResizeToContents);
    m_queueTable->verticalHeader()->hide();
    m_queueTable->verticalHeader()->setDefaultSectionSize(22);
    m_queueTable->setMinimumHeight(90);m_queueTable->setMaximumHeight(160);
    lay->addWidget(m_queueTable);
    return grp;
}

// ── Search space ──────────────────────────────────────────────────────────────
QGroupBox* ArchSearchPanel::buildSearchSpace(){
    auto* grp=new QGroupBox("Search Space");
    m_searchSpaceGroup=grp;
    grp->setStyleSheet("QGroupBox{border:1px solid #E3B341;border-radius:6px;padding-top:14px;}"
                       "QGroupBox::title{color:#E3B341;font-weight:600;}");
    auto* f=new QFormLayout(grp);f->setSpacing(5);

    m_minLayersSpin=new QSpinBox;m_minLayersSpin->setRange(1,6);m_minLayersSpin->setValue(1);
    m_maxLayersSpin=new QSpinBox;m_maxLayersSpin->setRange(1,6);m_maxLayersSpin->setValue(4);
    connect(m_minLayersSpin,&QSpinBox::valueChanged,this,[this](int v){
        if(m_maxLayersSpin->value()<v)m_maxLayersSpin->setValue(v);
    });
    auto* layRow=new QHBoxLayout;
    layRow->addWidget(new QLabel("Min:")); layRow->addWidget(m_minLayersSpin);
    layRow->addSpacing(8);
    layRow->addWidget(new QLabel("Max:")); layRow->addWidget(m_maxLayersSpin);
    layRow->addStretch();
    f->addRow("Hidden layers:", layRow);

    m_sizesEdit=new QLineEdit("4, 8, 16, 32, 64, 128");
    m_sizesEdit->setToolTip("Comma-separated candidate neuron counts per layer");
    // Preset buttons for convenience
    auto* presetsRow=new QHBoxLayout;
    auto mkPreset=[&](const QString&lbl,const QString&val)->QPushButton*{
        auto*b=new QPushButton(lbl);b->setFixedHeight(22);
        b->setStyleSheet("QPushButton{border:1px solid palette(mid);border-radius:3px;"
                         "padding:0 8px;font-size:10px;background:palette(button);}");
        connect(b,&QPushButton::clicked,this,[this,val](){m_sizesEdit->setText(val);});
        return b;
    };
    presetsRow->addWidget(mkPreset("XS","4, 8, 16"));
    presetsRow->addWidget(mkPreset("S","8, 16, 32, 64"));
    presetsRow->addWidget(mkPreset("M","16, 32, 64, 128"));
    presetsRow->addWidget(mkPreset("L","32, 64, 128, 256"));
    presetsRow->addWidget(mkPreset("XL","64, 128, 256, 512"));
    presetsRow->addStretch();
    auto* sizesCol=new QVBoxLayout;sizesCol->addWidget(m_sizesEdit);sizesCol->addLayout(presetsRow);
    f->addRow("Candidate sizes:", sizesCol);

    auto* actRow=new QHBoxLayout;
    m_actRelu   =new QCheckBox("ReLU");    m_actRelu->setChecked(true);
    m_actTanh   =new QCheckBox("Tanh");    m_actTanh->setChecked(true);
    m_actSigmoid=new QCheckBox("Sigmoid"); m_actSigmoid->setChecked(true);
    m_actLinear =new QCheckBox("Linear");
    actRow->addWidget(m_actRelu);actRow->addWidget(m_actTanh);
    actRow->addWidget(m_actSigmoid);actRow->addWidget(m_actLinear);actRow->addStretch();
    f->addRow("Activations:", actRow);

    m_valSpin=new QDoubleSpinBox;m_valSpin->setRange(0,0.5);m_valSpin->setValue(0.2);m_valSpin->setSingleStep(0.05);
    f->addRow("Validation split:", m_valSpin);
    m_topKSpin=new QSpinBox;m_topKSpin->setRange(1,50);m_topKSpin->setValue(10);
    f->addRow("Show top-K:", m_topKSpin);
    return grp;
}

// ── Optimizer area ────────────────────────────────────────────────────────────
void ArchSearchPanel::buildOptimizerArea(QVBoxLayout* root){
    static const QStringList kOLabels={"GA  (Genetic Algorithm)","DE  (Differential Evolution)",
        "PSO  (Particle Swarm)","CMA-ES","JSO","LM-CMA-ES","ACOR","CLPSO","ML-SHADE-RL"};
    static const QStringList kOKeys={"ga","de","pso","cmaes","jso","lmcmaes","acor","clpso","mlshaderl"};

    auto fillCombo=[&](QComboBox*c,int def=0){
        for(int i=0;i<kOLabels.size();++i)c->addItem(kOLabels[i],kOKeys[i]);
        c->setCurrentIndex(def);
    };

    auto* cfgRow=new QHBoxLayout;cfgRow->setSpacing(10);

    // Outer optimizer
    auto* g1=new QGroupBox("Outer Optimizer  —  Architecture Search");
    m_outerOptGroup=g1;
    g1->setStyleSheet("QGroupBox{border:1px solid #E3B341;border-radius:6px;padding-top:14px;}"
                      "QGroupBox::title{color:#E3B341;font-weight:600;}");
    auto* f1=new QFormLayout(g1);f1->setSpacing(4);
    m_outerOptCombo=new QComboBox;fillCombo(m_outerOptCombo,0); // GA default
    m_outerPopSpin =new QSpinBox;m_outerPopSpin->setRange(5,500);m_outerPopSpin->setValue(30);
    m_outerGensSpin=new QSpinBox;m_outerGensSpin->setRange(5,1000);m_outerGensSpin->setValue(50);
    f1->addRow("Optimizer:",  m_outerOptCombo);
    f1->addRow("Population:", m_outerPopSpin);
    f1->addRow("Generations:",m_outerGensSpin);
    auto* est=new QLabel;
    est->setStyleSheet("color:palette(mid);font-size:10px;");
    auto updateEst=[this,est](){
        const int n=m_outerPopSpin->value()*m_outerGensSpin->value();
        est->setText(QString("≤ %1 architecture evaluations").arg(n));
    };
    updateEst();
    connect(m_outerPopSpin, &QSpinBox::valueChanged, this, updateEst);
    connect(m_outerGensSpin,&QSpinBox::valueChanged, this, updateEst);
    f1->addRow("",est);
    cfgRow->addWidget(g1);

    // Inner optimizer
    auto* g2=new QGroupBox("Inner Optimizer  —  Network Training");
    m_innerOptGroup=g2;
    g2->setStyleSheet("QGroupBox{border:1px solid #58A6FF;border-radius:6px;padding-top:14px;}"
                      "QGroupBox::title{color:#58A6FF;font-weight:600;}");
    auto* f2=new QFormLayout(g2);f2->setSpacing(4);
    m_innerOptCombo =new QComboBox;fillCombo(m_innerOptCombo,1); // DE default
    m_innerPopSpin  =new QSpinBox;m_innerPopSpin->setRange(10,500);m_innerPopSpin->setValue(50);
    m_innerItersSpin=new QSpinBox;m_innerItersSpin->setRange(20,2000);m_innerItersSpin->setValue(50);
    m_innerWBSpin   =new QDoubleSpinBox;m_innerWBSpin->setRange(0.5,50);m_innerWBSpin->setValue(10.0);m_innerWBSpin->setSingleStep(0.5);
    f2->addRow("Optimizer:",   m_innerOptCombo);
    f2->addRow("Population:",  m_innerPopSpin);
    f2->addRow("Iterations:",  m_innerItersSpin);
    f2->addRow("Weight bound:",m_innerWBSpin);
    m_innerPolishSpin=new QSpinBox;m_innerPolishSpin->setRange(0,20000);m_innerPolishSpin->setValue(400);
    m_innerPolishSpin->setSingleStep(100);m_innerPolishSpin->setSpecialValueText("off");
    m_innerPolishSpin->setToolTip("Bound-respecting L-BFGS after the optimizer above, in evaluations.\n"
        "Without it every architecture gets the same derivative-free budget whatever its number of\n"
        "weights, so large networks are compared half-trained and the search favours tiny ones.");
    f2->addRow("L-BFGS polish:",m_innerPolishSpin);
    auto* warn=new QLabel("Each unique arch runs this training.");
    warn->setStyleSheet("color:palette(mid);font-size:10px;");
    f2->addRow("",warn);
    cfgRow->addWidget(g2);
    root->addLayout(cfgRow);
}

// ── Progress area ─────────────────────────────────────────────────────────────
void ArchSearchPanel::buildProgressArea(QVBoxLayout* root){
    auto* bRow=new QHBoxLayout;
    m_runBtn=new QPushButton("\u25B6  Run Architecture Search");
    m_stopBtn=new QPushButton("\u25A0  Stop");m_stopBtn->setEnabled(false);
    m_runBtn->setStyleSheet("QPushButton{background:#1D9E75;color:white;border-radius:6px;padding:6px 18px;"
        "font-weight:600;font-size:13px;}QPushButton:disabled{background:#555;color:#999;}");
    m_stopBtn->setStyleSheet("QPushButton{background:#B03030;color:white;border-radius:6px;padding:6px 18px;"
        "font-weight:600;font-size:13px;}QPushButton:disabled{background:#555;color:#999;}");
    m_nRunsSpin=new QSpinBox;m_nRunsSpin->setRange(1,30);m_nRunsSpin->setValue(1);
    m_nRunsSpin->setToolTip("Independent runs. When N>1, Comparison tab shows mean \u00b1 std.");
    auto* nLbl=new QLabel("Runs:");nLbl->setToolTip(m_nRunsSpin->toolTip());

    // "Apply to Mode 1" — hidden until a search completes
    m_applyBtn = new QPushButton("\u25B6  Apply to Mode 1");
    m_applyBtn->setVisible(false);
    m_applyBtn->setToolTip(
        "Copy the best architecture to Mode 1 \u2192 Network tab.\n"
        "Then click Build Network there and proceed to Training.");
    m_applyBtn->setStyleSheet(
        "QPushButton{background:#3D5A80;color:white;border-radius:6px;padding:6px 18px;"
        "font-weight:600;font-size:13px;}"
        "QPushButton:hover{background:#4D6A90;}");

    bRow->addWidget(m_runBtn);bRow->addWidget(m_stopBtn);
    bRow->addSpacing(16);bRow->addWidget(nLbl);bRow->addWidget(m_nRunsSpin);
    bRow->addSpacing(24);bRow->addWidget(m_applyBtn);
    bRow->addStretch();
    root->addLayout(bRow);
    connect(m_runBtn, &QPushButton::clicked,this,&ArchSearchPanel::onRunClicked);
    connect(m_stopBtn,&QPushButton::clicked,this,&ArchSearchPanel::onStopClicked);
    connect(m_applyBtn,&QPushButton::clicked,this,[this](){
        if(m_applyInfos.isEmpty()) return;

        if(m_applyInfos.size()==1){
            // Single dataset — apply directly
            const auto& info=m_applyInfos.first();
            emit applyToMode1(info.datasetPath,info.bestResult.sizes,info.bestResult.actNames);
        } else {
            // Multiple datasets — let user choose
            QDialog dlg(this);
            dlg.setWindowTitle("Select Dataset to Apply to Mode 1");
            dlg.setMinimumWidth(560);
            auto* lay=new QVBoxLayout(&dlg);
            lay->addWidget(new QLabel(
                "Mode 5 searched <b>multiple datasets</b>. "
                "Select which dataset and best architecture to apply to Mode 1:"));

            auto* tbl=new QTableWidget(m_applyInfos.size(),3);
            tbl->setHorizontalHeaderLabels({"Dataset","Best Architecture","Val Error"});
            tbl->setEditTriggers(QAbstractItemView::NoEditTriggers);
            tbl->setSelectionBehavior(QAbstractItemView::SelectRows);
            tbl->setSelectionMode(QAbstractItemView::SingleSelection);
            tbl->verticalHeader()->hide();
            tbl->horizontalHeader()->setSectionResizeMode(0,QHeaderView::ResizeToContents);
            tbl->horizontalHeader()->setSectionResizeMode(1,QHeaderView::Stretch);
            tbl->horizontalHeader()->setSectionResizeMode(2,QHeaderView::ResizeToContents);

            for(int i=0;i<m_applyInfos.size();++i){
                const auto& info=m_applyInfos[i];
                const QString errStr=info.isCls
                    ? QString("%1%").arg(info.bestResult.valErr,0,'f',2)
                    : QString("%1").arg(info.bestResult.valErr,0,'g',4);
                tbl->setItem(i,0,new QTableWidgetItem(info.dsName));
                auto* archItem=new QTableWidgetItem(info.bestResult.desc);
                QFont f=archItem->font();f.setBold(true);archItem->setFont(f);
                tbl->setItem(i,1,archItem);
                tbl->setItem(i,2,new QTableWidgetItem(errStr));
            }
            tbl->selectRow(0);
            lay->addWidget(tbl);

            auto* hint=new QLabel(
                "<i>The best architecture for each dataset is highlighted. "
                "Click the row you want, then press Apply.</i>");
            hint->setWordWrap(true);
            hint->setStyleSheet("color:palette(mid);font-size:11px;");
            lay->addWidget(hint);

            auto* btnBox=new QDialogButtonBox;
            auto* applyBtn=btnBox->addButton("Apply to Mode 1",QDialogButtonBox::AcceptRole);
            applyBtn->setStyleSheet(
                "QPushButton{background:#1D9E75;color:white;border-radius:5px;"
                "padding:5px 16px;font-weight:600;}");
            btnBox->addButton(QDialogButtonBox::Cancel);
            connect(btnBox,&QDialogButtonBox::accepted,&dlg,&QDialog::accept);
            connect(btnBox,&QDialogButtonBox::rejected,&dlg,&QDialog::reject);
            lay->addWidget(btnBox);

            if(dlg.exec()==QDialog::Accepted){
                const int row=tbl->currentRow();
                if(row>=0&&row<m_applyInfos.size()){
                    const auto& info=m_applyInfos[row];
                    emit applyToMode1(info.datasetPath,
                                      info.bestResult.sizes,
                                      info.bestResult.actNames);
                }
            }
        }
    });

    // Combined info line
    m_infoLbl=new QLabel("Add datasets and click Run.");
    m_infoLbl->setStyleSheet("color:#58A6FF;font-size:11px;font-family:monospace;");
    m_infoLbl->setWordWrap(true);
    root->addWidget(m_infoLbl);

    // ── Progress bars: fastest (top) → slowest (bottom) ─────────────────────
    // Hierarchy: every time Eval fills → Gen advances one step.
    //            every time Gen fills  → Dataset advances one step.
    //            every time Dataset fills → Run advances one step.
    //            Overall always reflects total combined progress.
    // Gen/Dataset/Run rows are hidden entirely when their loop count is 1
    // (e.g. only 1 generation configured), since a bar that can only ever
    // read 0% or 100% adds no information and just clutters the UI.
    auto mkBar=[&](const QString&lbl,const QString&color,QWidget** outRow)
            ->QProgressBar*{
        auto*rowWidget=new QWidget;
        auto*row=new QHBoxLayout(rowWidget);
        row->setContentsMargins(0,0,0,0);row->setSpacing(6);
        auto*l=new QLabel(lbl);l->setFixedWidth(70);
        auto*bar=new QProgressBar;bar->setRange(0,100);bar->setValue(0);
        bar->setTextVisible(true);bar->setMinimumHeight(18);bar->setMaximumHeight(18);
        bar->setStyleSheet(QString(
            "QProgressBar{border:1px solid #333;border-radius:5px;background:#1a1a2e;"
            "font-size:10px;}"
            "QProgressBar::chunk{background:%1;border-radius:4px;}").arg(color));
        row->addWidget(l);row->addWidget(bar);
        root->addWidget(rowWidget);
        if(outRow)*outRow=rowWidget;
        return bar;
    };
    m_evalBar    = mkBar("Eval:",     "#58A6FF", nullptr);   // blue   — always shown
    m_genBar     = mkBar("Gen:",      "#E3B341", &m_genRow); // amber  — shown if G>1
    m_dsBar      = mkBar("Dataset:",  "#BC8CFF", &m_dsRow);  // purple — shown if D>1
    m_runBar     = mkBar("Run:",      "#F778A1", &m_runRow); // pink   — shown if N>1
    m_overallBar = mkBar("Overall:",  "#56D364", nullptr);   // green  — always shown

    auto* stRow=new QHBoxLayout;
    auto* stLbl=new QLabel; // placeholder — info is in m_infoLbl
    m_timeLbl=new QLabel;m_timeLbl->hide();
    m_timeLbl->setStyleSheet("QLabel{font-family:monospace;font-size:11px;color:#1D9E75;}");
    stRow->addStretch();stRow->addWidget(m_timeLbl);
    root->addLayout(stRow);

    m_tickTimer=new QTimer(this);m_tickTimer->setInterval(500);
    connect(m_tickTimer,&QTimer::timeout,this,&ArchSearchPanel::onTimerTick);
    (void)stLbl;
}

// ── Tab widget ────────────────────────────────────────────────────────────────
static QPushButton* mkExpBtn(const QString&l){
    auto*b=new QPushButton(l);
    b->setStyleSheet("QPushButton{background:#3D5A80;color:white;border-radius:5px;padding:5px 14px;font-size:12px;}");
    return b;
}

void ArchSearchPanel::buildTabWidget(QVBoxLayout* root){
    m_tabs=new QTabWidget;m_tabs->setMinimumHeight(340);

    // Tab 0: Top-K results table
    {
        auto* tab=new QWidget;auto* tl=new QVBoxLayout(tab);tl->setContentsMargins(8,8,8,8);
        m_resultsTable=new QTableWidget(0,9);
        m_resultsTable->setHorizontalHeaderLabels(
            {"Dataset","Rank","Architecture","Val Err","Test Err","Train Err","Loss","Params","Eval#"});
        m_resultsTable->horizontalHeader()->setSectionResizeMode(0,QHeaderView::ResizeToContents);
        m_resultsTable->horizontalHeader()->setSectionResizeMode(1,QHeaderView::ResizeToContents);
        m_resultsTable->horizontalHeader()->setSectionResizeMode(2,QHeaderView::Stretch);
        m_resultsTable->horizontalHeader()->setSectionResizeMode(3,QHeaderView::ResizeToContents);
        m_resultsTable->horizontalHeader()->setSectionResizeMode(4,QHeaderView::ResizeToContents);
        m_resultsTable->horizontalHeader()->setSectionResizeMode(5,QHeaderView::ResizeToContents);
        m_resultsTable->horizontalHeader()->setSectionResizeMode(6,QHeaderView::ResizeToContents);
        m_resultsTable->horizontalHeader()->setSectionResizeMode(7,QHeaderView::ResizeToContents);
        m_resultsTable->horizontalHeader()->setSectionResizeMode(8,QHeaderView::ResizeToContents);
        m_resultsTable->horizontalHeaderItem(3)->setToolTip("Validation error: the search SELECTED on it, so it is optimistic");
        m_resultsTable->horizontalHeaderItem(4)->setToolTip("Error on the companion test file (never seen by the search); - = no test file");
        m_resultsTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
        m_resultsTable->setAlternatingRowColors(true);
        m_resultsTable->setSortingEnabled(true);
        tl->addWidget(m_resultsTable);
        auto* er=new QHBoxLayout;auto*eb=mkExpBtn("\xF0\x9F\x92\xBE  Export Results XLSX");
        connect(eb,&QPushButton::clicked,this,&ArchSearchPanel::onExportResultsXlsx);
        er->addWidget(eb);er->addStretch();tl->addLayout(er);
        m_tabs->addTab(tab,"\xF0\x9F\x8F\x86  Top Architectures");
    }

    // Tab 1: Comparison (N>1 runs)
    {
        auto* tab=new QWidget;auto* tl=new QVBoxLayout(tab);tl->setContentsMargins(8,8,8,8);
        m_compTable=new QTableWidget(0,6);
        m_compTable->setHorizontalHeaderLabels(
            {"Dataset","Rank","Architecture","Val Err (mean \u00b1 std)","Test Err (mean \u00b1 std)","Runs"});
        m_compTable->horizontalHeader()->setSectionResizeMode(0,QHeaderView::ResizeToContents);
        m_compTable->horizontalHeader()->setSectionResizeMode(1,QHeaderView::ResizeToContents);
        m_compTable->horizontalHeader()->setSectionResizeMode(2,QHeaderView::Stretch);
        m_compTable->horizontalHeader()->setSectionResizeMode(3,QHeaderView::ResizeToContents);
        m_compTable->horizontalHeader()->setSectionResizeMode(4,QHeaderView::ResizeToContents);
        m_compTable->horizontalHeader()->setSectionResizeMode(5,QHeaderView::ResizeToContents);
        m_compTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
        m_compTable->setAlternatingRowColors(true);
        tl->addWidget(m_compTable);
        m_tabs->addTab(tab,"\xF0\x9F\x94\xAC  Comparison");
    }

    root->addWidget(m_tabs);
}

// ── Queue management ──────────────────────────────────────────────────────────
void ArchSearchPanel::addToQueue(const QStringList& paths){
    for(const QString&p:paths)if(!m_queue.contains(p))m_queue<<p;
    refreshQueueTable();
    m_infoLbl->setText(QString("%1 dataset(s) queued.").arg(m_queue.size()));
}
void ArchSearchPanel::refreshQueueTable(){
    m_queueTable->setRowCount(m_queue.size());
    for(int i=0;i<m_queue.size();++i){
        auto*n=new QTableWidgetItem(QString::number(i+1));n->setTextAlignment(Qt::AlignCenter);
        m_queueTable->setItem(i,0,n);
        auto*nm=new QTableWidgetItem(QFileInfo(m_queue[i]).fileName());nm->setToolTip(m_queue[i]);
        m_queueTable->setItem(i,1,nm);
        m_queueTable->setItem(i,2,new QTableWidgetItem("Queued"));
    }
}
void ArchSearchPanel::onAddFiles(){
    const QStringList p=QFileDialog::getOpenFileNames(this,"Add datasets",{},DatasetLoader::fileDialogFilter());
    if(!p.isEmpty())addToQueue(p);
}
void ArchSearchPanel::onRemoveSelected(){
    if(m_running)return;
    QSet<int>toRm;for(auto*i:m_queueTable->selectedItems())toRm.insert(i->row());
    QList<int>s(toRm.begin(),toRm.end());std::sort(s.begin(),s.end(),std::greater<int>());
    for(int r:s)m_queue.removeAt(r);refreshQueueTable();
}
void ArchSearchPanel::onClearQueue(){
    if(m_running)return;m_queue.clear();refreshQueueTable();m_infoLbl->setText("Queue cleared.");
}

// ── Info label ────────────────────────────────────────────────────────────────
void ArchSearchPanel::refreshInfoLabel(){
    QString s;
    if(m_totalRuns>1) s+=QString("Run %1/%2   ").arg(m_currentRun+1).arg(m_totalRuns);
    if(m_totalDs>0)   s+=QString("Dataset %1/%2: %3   ").arg(m_currentDsIdx+1).arg(m_totalDs).arg(m_currentDsName);

    // m_currentGen = completed generations. Show the one CURRENTLY EXECUTING = completed+1.
    // Cap at totalGens so after the last gen it still shows Gen 5/5.
    if(m_totalGens>0){
        const int dispGen = std::min(m_currentGen + 1, m_totalGens);
        s += QString("Gen %1/%2   ").arg(dispGen).arg(m_totalGens);
    }

    if(m_totalEvals>0) s+=QString("Eval %1/%2 (%3 unique / %4 cached)")
                           .arg(m_currentEval).arg(m_totalEvals)
                           .arg(m_uniqueEvals).arg(m_cacheHits);
    if(!m_bestDesc.isEmpty()){
        const QString fStr=m_isCls?QString("%1%").arg(m_bestFitness,0,'f',2)
                                  :QString("%1").arg(m_bestFitness,0,'g',4);
        s+=QString("\nBest: %1  \u2192  %2").arg(fStr).arg(m_bestDesc);
    }
    m_infoLbl->setText(s);
}

// ── buildConfig ───────────────────────────────────────────────────────────────
ArchSearchConfig ArchSearchPanel::buildConfig() const{
    ArchSearchConfig cfg;
    cfg.datasetPaths=m_queue;
    cfg.minLayers=m_minLayersSpin->value();
    cfg.maxLayers=m_maxLayersSpin->value();
    // Parse candidate sizes
    for(const QString&t:m_sizesEdit->text().split(',',Qt::SkipEmptyParts)){
        bool ok;int n=t.trimmed().toInt(&ok);if(ok&&n>0)cfg.candidateSizes.append(n);
    }
    if(cfg.candidateSizes.isEmpty())cfg.candidateSizes={4,8,16,32,64,128};
    // Activations
    if(m_actRelu->isChecked())    cfg.candidateActs<<"relu";
    if(m_actTanh->isChecked())    cfg.candidateActs<<"tanh";
    if(m_actSigmoid->isChecked()) cfg.candidateActs<<"sigmoid";
    if(m_actLinear->isChecked())  cfg.candidateActs<<"linear";
    if(cfg.candidateActs.isEmpty())cfg.candidateActs<<"relu";
    cfg.valSplit    =m_valSpin->value();
    cfg.topK        =m_topKSpin->value();
    cfg.outerOpt    =m_outerOptCombo->currentData().toString();
    cfg.outerPop    =m_outerPopSpin->value();
    cfg.outerGens   =m_outerGensSpin->value();
    cfg.innerOpt    =m_innerOptCombo->currentData().toString();
    cfg.innerPop    =m_innerPopSpin->value();
    cfg.innerIters  =m_innerItersSpin->value();
    cfg.innerWBound =m_innerWBSpin->value();
    cfg.innerPolishEvals=m_innerPolishSpin?m_innerPolishSpin->value():0;
    return cfg;
}

// ── launchWorker ──────────────────────────────────────────────────────────────
void ArchSearchPanel::launchWorker(const ArchSearchConfig& cfg){
    auto*worker=new ArchSearchWorker;worker->setConfig(cfg);
    auto*thread=new QThread(this);worker->moveToThread(thread);
    m_thread=thread;m_worker=worker;
    connect(thread,&QThread::started,               worker,&ArchSearchWorker::run);
    connect(worker,&ArchSearchWorker::datasetStarted,this,&ArchSearchPanel::onDatasetStarted);
    connect(worker,&ArchSearchWorker::evalDone,      this,&ArchSearchPanel::onEvalDone);
    connect(worker,&ArchSearchWorker::generationDone,this,&ArchSearchPanel::onGenerationDone);
    connect(worker,&ArchSearchWorker::datasetFinished,this,&ArchSearchPanel::onDatasetFinished);
    connect(worker,&ArchSearchWorker::allDone,       this,&ArchSearchPanel::onAllDone);
    connect(worker,&ArchSearchWorker::errorOccurred, this,&ArchSearchPanel::onError);
    connect(m_stopBtn,&QPushButton::clicked,         worker,&ArchSearchWorker::requestStop);
    connect(worker,&ArchSearchWorker::allDone,        thread,&QThread::quit);
    connect(worker,&ArchSearchWorker::errorOccurred,  thread,&QThread::quit);
    connect(thread,&QThread::finished,               worker,&QObject::deleteLater);
    connect(thread,&QThread::finished,               thread,&QObject::deleteLater);
    thread->start();
}

// ── Run / Stop ────────────────────────────────────────────────────────────────
void ArchSearchPanel::onRunClicked(){
    if(m_queue.isEmpty()){QMessageBox::information(this,"Architecture Search","Add at least one dataset.");return;}
    const ArchSearchConfig cfg=buildConfig();
    if(cfg.candidateSizes.isEmpty()||cfg.candidateActs.isEmpty()){
        QMessageBox::information(this,"Architecture Search","Check search space settings.");return;
    }
    if(m_worker)m_worker->requestStop();
    if(m_thread&&m_thread->isRunning()){m_thread->quit();m_thread->wait(4000);}
    m_worker=nullptr;m_thread=nullptr;

    // Reset UI
    m_resultsTable->clearContents();m_resultsTable->setRowCount(0);
    m_compTable->clearContents();m_compTable->setRowCount(0);
    for(int i=0;i<m_queue.size();++i)m_queueTable->setItem(i,2,new QTableWidgetItem("Queued"));
    if(m_overallBar)m_overallBar->setValue(0);
    if(m_runBar)m_runBar->setValue(0);
    m_dsBar->setValue(0);m_genBar->setValue(0);m_evalBar->setValue(0);
    m_applyBtn->setVisible(false);
    m_applyInfos.clear();

    // N-runs
    m_accumResults.clear();
    m_totalRuns    =m_nRunsSpin->value();
    m_currentRun   =0;
    m_stopRequested=false;
    m_lastConfig   =cfg;
    m_bestFitness  =1e9;m_bestDesc.clear();

    updateBarVisibility();   // show/hide Gen/Dataset/Run rows based on G/D/N for this run

    setRunning(true);
    launchWorker(m_lastConfig);
}
void ArchSearchPanel::onStopClicked(){m_stopRequested=true;if(m_worker)m_worker->requestStop();}

// ── Worker signals ────────────────────────────────────────────────────────────
void ArchSearchPanel::onDatasetStarted(int dsIdx,int totalDs,QString dsName){
    m_currentDsIdx=dsIdx;m_totalDs=totalDs;m_currentDsName=dsName;
    m_currentGen=0;m_currentEval=0;m_totalGens=m_lastConfig.outerGens;
    m_totalEvals=m_lastConfig.outerPop*m_lastConfig.outerGens;
    m_uniqueEvals=0;m_cacheHits=0;m_bestFitness=1e9;m_bestDesc.clear();
    m_genBar->setValue(0);m_evalBar->setValue(0);   // dsBar carries across datasets within a run
    if(dsIdx<m_queueTable->rowCount())
        m_queueTable->setItem(dsIdx,2,new QTableWidgetItem("Running…"));
    refreshInfoLabel();
    m_tabs->setCurrentIndex(0);
}

void ArchSearchPanel::onEvalDone(int evalNum,int totalEvals,double fitness,
                                  QString /*desc*/,bool cacheHit){
    m_currentEval=evalNum; m_totalEvals=totalEvals;
    if(cacheHit)++m_cacheHits; else ++m_uniqueEvals;

    const int P = std::max(1, m_lastConfig.outerPop);
    const int G = std::max(1, m_lastConfig.outerGens);
    const int D = std::max(1, (int)m_queue.size());
    const int N = std::max(1, m_totalRuns);

    // ── Eval (blue): 0→100% within current generation — fastest
    // Each completion = one generation done; triggers Gen += 100/G
    const int withinGen = ((evalNum - 1) % P) + 1;
    m_evalBar->setValue(withinGen * 100 / P);

    // ── Gen (amber): 0→100% within current dataset
    // evalNum resets per dataset so this auto-resets; each completion triggers Dataset += 100/D
    m_genBar->setValue(evalNum * 100 / (G * P));

    // ── Dataset (purple): 0→100% within current run, carries across datasets
    // Each completion = one run done; triggers Run += 100/N
    const int evalInRun = m_currentDsIdx * G * P + evalNum;
    m_dsBar->setValue(evalInRun * 100 / std::max(1, D * G * P));

    // ── Run (pink): DISCRETE — completed runs / N. Only the Dataset bar moves
    // continuously through a run; the Run bar stays put until the whole run
    // finishes, then jumps by 100/N. This keeps it visibly different from the
    // Overall bar (which always moves continuously with every single eval).
    if(m_runBar) m_runBar->setValue(m_currentRun * 100 / N);

    // ── Overall (green): 0→100% total — continuous, never resets, always visible
    const int overallDone  = m_currentRun * D * G * P + evalInRun;
    const int overallTotal = std::max(1, N * D * G * P);
    m_overallBar->setValue(overallDone * 100 / overallTotal);

    refreshInfoLabel();
}

void ArchSearchPanel::onGenerationDone(int gen,int totalGens,
                                        double bestFitness,QString bestDesc,
                                        int uniqueEvals,int cacheHits){
    m_currentGen=gen;m_totalGens=totalGens;
    m_uniqueEvals=uniqueEvals;m_cacheHits=cacheHits;
    m_bestFitness=bestFitness;m_bestDesc=bestDesc;
    refreshInfoLabel();
}

void ArchSearchPanel::onDatasetFinished(QString dsName,QVector<ArchResult> topResults,
                                         QVector<double> convergence,bool isCls){
    ++m_dsDone;
    m_isCls=isCls;
    for(int i=0;i<m_queue.size();++i)
        if(QFileInfo(m_queue[i]).baseName()==dsName)
            m_queueTable->setItem(i,2,new QTableWidgetItem(
                m_currentRun<m_totalRuns-1
                    ?QString("Run %1/%2 done").arg(m_currentRun+1).arg(m_totalRuns)
                    :"Done \u2713"));

    // Detect flat/degenerate convergence: all top results have identical fitness
    bool isDegenerate = false;
    if(topResults.size() >= 2){
        const double ref = topResults[0].valErr;
        isDegenerate = std::all_of(topResults.begin(), topResults.end(),
            [ref](const ArchResult& r){ return std::abs(r.valErr - ref) < 1e-6; });
    }

    // Accumulate for N runs
    auto&acc=m_accumResults[dsName];
    acc.isCls=isCls;
    acc.runResults.append(topResults);
    Q_UNUSED(convergence);   // per-generation curve no longer displayed (tab removed)

    // Live update of results table (shows current run's results)
    populateResultsTable(dsName,topResults,isCls,false);

    // Show warning in info label for degenerate case
    if(isDegenerate){
        m_infoLbl->setText(
            m_infoLbl->text() +
            QString("\n\u26A0  [%1] All %2 architectures have identical Val Err = %3%4. "
                    "This dataset may be degenerate for NN training (e.g. multiclass with 1 output). "
                    "Try a different dataset.")
            .arg(dsName).arg(topResults.size())
            .arg(topResults[0].valErr, 0, 'f', 2)
            .arg(isCls ? "%" : ""));
    }

    m_tabs->setCurrentIndex(0);   // Top Architectures
}

void ArchSearchPanel::onAllDone(){
    if(!m_stopRequested&&m_currentRun<m_totalRuns-1){
        ++m_currentRun;
        for(int i=0;i<m_queue.size();++i)m_queueTable->setItem(i,2,new QTableWidgetItem("Queued"));
        m_dsBar->setValue(0);m_genBar->setValue(0);m_evalBar->setValue(0);
        m_bestFitness=1e9;m_bestDesc.clear();
        m_infoLbl->setText(QString("Run %1/%2 — starting...").arg(m_currentRun+1).arg(m_totalRuns));
        launchWorker(m_lastConfig);
    } else {
        finalizeAllRuns();
    }
}

void ArchSearchPanel::onError(QString msg){
    m_stopRequested=true;
    finalizeAllRuns();
    QMessageBox::critical(this,"Architecture Search Error",msg);
}

// ── finalizeAllRuns ───────────────────────────────────────────────────────────
void ArchSearchPanel::finalizeAllRuns(){
    const int N=m_currentRun+1;

    // Rebuild full results table from all accumulated runs
    m_resultsTable->clearContents();m_resultsTable->setRowCount(0);
    for(auto it=m_accumResults.begin();it!=m_accumResults.end();++it){
        const auto&acc=it.value();
        if(acc.runResults.isEmpty())continue;
        // Show results from last run (or best run by rank-1 fitness)
        int bestRun=0;
        double bestVal=1e9;
        for(int r=0;r<acc.runResults.size();++r){
            if(!acc.runResults[r].isEmpty()&&acc.runResults[r][0].valErr<bestVal){
                bestVal=acc.runResults[r][0].valErr;bestRun=r;
            }
        }
        populateResultsTable(it.key(),acc.runResults[bestRun],acc.isCls,true);
    }

    // Rebuild comparison table
    rebuildComparisonTable();

    // Only force all bars to 100% on a genuine, natural completion of the
    // full search. If the user pressed Stop (or an error occurred), leave
    // the bars showing the actual partial progress reached.
    if(!m_stopRequested){
        m_evalBar->setValue(100);
        m_genBar->setValue(100);
        m_dsBar->setValue(100);
        if(m_runBar)     m_runBar->setValue(100);
        if(m_overallBar) m_overallBar->setValue(100);
    }
    setRunning(false);
    const QString nStr=N>1?QString(" (%1 runs, mean \u00b1 std)").arg(N):"";
    m_infoLbl->setText(QString("\u2713  All %1 dataset(s) complete%2.")
                       .arg(m_queue.size()).arg(nStr));

    // Build per-dataset apply info (best rank-1 result for each dataset)
    m_applyInfos.clear();
    for(auto it=m_accumResults.begin();it!=m_accumResults.end();++it){
        const QString& dsName=it.key();
        int bestRun=0; double bestVal=1e9;
        for(int r=0;r<it.value().runResults.size();++r){
            if(!it.value().runResults[r].isEmpty()&&it.value().runResults[r][0].valErr<bestVal){
                bestVal=it.value().runResults[r][0].valErr; bestRun=r;
            }
        }
        if(!it.value().runResults[bestRun].isEmpty()){
            DatasetApplyInfo info;
            // Find the original file path for this dataset
            for(const QString& p:m_queue)
                if(QFileInfo(p).baseName()==dsName){info.datasetPath=p;break;}
            info.dsName     = dsName;
            info.isCls      = it.value().isCls;
            info.bestResult = it.value().runResults[bestRun][0];
            m_applyInfos.append(info);
        }
    }

    // Show apply button with appropriate label
    if(!m_applyInfos.isEmpty()){
        if(m_applyInfos.size()==1){
            const auto& info=m_applyInfos.first();
            const QString errStr=info.isCls
                ? QString("%1%").arg(info.bestResult.valErr,0,'f',2)
                : QString("%1").arg(info.bestResult.valErr,0,'g',4);
            m_applyBtn->setText(
                QString("\u25B6  Apply to Mode 1 \u2014 %1  (%2 val err)")
                .arg(info.bestResult.desc).arg(errStr));
        } else {
            m_applyBtn->setText(
                QString("\u25B6  Apply to Mode 1\u2026  (%1 datasets \u2014 select one)")
                .arg(m_applyInfos.size()));
        }
        m_applyBtn->setVisible(true);
    }

    m_tabs->setCurrentIndex(N>1?1:0);   // Comparison if N>1, else Top Architectures
}

// ── populateResultsTable ──────────────────────────────────────────────────────
void ArchSearchPanel::populateResultsTable(const QString& dsName,
                                            const QVector<ArchResult>& results,
                                            bool isCls, bool append){
    if(!append){m_resultsTable->clearContents();m_resultsTable->setRowCount(0);}
    m_resultsTable->setSortingEnabled(false);
    const QString errLbl=isCls?"Error %":"MSE";
    for(const auto&r:results){
        const int row=m_resultsTable->rowCount();m_resultsTable->insertRow(row);
        // numeric columns sort by value: QTableWidgetItem compares the TEXT, so "10.5" sorted
        // before "9.2" (the value stored in UserRole was never used by the sort)
        struct NumItem : QTableWidgetItem {
            using QTableWidgetItem::QTableWidgetItem;
            bool operator<(const QTableWidgetItem& o) const override {
                const QVariant a = data(Qt::UserRole), b = o.data(Qt::UserRole);
                if (a.isValid() && b.isValid()) return a.toDouble() < b.toDouble();
                return QTableWidgetItem::operator<(o);
            }
        };
        auto mk=[&](const QString&t,double sortVal=-1)->QTableWidgetItem*{
            auto*it=new NumItem(t);it->setTextAlignment(Qt::AlignCenter);
            it->setData(Qt::UserRole,sortVal>=0?sortVal:1e300);   // "-" (no value) sorts last
            return it;
        };
        // error %: two decimals; MSE: four significant digits ('f',4 printed 0.0000 for small MSE)
        auto err=[&](double v){ return v<0?QString("-"):isCls?QString::number(v,'f',2):QString::number(v,'g',4); };
        m_resultsTable->setItem(row,0,mk(dsName));
        m_resultsTable->setItem(row,1,mk(QString::number(r.rank),r.rank));
        auto*descItem=new QTableWidgetItem(r.desc);
        descItem->setTextAlignment(Qt::AlignLeft|Qt::AlignVCenter);
        if(r.rank==1){QFont f=descItem->font();f.setBold(true);descItem->setFont(f);}
        m_resultsTable->setItem(row,2,descItem);
        m_resultsTable->setItem(row,3,mk(err(r.valErr),r.valErr));
        m_resultsTable->setItem(row,4,mk(err(r.testErr),r.testErr));
        m_resultsTable->setItem(row,5,mk(err(r.trainErr),r.trainErr));
        m_resultsTable->setItem(row,6,mk(QString::number(r.trainLoss,'g',4),r.trainLoss));
        m_resultsTable->setItem(row,7,mk(QString::number(r.numParams),r.numParams));
        m_resultsTable->setItem(row,8,mk(QString::number(r.evalNum),r.evalNum));
    }
    m_resultsTable->setSortingEnabled(true);
}

// ── rebuildComparisonTable ────────────────────────────────────────────────────
void ArchSearchPanel::rebuildComparisonTable(){
    m_compTable->clearContents();m_compTable->setRowCount(0);
    const int N=m_currentRun+1;

    for(auto it=m_accumResults.begin();it!=m_accumResults.end();++it){
        const QString&dsName=it.key();
        const auto&acc=it.value();
        if(acc.runResults.isEmpty())continue;
        const bool isCls=acc.isCls;

        // Collect top-K architectures across all runs (by description)
        // For each unique architecture desc, collect val errors from all runs that had it
        QMap<QString,QVector<double>> archVals;  // desc → list of val errors
        QMap<QString,QVector<double>> archTest;  // desc → list of test errors (where a test file exists)
        QMap<QString,int>             archRanks; // desc → best rank seen

        for(int r=0;r<acc.runResults.size();++r){
            for(const auto&ar:acc.runResults[r]){
                archVals[ar.desc].append(ar.valErr);
                if(ar.testErr>=0) archTest[ar.desc].append(ar.testErr);
                if(!archRanks.contains(ar.desc)||ar.rank<archRanks[ar.desc])
                    archRanks[ar.desc]=ar.rank;
            }
        }

        // Sort by mean val error, tiebreaker: best (lowest) numParams seen
        QMap<QString,int> archMinParams;
        for(int r=0;r<acc.runResults.size();++r)
            for(const auto&ar:acc.runResults[r])
                if(!archMinParams.contains(ar.desc)||ar.numParams<archMinParams[ar.desc])
                    archMinParams[ar.desc]=ar.numParams;

        QList<std::tuple<double,int,QString>> sorted;  // (meanVal, minParams, desc)
        for(auto jt=archVals.begin();jt!=archVals.end();++jt){
            auto[m,s]=meanStd(jt.value());
            sorted.append({m, archMinParams.value(jt.key(), 999999), jt.key()});
        }
        std::sort(sorted.begin(),sorted.end(),[](const auto&a,const auto&b){
            if(std::abs(std::get<0>(a)-std::get<0>(b))>1e-6) return std::get<0>(a)<std::get<0>(b);
            return std::get<1>(a)<std::get<1>(b);  // tiebreaker: fewer params
        });

        int rank=0;
        for(const auto&[meanVal,minParams,desc]:sorted){
            const auto&vals=archVals[desc];
            ++rank;
            const int row=m_compTable->rowCount();m_compTable->insertRow(row);
            m_compTable->setItem(row,0,new QTableWidgetItem(dsName));
            m_compTable->setItem(row,1,new QTableWidgetItem(QString::number(rank)));
            auto*dItem=new QTableWidgetItem(desc);
            dItem->setTextAlignment(Qt::AlignLeft|Qt::AlignVCenter);
            if(rank==1){QFont f=dItem->font();f.setBold(true);dItem->setFont(f);}
            m_compTable->setItem(row,2,dItem);
            m_compTable->setItem(row,3,new QTableWidgetItem(isCls?fmtMS(vals,'f',2):fmtMS(vals,'g',4)));
            const auto tv=archTest.value(desc);
            m_compTable->setItem(row,4,new QTableWidgetItem(tv.isEmpty()?QString("-"):isCls?fmtMS(tv,'f',2):fmtMS(tv,'g',4)));
            m_compTable->setItem(row,5,new QTableWidgetItem(
                QString("%1/%2").arg(vals.size()).arg(N)));
        }
    }
}

// ── setRunning ────────────────────────────────────────────────────────────────
void ArchSearchPanel::setRunning(bool r){
    m_running=r;m_runBtn->setEnabled(!r);m_stopBtn->setEnabled(r);m_nRunsSpin->setEnabled(!r);
    m_addBtn->setEnabled(!r);m_removeBtn->setEnabled(!r);m_clearBtn->setEnabled(!r);
    // Lock configuration while a search is running — changing the search space
    // or optimizer settings mid-run would not affect the in-flight worker and
    // would be misleading to the user.
    if(m_searchSpaceGroup) m_searchSpaceGroup->setEnabled(!r);
    if(m_outerOptGroup)    m_outerOptGroup->setEnabled(!r);
    if(m_innerOptGroup)    m_innerOptGroup->setEnabled(!r);
    if(r){m_dsDone=0;m_clock.start();m_tickTimer->start();m_timeLbl->setText("Elapsed  0:00");m_timeLbl->show();}
    else{m_tickTimer->stop();onTimerTick();}
    emit runningStateChanged(r);
}

// ── updateBarVisibility ────────────────────────────────────────────────────────
//  Hides a bar entirely when its loop only ever runs once — a bar that can
//  only read 0% or 100% the whole time carries no information for the user.
//  Based on m_lastConfig (G) and counts known at Run-time (D = queue size,
//  N = requested runs).
void ArchSearchPanel::updateBarVisibility(){
    const int G = m_lastConfig.outerGens;
    const int D = m_queue.size();
    const int N = m_totalRuns;
    if(m_genRow) m_genRow->setVisible(G > 1);
    if(m_dsRow)  m_dsRow->setVisible (D > 1);
    if(m_runRow) m_runRow->setVisible(N > 1);
}

// ── Timer tick ────────────────────────────────────────────────────────────────
void ArchSearchPanel::onTimerTick(){
    if(!m_running)return;
    const int sec=static_cast<int>(m_clock.elapsed()/1000);
    const int total=m_totalRuns*m_queue.size();
    QString eta;
    if(m_dsDone>0&&m_dsDone<total){
        const int etaSec=static_cast<int>((double)m_clock.elapsed()/m_dsDone*(total-m_dsDone)/1000.0);
        eta=QString("  |  ETA %1m%2s").arg(etaSec/60).arg(etaSec%60,2,10,QChar('0'));
    }
    m_timeLbl->setText(QString("Elapsed  %1:%2%3").arg(sec/60).arg(sec%60,2,10,QChar('0')).arg(eta));
}

// ── Exports ───────────────────────────────────────────────────
static void tblXlsx(QTableWidget*tbl,const QString&path,const QString&sheet){
    XlsxWriter xls;auto*sh=xls.addSheet(sheet);
    QStringList hdr;for(int c=0;c<tbl->columnCount();++c)
        hdr<<(tbl->horizontalHeaderItem(c)?tbl->horizontalHeaderItem(c)->text():"");
    sh->setHeader(hdr);
    for(int r=0;r<tbl->rowCount();++r){
        std::vector<XlsxCell>row;
        for(int c=0;c<tbl->columnCount();++c){
            const QString t=tbl->item(r,c)?tbl->item(r,c)->text():"";
            bool ok;double v=t.toDouble(&ok);row.push_back(ok?XlsxCell(v):XlsxCell(t));
        }
        sh->addRow(row);
    }
    xls.save(path);
}
void ArchSearchPanel::onExportResultsXlsx(){
    const QString p=QFileDialog::getSaveFileName(this,"Export",
        "arch_search_results.xlsx",XlsxWriter::fileFilter());
    if(p.isEmpty())return;
    try{
        XlsxWriter xls;
        // Sheet 1: Top architectures
        auto*sh1=xls.addSheet("Top Architectures");
        sh1->setHeader({"Dataset","Rank","Architecture","Val Err","Test Err","Train Err","Loss","Params","Eval#"});
        for(int r=0;r<m_resultsTable->rowCount();++r){
            std::vector<XlsxCell>row;
            for(int c=0;c<m_resultsTable->columnCount();++c){
                const QString t=m_resultsTable->item(r,c)?m_resultsTable->item(r,c)->text():"";
                bool ok;double v=t.toDouble(&ok);row.push_back(ok?XlsxCell(v):XlsxCell(t));
            }
            sh1->addRow(row);
        }
        // Sheet 2: Comparison (if N>1)
        if(m_compTable->rowCount()>0){
            auto*sh2=xls.addSheet("Comparison");
            sh2->setHeader({"Dataset","Rank","Architecture","Val Err (mean±std)","Test Err (mean±std)","Runs"});
            for(int r=0;r<m_compTable->rowCount();++r){
                std::vector<XlsxCell>row;
                for(int c=0;c<m_compTable->columnCount();++c){
                    const QString t=m_compTable->item(r,c)?m_compTable->item(r,c)->text():"";
                    bool ok;double v=t.toDouble(&ok);row.push_back(ok?XlsxCell(v):XlsxCell(t));
                }
                sh2->addRow(row);
            }
        }
        xls.save(p);
        m_infoLbl->setText("Exported: "+QFileInfo(p).fileName());
    }catch(const std::exception&ex){QMessageBox::critical(this,"Export Error",ex.what());}
}

} // namespace NeuralStudio
