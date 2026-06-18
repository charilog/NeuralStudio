#include "NetworkPanel.h"
#include <QGroupBox>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QScrollArea>
#include <QFrame>
#include "core/nn/NeuralNetwork.h"

namespace NeuralStudio {

NetworkPanel::NetworkPanel(QWidget* parent) : QWidget(parent) { buildUi(); }

bool NetworkPanel::isRBFMode() const {
    return m_stack && m_stack->currentIndex() == 1;
}

void NetworkPanel::buildUi() {
    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(24, 20, 24, 20);
    root->setSpacing(12);

    // ── Type switcher ─────────────────────────────────────────────────────────
    auto* typeRow = new QHBoxLayout;
    m_mlpBtn = new QPushButton("Multilayer Perceptron  (MLP)");
    m_rbfBtn = new QPushButton("Radial Basis Function  (RBF)");
    const auto styleOn  = QString("QPushButton{background:#1D9E75;color:white;border-radius:5px;font-weight:700;padding:4px 16px;}");
    const auto styleOff = QString("QPushButton{background:palette(button);border-radius:5px;padding:4px 16px;}");
    m_mlpBtn->setStyleSheet(styleOn); m_rbfBtn->setStyleSheet(styleOff);
    typeRow->addWidget(m_mlpBtn); typeRow->addWidget(m_rbfBtn); typeRow->addStretch();
    root->addLayout(typeRow);
    connect(m_mlpBtn, &QPushButton::clicked, this, &NetworkPanel::onTypeToggled);
    connect(m_rbfBtn, &QPushButton::clicked, this, &NetworkPanel::onTypeToggled);

    m_stack = new QStackedWidget;
    root->addWidget(m_stack, 1);

    // ──────────────────────────────────────────────────────────────────────────
    // PAGE 0: MLP
    // ──────────────────────────────────────────────────────────────────────────
    auto* mlpPage = new QWidget;
    auto* mlpRoot = new QVBoxLayout(mlpPage);
    mlpRoot->setContentsMargins(0,4,0,0); mlpRoot->setSpacing(12);

    m_infoLbl = new QLabel("Load a dataset first.");
    m_infoLbl->setStyleSheet("QLabel{background:palette(midlight);border-radius:4px;padding:6px 12px;font-size:12px;}");
    mlpRoot->addWidget(m_infoLbl);

    auto* taskGroup = new QGroupBox("Task");
    auto* taskForm  = new QFormLayout(taskGroup);
    m_taskCombo = new QComboBox;
    m_taskCombo->addItem("Binary Classification",      int(TaskType::BinaryClassification));
    m_taskCombo->addItem("Multi-Class Classification", int(TaskType::MultiClassClassification));
    m_taskCombo->addItem("Regression",                 int(TaskType::Regression));
    taskForm->addRow("Type:", m_taskCombo);
    mlpRoot->addWidget(taskGroup);

    auto* layersGroup = new QGroupBox("Hidden Layers");
    auto* layersRoot  = new QVBoxLayout(layersGroup);
    auto* topRow = new QHBoxLayout;
    topRow->addWidget(new QLabel("Number of hidden layers:"));
    m_layersSpin = new QSpinBox;
    m_layersSpin->setRange(0,6); m_layersSpin->setValue(2); m_layersSpin->setFixedWidth(70);
    topRow->addWidget(m_layersSpin); topRow->addStretch();
    layersRoot->addLayout(topRow);

    auto* scroll = new QScrollArea;
    scroll->setWidgetResizable(true); scroll->setFrameShape(QFrame::NoFrame);
    scroll->setMaximumHeight(220);
    m_layersWidget = new QWidget;
    m_layersLayout = new QVBoxLayout(m_layersWidget);
    m_layersLayout->setContentsMargins(0,4,0,4); m_layersLayout->setSpacing(6);
    scroll->setWidget(m_layersWidget);
    layersRoot->addWidget(scroll);

    auto* dropRow = new QHBoxLayout;
    m_dropoutSpin = new QDoubleSpinBox;
    m_dropoutSpin->setRange(0.0,0.9); m_dropoutSpin->setValue(0.0);
    m_dropoutSpin->setDecimals(2); m_dropoutSpin->setSingleStep(0.05); m_dropoutSpin->setFixedWidth(80);
    dropRow->addWidget(new QLabel("Dropout (hidden layers):")); dropRow->addWidget(m_dropoutSpin); dropRow->addStretch();
    layersRoot->addLayout(dropRow);
    mlpRoot->addWidget(layersGroup);

    m_buildBtn = new QPushButton("\u2699  Build Network");
    m_buildBtn->setFixedHeight(38); m_buildBtn->setEnabled(false);
    m_buildBtn->setStyleSheet("QPushButton:enabled{background:#1D9E75;color:white;border-radius:5px;font-weight:600;}"
                              "QPushButton:disabled{background:palette(midlight);border-radius:5px;}");
    mlpRoot->addWidget(m_buildBtn);

    m_summaryLbl = new QLabel;
    m_summaryLbl->setWordWrap(true);
    m_summaryLbl->setStyleSheet("QLabel{background:palette(midlight);border-radius:4px;padding:8px 12px;font-family:monospace;font-size:11px;}");
    m_summaryLbl->hide();
    mlpRoot->addWidget(m_summaryLbl);
    mlpRoot->addStretch();

    connect(m_layersSpin,&QSpinBox::valueChanged, this,&NetworkPanel::onNumLayersChanged);
    connect(m_buildBtn,  &QPushButton::clicked,   this,&NetworkPanel::onBuildClicked);
    rebuildLayerRows(2);
    m_stack->addWidget(mlpPage);   // page 0

    // ──────────────────────────────────────────────────────────────────────────
    // PAGE 1: RBF
    // ──────────────────────────────────────────────────────────────────────────
    auto* rbfPage = new QWidget;
    auto* rbfRoot = new QVBoxLayout(rbfPage);
    rbfRoot->setContentsMargins(0,4,0,0); rbfRoot->setSpacing(12);

    m_rbfInfoLbl = new QLabel("Load a dataset first.");
    m_rbfInfoLbl->setStyleSheet("QLabel{background:palette(midlight);border-radius:4px;padding:6px 12px;font-size:12px;}");
    m_rbfInfoLbl->setWordWrap(true);
    rbfRoot->addWidget(m_rbfInfoLbl);

    auto* rbfCfg  = new QGroupBox("RBF Configuration");
    auto* rbfForm = new QFormLayout(rbfCfg); rbfForm->setSpacing(8);

    m_rbfTaskCombo = new QComboBox;
    m_rbfTaskCombo->addItem("Binary Classification",      int(TaskType::BinaryClassification));
    m_rbfTaskCombo->addItem("Multi-Class Classification", int(TaskType::MultiClassClassification));
    m_rbfTaskCombo->addItem("Regression",                 int(TaskType::Regression));
    rbfForm->addRow("Task:", m_rbfTaskCombo);

    m_rbfCentersSpin = new QSpinBox;
    m_rbfCentersSpin->setRange(2,500); m_rbfCentersSpin->setValue(20);
    m_rbfCentersSpin->setToolTip("k = number of RBF centres. More centres → higher capacity, slower K-means. Typical: 10–100.");
    rbfForm->addRow("Centres (k):", m_rbfCentersSpin);

    m_rbfKernelCombo = new QComboBox;
    m_rbfKernelCombo->addItem("Gaussian   exp(-r²/2σ²)",         int(RBFKernel::Gaussian));
    m_rbfKernelCombo->addItem("Multiquadric   √(r²+σ²)",         int(RBFKernel::Multiquadric));
    m_rbfKernelCombo->addItem("Inv. Multiquadric   1/√(r²+σ²)", int(RBFKernel::InvMultiquadric));
    rbfForm->addRow("Kernel:", m_rbfKernelCombo);

    m_rbfWidthCombo = new QComboBox;
    m_rbfWidthCombo->addItem("Max Distance   σ = d_max/√(2k)",    int(RBFWidthStrategy::MaxDist));
    m_rbfWidthCombo->addItem("Nearest Neighbour (per-centre)",     int(RBFWidthStrategy::NearestNeighbor));
    m_rbfWidthCombo->addItem("Mean of 3 NNs   (per-centre)",      int(RBFWidthStrategy::MeanNN));
    rbfForm->addRow("Width Strategy:", m_rbfWidthCombo);

    m_rbfWidthScale = new QDoubleSpinBox;
    m_rbfWidthScale->setRange(0.01,100.0); m_rbfWidthScale->setValue(1.0);
    m_rbfWidthScale->setDecimals(2); m_rbfWidthScale->setSingleStep(0.1);
    m_rbfWidthScale->setToolTip("Multiplier on computed σ. >1 → wider/smoother, <1 → sharper/local.");
    rbfForm->addRow("Width Scale:", m_rbfWidthScale);

    m_rbfLambdaSpin = new QDoubleSpinBox;
    m_rbfLambdaSpin->setRange(1e-9,10.0); m_rbfLambdaSpin->setValue(1e-4);
    m_rbfLambdaSpin->setDecimals(8); m_rbfLambdaSpin->setSingleStep(1e-4);
    m_rbfLambdaSpin->setToolTip("Ridge regularisation λ in (ΦᵀΦ+λI)W=ΦᵀY. Prevents output weight explosion.");
    rbfForm->addRow("Ridge λ:", m_rbfLambdaSpin);

    m_rbfKmeansIter = new QSpinBox;
    m_rbfKmeansIter->setRange(10,1000); m_rbfKmeansIter->setValue(150);
    rbfForm->addRow("K-means iterations:", m_rbfKmeansIter);
    rbfRoot->addWidget(rbfCfg);

    m_buildRBFBtn = new QPushButton("\u2699  Build RBF Network");
    m_buildRBFBtn->setFixedHeight(38); m_buildRBFBtn->setEnabled(true);
    m_buildRBFBtn->setStyleSheet("QPushButton{background:#185fa5;color:white;border-radius:5px;font-weight:600;}"
                                 "QPushButton:disabled{background:palette(midlight);border-radius:5px;}");
    rbfRoot->addWidget(m_buildRBFBtn);

    auto* note = new QLabel("Training is analytic (K-means++ → widths → least squares). Completes in seconds. "
                            "Click \u25B6 Train after building.");
    note->setWordWrap(true);
    note->setStyleSheet("color:#8090C0;font-size:11px;font-style:italic;");
    rbfRoot->addWidget(note);
    rbfRoot->addStretch();

    connect(m_buildRBFBtn, &QPushButton::clicked, this, &NetworkPanel::onBuildRBFClicked);
    m_stack->addWidget(rbfPage);   // page 1
}

// ─── onTypeToggled ────────────────────────────────────────────────────────────
void NetworkPanel::onTypeToggled() {
    const bool wantRBF = (sender() == m_rbfBtn);
    m_stack->setCurrentIndex(wantRBF ? 1 : 0);
    const auto on  = QString("QPushButton{background:#1D9E75;color:white;border-radius:5px;font-weight:700;padding:4px 16px;}");
    const auto off = QString("QPushButton{background:palette(button);border-radius:5px;padding:4px 16px;}");
    m_mlpBtn->setStyleSheet(wantRBF ? off : on);
    m_rbfBtn->setStyleSheet(wantRBF ? on  : off);
}

// ─── onDatasetLoaded ─────────────────────────────────────────────────────────
void NetworkPanel::onDatasetLoaded(const Dataset* ds) {
    m_ds = ds;
    if (!ds) return;
    TaskType detected = detectTask(ds);
    int ci = (detected==TaskType::MultiClassClassification) ? 1
           : (detected==TaskType::Regression) ? 2 : 0;
    m_taskCombo->setCurrentIndex(ci);
    m_rbfTaskCombo->setCurrentIndex(ci);

    auto unique = ds->uniqueOutputValues(0);
    QString taskStr;
    if      (detected==TaskType::BinaryClassification)     taskStr = "Binary Classification";
    else if (detected==TaskType::MultiClassClassification) taskStr = QString("Multi-Class (%1 classes)").arg(unique.size());
    else                                                   taskStr = "Regression";

    QString info = QString("%1 inputs  |  %2 outputs  →  %3")
        .arg(ds->inputCount).arg(unique.size()).arg(taskStr);
    m_infoLbl->setText(info); m_rbfInfoLbl->setText(info);
    m_buildBtn->setEnabled(true); m_buildRBFBtn->setEnabled(true);
}

// ─── onNumLayersChanged ───────────────────────────────────────────────────────
void NetworkPanel::onNumLayersChanged(int n) { rebuildLayerRows(n); }

void NetworkPanel::rebuildLayerRows(int count) {
    QLayoutItem* item;
    while ((item = m_layersLayout->takeAt(0))) {
        if (item->widget()) item->widget()->deleteLater();
        delete item;
    }
    m_layerRows.clear();
    const int defaults[] = {64,32,16,8,8,4};
    for (int i=0; i<count; ++i) {
        auto* row=new QWidget; auto* rl=new QHBoxLayout(row);
        rl->setContentsMargins(0,0,0,0); rl->setSpacing(8);
        auto* lbl=new QLabel(QString("Layer %1:").arg(i+1)); lbl->setFixedWidth(60);
        auto* ns=new QSpinBox; ns->setRange(1,1024); ns->setValue(defaults[std::min(i,5)]); ns->setFixedWidth(80);
        auto* ac=new QComboBox; ac->addItems({"ReLU","Tanh","Sigmoid","Linear"}); ac->setFixedWidth(100);
        rl->addWidget(lbl); rl->addWidget(ns); rl->addWidget(new QLabel("neurons")); rl->addWidget(ac); rl->addStretch();
        m_layersLayout->addWidget(row);
        m_layerRows.push_back({ns,ac});
    }
    if (count==0) {
        auto* h=new QLabel("(No hidden layers — direct input→output)");
        h->setStyleSheet("color:#C0C0D0;font-style:italic;font-size:11px;");
        m_layersLayout->addWidget(h);
    }
}

// ─── onBuildClicked (MLP) ─────────────────────────────────────────────────────
void NetworkPanel::onBuildClicked() {
    if (!m_ds) return;
    TaskType task = static_cast<TaskType>(m_taskCombo->currentData().toInt());
    NetworkConfig cfg;
    cfg.task = task; cfg.dropoutRate = m_dropoutSpin->value();
    for (const auto& row : m_layerRows) {
        cfg.hiddenSizes.push_back(row.neurons->value());
        cfg.hiddenActivations.push_back(activationFromName(row.activation->currentText()));
    }
    m_net = std::make_shared<NeuralNetwork>();
    auto unique = m_ds->uniqueOutputValues(0);
    int nOut = (task==TaskType::MultiClassClassification) ? int(unique.size())
             : (task==TaskType::BinaryClassification)    ? 1
                                                         : m_ds->outputCount;
    if (task==TaskType::MultiClassClassification)
        m_net->setClassValues(std::vector<double>(unique.begin(), unique.end()));
    m_net->build(m_ds->inputCount, nOut, cfg);
    m_net->setNormalization(m_ds->inputStats);
    if (unique.size()>=2) m_net->setOutputMapping(unique.front(), unique.back());
    m_summaryLbl->setText("Architecture:  " + m_net->summary());
    m_summaryLbl->show();
    emit networkBuilt(m_net);
}

// ─── onBuildRBFClicked ────────────────────────────────────────────────────────
void NetworkPanel::onBuildRBFClicked() {
    if (!m_ds) {
        m_rbfInfoLbl->setText("Load a dataset first.");
        return;
    }
    TaskType task = static_cast<TaskType>(m_rbfTaskCombo->currentData().toInt());
    auto unique = m_ds->uniqueOutputValues(0);
    int nOut = (task==TaskType::MultiClassClassification) ? int(unique.size())
             : (task==TaskType::BinaryClassification)    ? 1
                                                         : m_ds->outputCount;
    RBFConfig cfg;
    cfg.nCenters      = m_rbfCentersSpin->value();
    cfg.kernel        = static_cast<RBFKernel>(m_rbfKernelCombo->currentData().toInt());
    cfg.widthStrategy = static_cast<RBFWidthStrategy>(m_rbfWidthCombo->currentData().toInt());
    cfg.widthScale    = m_rbfWidthScale->value();
    cfg.ridgeLambda   = m_rbfLambdaSpin->value();
    cfg.kmeansMaxIter = m_rbfKmeansIter->value();

    m_rbfNet = std::make_shared<RBFNetwork>();
    m_rbfNet->init(m_ds->inputCount, nOut, task, cfg);

    // ── Visual feedback ───────────────────────────────────────────────────────
    const QString kernelName = m_rbfKernelCombo->currentText().split(' ').first();
    m_rbfInfoLbl->setText(
        QString("\u2713  RBF Network built  \u2014  k=%1 centres | %2 | %3 inputs \u2192 %4 outputs\n"
                "Navigate to \u25B6 Training to start K-means + least squares.")
            .arg(cfg.nCenters).arg(kernelName)
            .arg(m_ds->inputCount).arg(nOut));
    m_rbfInfoLbl->setStyleSheet(
        "QLabel{background:#163D2B;border-radius:6px;padding:10px 14px;"
        "font-size:12px;color:#4EDDAD;font-weight:600;line-height:1.4;}");

    m_buildRBFBtn->setText("\u2713  RBF Network Built  \u2014  click again to rebuild");
    m_buildRBFBtn->setStyleSheet(
        "QPushButton{background:#1D9E75;color:white;border-radius:5px;font-weight:600;}");

    emit rbfBuilt(m_rbfNet);
}

void NetworkPanel::updateRBFInfo() {}

} // namespace NeuralStudio
