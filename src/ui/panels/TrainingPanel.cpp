#include "TrainingPanel.h"
#include "utils/Logger.h"
#include <QApplication>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QFormLayout>
#include <QMessageBox>
#include <QFileDialog>
#include <QFile>
#include <QTextStream>
#include <QChart>
#include <QChartView>
#include <QLineSeries>
#include <QValueAxis>
#include <QLegend>
#include <QLegendMarker>
#include <algorithm>
#include <cmath>

namespace NeuralStudio {

TrainingPanel::TrainingPanel(QWidget* parent) : QWidget(parent) { buildUi(); }

// ─── styleChart ──────────────────────────────────────────────────────────────
//  Force-light styling for legend & axes so labels are always visible
//  regardless of system theme (the issue you saw with invisible legend was
//  because the dark theme paints white-on-white when the window is light).
void TrainingPanel::styleChart(QChart* c) {
    c->setBackgroundBrush(QBrush(QColor("#1e1e2a")));
    c->setBackgroundRoundness(6);
    c->setPlotAreaBackgroundBrush(QBrush(QColor("#272735")));
    c->setPlotAreaBackgroundVisible(true);
    c->setMargins(QMargins(8, 8, 8, 8));

    // Title
    c->setTitleBrush(QBrush(QColor("#E4E4EC")));
    QFont tf = c->titleFont(); tf.setBold(true); tf.setPointSize(11);
    c->setTitleFont(tf);

    // Legend
    c->legend()->setVisible(true);
    c->legend()->setAlignment(Qt::AlignTop);
    c->legend()->setLabelBrush(QBrush(QColor("#E4E4EC")));
    QFont lf = c->legend()->font(); lf.setPointSize(10);
    c->legend()->setFont(lf);
    c->legend()->setBackgroundVisible(false);
}

void TrainingPanel::buildUi() {
    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(20, 16, 20, 16);
    root->setSpacing(10);

    auto* title = new QLabel("Training");
    { QFont f = title->font(); f.setPointSize(14); f.setBold(true); title->setFont(f); }
    root->addWidget(title);

    // ── Config row ───────────────────────────────────────────────────────────
    auto* cfgGroup = new QGroupBox("Configuration");
    auto* cfgLay   = new QHBoxLayout(cfgGroup);
    cfgLay->setSpacing(14);

    auto addCol = [&](const QString& lbl, QWidget* w) {
        auto* col = new QVBoxLayout;
        auto* l   = new QLabel(lbl);
        l->setStyleSheet("color:#E8E8F0;font-size:11px;font-weight:600;");
        col->addWidget(l); col->addWidget(w);
        cfgLay->addLayout(col);
    };

    m_optimCombo = new QComboBox;
    // ── 1. Gradient Descent ─────────────────────────────────────────────────
    m_optimCombo->addItem("Adam",                                    QVariant("adam"));
    m_optimCombo->addItem("SGD",                                     QVariant("sgd"));
    m_optimCombo->addItem("AdamW  (decoupled weight decay)",         QVariant("adamw"));
    m_optimCombo->addItem("RMSProp",                                 QVariant("rmsprop"));
    m_optimCombo->addItem("Nadam  (Nesterov + Adam)",                QVariant("nadam"));
    m_optimCombo->addItem("AdaGrad",                                 QVariant("adagrad"));
    // ── 2. Quasi-Newton ─────────────────────────────────────────────────────
    m_optimCombo->insertSeparator(m_optimCombo->count());
    m_optimCombo->addItem("L-BFGS  (Limited Memory BFGS)",          QVariant("lbfgs"));
    m_optimCombo->addItem("BFGS  (quasi-Newton, m=20)",              QVariant("bfgs"));
    m_optimCombo->addItem("Conjugate Gradient  (Polak-Ribière)",     QVariant("cg"));
    m_optimCombo->addItem("Levenberg-Marquardt  (diagonal LM)",      QVariant("lm"));
    // ── 3. Single-point derivative-free ─────────────────────────────────────
    m_optimCombo->insertSeparator(m_optimCombo->count());
    m_optimCombo->addItem("Simulated Annealing",                     QVariant("sa"));
    m_optimCombo->addItem("Nelder-Mead Simplex",                     QVariant("neldermead"));
    // ── 4. Evolutionary / Swarm ──────────────────────────────────────────────
    m_optimCombo->insertSeparator(m_optimCombo->count());
    m_optimCombo->addItem("GA  (Genetic Algorithm)",                 QVariant("ga"));
    m_optimCombo->addItem("DE  (Differential Evolution)",            QVariant("de"));
    m_optimCombo->addItem("PSO  (Particle Swarm)",                   QVariant("pso"));
    m_optimCombo->addItem("CLPSO  (Comprehensive Learning PSO)",     QVariant("clpso"));
    m_optimCombo->addItem("ACOR  (Ant Colony — Continuous)",         QVariant("acor"));
    m_optimCombo->addItem("CMA-ES  (Covariance Matrix Adaptation)",  QVariant("cmaes"));
    m_optimCombo->addItem("LM-CMA-ES  (Limited Memory CMA-ES)",     QVariant("lmcmaes"));
    m_optimCombo->addItem("JSO  (jSO / Success-History SHADE)",      QVariant("jso"));
    m_optimCombo->addItem("ML-SHADE-RL  (Multi-operator SHADE+RL)",  QVariant("mlshaderl"));
    m_optimCombo->addItem("ARQ3  (NLPSR+SHADE+jSO+Eigen+TS+OBL)",   QVariant("arq3"));
    // ── 5. EDA ──────────────────────────────────────────────────────────────
    m_optimCombo->insertSeparator(m_optimCombo->count());
    m_optimCombo->addItem("PBIL  (Population-Based Incremental Learning)", QVariant("pbil"));
    m_optimCombo->addItem("UMDA  (Univariate Marginal Distribution)",       QVariant("umda"));
    m_optimCombo->setFixedWidth(350);
    m_optimCombo->setToolTip(
        "Gradient Descent (Adam…AdaGrad): use backprop gradient + LR.\n"
        "Quasi-Newton (L-BFGS…LM): backprop gradient, no LR needed.\n"
        "Single-Point (SA, NM): gradient-free, starts from current weights.\n"
        "Evolutionary/EDA (GA…UMDA): population-based, gradient-free.");
    addCol("Optimizer", m_optimCombo);

    m_lrSpin = new QDoubleSpinBox;
    m_lrSpin->setRange(1e-6, 1.0);
    m_lrSpin->setDecimals(6);
    m_lrSpin->setSingleStep(0.0001);
    m_lrSpin->setValue(0.001);                  // ← default fix
    m_lrSpin->setFixedWidth(100);
    // LR (gradient-only)
    {
        auto* col = new QVBoxLayout;
        m_lrLbl = new QLabel("Learning Rate");
        m_lrLbl->setStyleSheet("color:#E8E8F0;font-size:11px;font-weight:600;");
        col->addWidget(m_lrLbl);
        m_lrSpin = new QDoubleSpinBox;
        m_lrSpin->setRange(1e-6, 1.0); m_lrSpin->setDecimals(6);
        m_lrSpin->setSingleStep(0.0001); m_lrSpin->setValue(0.001); m_lrSpin->setFixedWidth(100);
        col->addWidget(m_lrSpin);
        cfgLay->addLayout(col);
    }

    // LR Schedule (gradient-only)
    {
        auto* col = new QVBoxLayout;
        m_schedLbl = new QLabel("LR Schedule");
        m_schedLbl->setStyleSheet("color:#E8E8F0;font-size:11px;font-weight:600;");
        col->addWidget(m_schedLbl);
        m_lrSchedCombo = new QComboBox;
        m_lrSchedCombo->addItem("Constant",   static_cast<int>(LRSchedule::Constant));
        m_lrSchedCombo->addItem("Step Decay", static_cast<int>(LRSchedule::StepDecay));
        m_lrSchedCombo->addItem("Exp. Decay", static_cast<int>(LRSchedule::ExponentialDecay));
        m_lrSchedCombo->setFixedWidth(95);
        col->addWidget(m_lrSchedCombo);
        cfgLay->addLayout(col);
    }

    // Epochs
    m_epochsSpin = new QSpinBox;
    m_epochsSpin->setRange(1, 10000);
    m_epochsSpin->setValue(300);
    m_epochsSpin->setFixedWidth(75);
    addCol("Epochs", m_epochsSpin);

    // Batch / Population (label changes dynamically)
    {
        auto* col = new QVBoxLayout;
        m_batchLbl = new QLabel("Batch");
        m_batchLbl->setStyleSheet("color:#E8E8F0;font-size:11px;font-weight:600;");
        col->addWidget(m_batchLbl);
        m_batchSpin = new QSpinBox;
        m_batchSpin->setRange(1, 2000); m_batchSpin->setValue(32); m_batchSpin->setFixedWidth(65);
        col->addWidget(m_batchSpin);
        cfgLay->addLayout(col);
    }

    // Val Split
    m_valSpin = new QDoubleSpinBox;
    m_valSpin->setRange(0.0, 0.5); m_valSpin->setValue(0.20);
    m_valSpin->setDecimals(2); m_valSpin->setSingleStep(0.05); m_valSpin->setFixedWidth(65);
    addCol("Val Split", m_valSpin);

    // Early stopping patience
    m_earlySpin = new QSpinBox;
    m_earlySpin->setRange(0, 500); m_earlySpin->setValue(20); m_earlySpin->setFixedWidth(65);
    m_earlySpin->setToolTip("Early stopping patience (0 = disabled).");
    addCol("Early Stop", m_earlySpin);

    // K-folds for cross-validation
    m_kfoldSpin = new QSpinBox;
    m_kfoldSpin->setRange(2, 20); m_kfoldSpin->setValue(5); m_kfoldSpin->setFixedWidth(60);
    m_kfoldSpin->setToolTip("Number of folds for K-fold cross-validation.");
    addCol("K-Folds",  m_kfoldSpin);

    // Weight bound (evolutionary-only)
    {
        auto* col = new QVBoxLayout;
        m_wBoundLbl = new QLabel("Weight Bound");
        m_wBoundLbl->setStyleSheet("color:#E8E8F0;font-size:11px;font-weight:600;");
        col->addWidget(m_wBoundLbl);
        m_wBoundSpin = new QDoubleSpinBox;
        m_wBoundSpin->setRange(0.5, 50.0); m_wBoundSpin->setValue(5.0);
        m_wBoundSpin->setDecimals(1); m_wBoundSpin->setSingleStep(0.5); m_wBoundSpin->setFixedWidth(70);
        m_wBoundSpin->setToolTip(
            "Search space bound: weights are restricted to [-b, +b].\n"
            "Default 5.0 works well for most networks.");
        col->addWidget(m_wBoundSpin);
        cfgLay->addLayout(col);
        m_wBoundLbl->hide(); m_wBoundSpin->hide(); // shown only for evolutionary
    }

    cfgLay->addStretch();
    root->addWidget(cfgGroup);

    connect(m_optimCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &TrainingPanel::onOptimizerChanged);

    // ── Buttons row ──────────────────────────────────────────────────────────
    m_trainBtn = new QPushButton("\u25B6  Train");
    m_trainBtn->setFixedHeight(36);
    m_trainBtn->setMinimumWidth(110);
    m_trainBtn->setEnabled(false);
    m_trainBtn->setStyleSheet(
        "QPushButton:enabled{background:#1D9E75;color:white;border-radius:5px;font-weight:600;font-size:13px;padding:0 16px;}"
        "QPushButton:disabled{background:palette(midlight);border-radius:5px;}");

    m_cvBtn = new QPushButton("K-Fold CV");
    m_cvBtn->setFixedHeight(36);
    m_cvBtn->setMinimumWidth(110);
    m_cvBtn->setEnabled(false);
    m_cvBtn->setStyleSheet(
        "QPushButton:enabled{background:#185fa5;color:white;border-radius:5px;font-weight:600;padding:0 14px;}"
        "QPushButton:disabled{background:palette(midlight);border-radius:5px;}");
    m_cvBtn->setToolTip(
        "Run K-fold cross-validation: trains K independent models to assess "
        "how stable performance is across data splits.");

    m_exportLogBtn = new QPushButton("Export Log CSV");
    m_exportLogBtn->setFixedHeight(36);
    m_exportLogBtn->setMinimumWidth(140);
    m_exportLogBtn->setEnabled(false);
    m_exportLogBtn->setStyleSheet(
        "QPushButton:enabled{border:1px solid palette(mid);border-radius:5px;padding:0 14px;}"
        "QPushButton:disabled{color:palette(mid);border:1px solid palette(midlight);border-radius:5px;padding:0 14px;}");

    m_statusLbl = new QLabel("Build a network first.");
    m_statusLbl->setStyleSheet("font-size:12px;color:#C8C8D8;");

    auto* ctrlRow = new QHBoxLayout;
    ctrlRow->addWidget(m_trainBtn);
    ctrlRow->addWidget(m_cvBtn);
    ctrlRow->addWidget(m_exportLogBtn);
    ctrlRow->addWidget(m_statusLbl, 1);
    root->addLayout(ctrlRow);

    m_progress = new QProgressBar;
    m_progress->setRange(0, 100); m_progress->setValue(0);
    m_progress->setFixedHeight(14); m_progress->hide();
    root->addWidget(m_progress);

    // ── CV result label ──────────────────────────────────────────────────────
    m_cvResultLbl = new QLabel;
    m_cvResultLbl->setWordWrap(true);
    m_cvResultLbl->setStyleSheet(
        "QLabel{background:#185fa5;color:white;border-radius:4px;"
        "padding:8px 12px;font-family:monospace;font-size:12px;}");
    m_cvResultLbl->hide();
    root->addWidget(m_cvResultLbl);

    // ── Charts (Loss + Accuracy tabs) ────────────────────────────────────────
    m_chartTabs = new QTabWidget;

    // Loss chart
    m_trainSeries = new QLineSeries; m_trainSeries->setName("Train Loss"); m_trainSeries->setColor(QColor("#1D9E75"));
    m_valSeries   = new QLineSeries; m_valSeries->setName("Validation Loss"); m_valSeries->setColor(QColor("#E8700A"));
    m_lossChart   = new QChart;
    m_lossChart->addSeries(m_trainSeries);
    m_lossChart->addSeries(m_valSeries);
    m_lossChart->setTitle("Loss over Epochs");
    styleChart(m_lossChart);

    m_lossAxisX = new QValueAxis;
    m_lossAxisX->setTitleText("Epoch");
    m_lossAxisX->setLabelFormat("%d");
    m_lossAxisX->setLabelsBrush(QBrush(QColor("#B8B8C8")));
    m_lossAxisX->setTitleBrush(QBrush(QColor("#E4E4EC")));
    m_lossAxisX->setGridLineColor(QColor("#3a3a4a"));
    m_lossChart->addAxis(m_lossAxisX, Qt::AlignBottom);
    m_trainSeries->attachAxis(m_lossAxisX); m_valSeries->attachAxis(m_lossAxisX);

    m_lossAxisY = new QValueAxis;
    m_lossAxisY->setTitleText("Loss");
    m_lossAxisY->setMin(0.0);
    m_lossAxisY->setLabelsBrush(QBrush(QColor("#B8B8C8")));
    m_lossAxisY->setTitleBrush(QBrush(QColor("#E4E4EC")));
    m_lossAxisY->setGridLineColor(QColor("#3a3a4a"));
    m_lossChart->addAxis(m_lossAxisY, Qt::AlignLeft);
    m_trainSeries->attachAxis(m_lossAxisY); m_valSeries->attachAxis(m_lossAxisY);

    m_lossView = new QChartView(m_lossChart);
    m_lossView->setRenderHint(QPainter::Antialiasing);
    m_chartTabs->addTab(m_lossView, "Loss");

    // Accuracy chart
    m_trainAccSer = new QLineSeries; m_trainAccSer->setName("Train Accuracy");      m_trainAccSer->setColor(QColor("#1D9E75"));
    m_valAccSer   = new QLineSeries; m_valAccSer->setName("Validation Accuracy");   m_valAccSer->setColor(QColor("#E8700A"));
    m_accChart    = new QChart;
    m_accChart->addSeries(m_trainAccSer);
    m_accChart->addSeries(m_valAccSer);
    m_accChart->setTitle("Accuracy over Epochs");
    styleChart(m_accChart);

    m_accAxisX = new QValueAxis;
    m_accAxisX->setTitleText("Epoch"); m_accAxisX->setLabelFormat("%d");
    m_accAxisX->setLabelsBrush(QBrush(QColor("#B8B8C8")));
    m_accAxisX->setTitleBrush(QBrush(QColor("#E4E4EC")));
    m_accAxisX->setGridLineColor(QColor("#3a3a4a"));
    m_accChart->addAxis(m_accAxisX, Qt::AlignBottom);
    m_trainAccSer->attachAxis(m_accAxisX); m_valAccSer->attachAxis(m_accAxisX);

    m_accAxisY = new QValueAxis;
    m_accAxisY->setTitleText("Accuracy");
    m_accAxisY->setRange(0.0, 1.0); m_accAxisY->setLabelFormat("%.2f");
    m_accAxisY->setLabelsBrush(QBrush(QColor("#B8B8C8")));
    m_accAxisY->setTitleBrush(QBrush(QColor("#E4E4EC")));
    m_accAxisY->setGridLineColor(QColor("#3a3a4a"));
    m_accChart->addAxis(m_accAxisY, Qt::AlignLeft);
    m_trainAccSer->attachAxis(m_accAxisY); m_valAccSer->attachAxis(m_accAxisY);

    m_accView = new QChartView(m_accChart);
    m_accView->setRenderHint(QPainter::Antialiasing);
    m_chartTabs->addTab(m_accView, "Accuracy");

    // ── RBF summary label (shown instead of chart info after RBF training) ────
    m_rbfSummaryLbl = new QLabel;
    m_rbfSummaryLbl->setWordWrap(true);
    m_rbfSummaryLbl->setAlignment(Qt::AlignCenter);
    m_rbfSummaryLbl->setTextFormat(Qt::RichText);
    m_rbfSummaryLbl->setMinimumHeight(120);
    m_rbfSummaryLbl->setStyleSheet(
        "QLabel{background:#0D2518;border-radius:8px;padding:20px;"
        "color:#4EDDAD;font-size:13px;}");
    m_rbfSummaryLbl->hide();
    root->addWidget(m_rbfSummaryLbl);
    root->addWidget(m_chartTabs, 1);

    connect(m_trainBtn,     &QPushButton::clicked, this, &TrainingPanel::onTrainClicked);
    connect(m_cvBtn,        &QPushButton::clicked, this, &TrainingPanel::onCVClicked);
    connect(m_exportLogBtn, &QPushButton::clicked, this, &TrainingPanel::onExportLogClicked);
}

bool TrainingPanel::isGradientDescent() const {
    const QString v = m_optimCombo->currentData().toString().toLower();
    static const QSet<QString> gd{"adam","sgd","adamw","rmsprop","nadam","adagrad"};
    return gd.contains(v);
}

bool TrainingPanel::isEvolutionary() const {
    const QString v = m_optimCombo->currentData().toString().toLower();
    static const QSet<QString> evo{
        "ga","de","pso","clpso","acor","cmaes","lmcmaes","jso","mlshaderl","arq3","pbil","umda"
    };
    return evo.contains(v);
}

void TrainingPanel::onOptimizerChanged(int /*index*/) {
    const bool gd  = isGradientDescent();
    const bool evo = isEvolutionary();

    // LR + schedule: only for gradient descent
    m_lrSpin      ->setVisible(gd);  m_lrLbl   ->setVisible(gd);
    m_lrSchedCombo->setVisible(gd);  m_schedLbl->setVisible(gd);

    // Weight bound: for all non-gradient methods
    m_wBoundSpin->setVisible(!gd);  m_wBoundLbl->setVisible(!gd);

    // Batch → Population (only for population-based methods)
    m_batchLbl->setText(evo ? "Population" : "Batch");
    if (evo && m_batchSpin->value() < 10)
        m_batchSpin->setValue(30);
}

void TrainingPanel::onDatasetLoaded(const Dataset* ds) { m_ds = ds; }

void TrainingPanel::onRBFBuilt(std::shared_ptr<RBFNetwork> net) {
    m_rbfNet = net;
    m_net.reset();
    resetCharts();          // clear any leftover MLP epoch data
    onOptimizerChanged(0);

    const auto& cfg = net->config();
    const QString kernel =
        cfg.kernel == RBFKernel::Gaussian     ? "Gaussian"    :
        cfg.kernel == RBFKernel::Multiquadric ? "Multiquadric" : "Inv. Multiquadric";
    m_rbfSummaryLbl->setText(
        QString("<b style='color:#D4A842;font-size:15px;'>"
                "\u23F3  RBF Network ready \u2014 click \u25B6 Train</b><br><br>"
                "<span style='color:#88A8C0;font-size:12px;'>"
                "%1 inputs &rarr; k=%2 centres &rarr; %3 output(s) &nbsp;|&nbsp; "
                "Kernel: %4 &nbsp;|&nbsp; Ridge &lambda;: %5</span>")
            .arg(net->nInputs()).arg(cfg.nCenters).arg(net->nOutputs())
            .arg(kernel).arg(cfg.ridgeLambda));
    m_rbfSummaryLbl->show();
    m_statusLbl->setText("RBF ready \u2014 click Train to run K-means + least squares.");
    m_trainBtn->setEnabled(true);
}

void TrainingPanel::onNetworkBuilt(std::shared_ptr<NeuralNetwork> net) {
    m_net = net;
    m_rbfNet.reset();
    m_rbfSummaryLbl->hide();   // hide RBF summary when switching to MLP
    m_rbfSummaryLbl->setText("");
    m_trainBtn->setEnabled(true);
    m_cvBtn->setEnabled(true);
    m_statusLbl->setText("Network ready \u2014 click Train to start training.");
    resetCharts();
}

void TrainingPanel::resetCharts() {
    m_trainSeries->clear(); m_valSeries->clear();
    m_trainAccSer->clear(); m_valAccSer->clear();
    m_lossAxisX->setRange(0, m_epochsSpin->value()); m_lossAxisY->setRange(0, 1.0);
    m_accAxisX ->setRange(0, m_epochsSpin->value()); m_accAxisY ->setRange(0, 1.0);
    m_history.clear();
    m_exportLogBtn->setEnabled(false);
    m_cvResultLbl->hide();
}

void TrainingPanel::onTrainClicked() {
    if (m_running) { if (m_trainer) m_trainer->requestStop();
                     if (m_metaTrainer) m_metaTrainer->requestStop(); return; }

    // ── RBF one-shot training (synchronous — analytic method, no gradient loop) ──
    if (m_rbfNet && !m_net) {
        if (!m_ds) {
            m_statusLbl->setText("No dataset loaded.");
            return;
        }

        // Disable button and show status before running
        m_trainBtn->setEnabled(false);
        m_statusLbl->setText("RBF training — K-means++ centre selection…");
        QApplication::processEvents();   // paint the status label

        m_rbfNet->config().valSplit = m_valSpin->value();

        // Create trainer as a plain local object (no thread needed)
        RBFTrainer rbfTrainer;
        rbfTrainer.setNetwork(m_rbfNet);
        rbfTrainer.setDataset(m_ds);

        // Capture results via direct lambda connections (same thread = direct)
        double trE = 0, valE = 0, trA = -1, valA = -1;
        bool   hasError = false;
        QString errorMsg;

        connect(&rbfTrainer, &RBFTrainer::progressMessage, this,
                [&](QString msg){
            m_statusLbl->setText(msg);
            QApplication::processEvents();   // keep UI alive between phases
        });
        connect(&rbfTrainer, &RBFTrainer::trainingCompleted, this,
                [&](double te, double ve, double ta, double va){
            trE = te; valE = ve; trA = ta; valA = va;
        });
        connect(&rbfTrainer, &RBFTrainer::trainingError, this,
                [&](QString err){ hasError = true; errorMsg = err; });

        rbfTrainer.run();   // ← runs synchronously; signals fire directly

        m_trainBtn->setEnabled(true);

        if (hasError) {
            m_rbfSummaryLbl->setText(
                "<b style='color:#FF6B6B;font-size:15px;'>\u274C  RBF Training Failed</b><br>"
                "<span style='color:#C08080;'>" + errorMsg + "</span>");
            m_statusLbl->setText("RBF training failed: " + errorMsg);
            return;
        }

        // Update summary label with training results
        const auto& cfg = m_rbfNet->config();
        const QString kernel =
            cfg.kernel == RBFKernel::Gaussian     ? "Gaussian"    :
            cfg.kernel == RBFKernel::Multiquadric ? "Multiquadric" : "Inv. Multiquadric";
        m_rbfSummaryLbl->setText(
            QString("<b style='color:#4EDDAD;font-size:15px;'>"
                    "\u2713  RBF Network Trained Successfully</b><br><br>"
                    "<span style='color:#88A8C0;font-size:12px;'>"
                    "%1 inputs &rarr; k=%2 centres &rarr; %3 output(s) &nbsp;|&nbsp; %4 kernel</span><br><br>"
                    "<span style='font-size:13px;'>"
                    "<b style='color:#80D0A0;'>Train MSE:</b> "
                    "<span style='color:#4EDDAD;font-size:16px;font-weight:700;'>%5</span>"
                    "&nbsp;&nbsp;&nbsp;"
                    "<b style='color:#80D0A0;'>Val MSE:</b> "
                    "<span style='color:#4EDDAD;font-size:16px;font-weight:700;'>%6</span>"
                    "</span><br><br>"
                    "<span style='color:#405060;font-size:11px;font-style:italic;'>"
                    "\u2139 Analytic training: K-means++ &rarr; width heuristic &rarr; "
                    "ridge regression.  No epochs.</span>")
                .arg(m_rbfNet->nInputs()).arg(m_rbfNet->nCenters()).arg(m_rbfNet->nOutputs())
                .arg(kernel)
                .arg(trE, 0,'f',5).arg(valE, 0,'f',5));

        QString msg = QString("RBF complete — Train MSE: %1  |  Val MSE: %2")
                          .arg(trE, 0,'f',5).arg(valE, 0,'f',5);
        if (trA >= 0)
            msg += QString("  |  Train Acc: %1%  |  Val Acc: %2%")
                       .arg(trA*100, 0,'f',1).arg(valA*100, 0,'f',1);
        m_statusLbl->setText(msg);
        emit rbfTrainingCompleted(m_rbfNet, trE, valE);
        return;
    }

    if (!m_net || !m_ds) return;

    setRunning(true);
    resetCharts();

    TrainerConfig cfg;
    cfg.epochs                = m_epochsSpin->value();
    cfg.batchSize             = m_batchSpin->value();
    cfg.validationSplit       = m_valSpin->value();
    cfg.earlyStoppingPatience = m_earlySpin->value();

    const QString optimName = m_optimCombo->currentData().toString();

    if (isGradientDescent()) {
        // ── Gradient-based (Adam / SGD / AdamW / RMSProp / Nadam / AdaGrad) ──
        const QString n = optimName.toLower();
        if      (n == "sgd")     cfg.optimizer = OptimizerType::SGD;
        else if (n == "adamw")   cfg.optimizer = OptimizerType::AdamW;
        else if (n == "rmsprop") cfg.optimizer = OptimizerType::RMSProp;
        else if (n == "nadam")   cfg.optimizer = OptimizerType::Nadam;
        else if (n == "adagrad") cfg.optimizer = OptimizerType::AdaGrad;
        else                     cfg.optimizer = OptimizerType::Adam;
        cfg.learningRate = m_lrSpin->value();
        cfg.lrSchedule   = static_cast<LRSchedule>(m_lrSchedCombo->currentData().toInt());

        m_lossAxisX->setRange(0, cfg.epochs);
        m_accAxisX ->setRange(0, cfg.epochs);
        m_progress->setRange(0, cfg.epochs);
        m_progress->setValue(0); m_progress->show();

        m_thread  = new QThread(this);
        m_trainer = new Trainer;
        m_trainer->setConfig(cfg);
        m_trainer->setNetwork(m_net.get());
        m_trainer->setDataset(m_ds);
        m_trainer->moveToThread(m_thread);

        connect(m_thread,  &QThread::started,          m_trainer, &Trainer::run);
        connect(m_trainer, &Trainer::epochCompleted,   this, &TrainingPanel::onEpochCompleted,    Qt::QueuedConnection);
        connect(m_trainer, &Trainer::trainingFinished, this, &TrainingPanel::onTrainingFinished,  Qt::QueuedConnection);
        connect(m_trainer, &Trainer::trainingError,    this, &TrainingPanel::onTrainingError,     Qt::QueuedConnection);
        connect(m_trainer, &Trainer::trainingFinished, m_thread, &QThread::quit);
        connect(m_trainer, &Trainer::trainingError,    m_thread, &QThread::quit);
        connect(m_thread,  &QThread::finished,         m_trainer, &QObject::deleteLater);
        connect(this, &TrainingPanel::requestStop,     m_trainer, &Trainer::requestStop, Qt::QueuedConnection);
        m_thread->start();

    } else {
        // ── Evolutionary / Swarm ──────────────────────────────────────────────
        // For evolutionary methods: epochs = iterations, batchSize = population.
        // The total "budget" = iterations × population evaluations.
        m_lossAxisX->setRange(0, cfg.epochs);
        m_accAxisX ->setRange(0, cfg.epochs);
        m_progress->setRange(0, cfg.epochs);
        m_progress->setValue(0); m_progress->show();

        m_thread      = new QThread(this);
        m_metaTrainer = new MetaTrainer;
        m_metaTrainer->setConfig(cfg);
        m_metaTrainer->setOptimizerName(optimName);
        m_metaTrainer->setNetwork(m_net.get());
        m_metaTrainer->setDataset(m_ds);
        m_metaTrainer->setWeightBound(m_wBoundSpin->value());
        m_metaTrainer->moveToThread(m_thread);

        connect(m_thread,      &QThread::started,            m_metaTrainer, &MetaTrainer::run);
        connect(m_metaTrainer, &MetaTrainer::epochCompleted,   this, &TrainingPanel::onEpochCompleted,    Qt::QueuedConnection);
        connect(m_metaTrainer, &MetaTrainer::trainingFinished, this, &TrainingPanel::onTrainingFinished,  Qt::QueuedConnection);
        connect(m_metaTrainer, &MetaTrainer::trainingError,    this, &TrainingPanel::onTrainingError,     Qt::QueuedConnection);
        connect(m_metaTrainer, &MetaTrainer::trainingFinished, m_thread, &QThread::quit);
        connect(m_metaTrainer, &MetaTrainer::trainingError,    m_thread, &QThread::quit);
        connect(m_thread,      &QThread::finished,             m_metaTrainer, &QObject::deleteLater);
        connect(this, &TrainingPanel::requestStop, m_metaTrainer, &MetaTrainer::requestStop, Qt::QueuedConnection);
        m_thread->start();
    }
}

void TrainingPanel::onEpochCompleted(EpochResult r) {
    m_history.push_back(r);
    m_trainSeries->append(r.epoch, r.trainLoss);
    if (r.valLoss >= 0.0) m_valSeries->append(r.epoch, r.valLoss);

    // Auto-scale Y axis based on actual data
    double maxL = r.trainLoss;
    if (r.valLoss > 0.0) maxL = std::max(maxL, r.valLoss);

    if (r.epoch == 1) {
        // Initial range based on the first epoch's loss — no wasted empty space
        m_lossAxisY->setRange(0.0, maxL * 1.3);
    } else if (maxL * 1.05 > m_lossAxisY->max()) {
        m_lossAxisY->setMax(maxL * 1.15);
    }

    if (r.trainAcc >= 0.0) {
        m_trainAccSer->append(r.epoch, r.trainAcc);
        if (r.valAcc >= 0.0) m_valAccSer->append(r.epoch, r.valAcc);
    }

    m_progress->setValue(r.epoch);

    QString status;
    if (r.trainAcc >= 0.0) {
        status = QString("Epoch %1  |  Loss: %2  |  Val: %3  |  Acc: %4%  |  ValAcc: %5  |  LR: %6")
            .arg(r.epoch)
            .arg(r.trainLoss, 0,'f',4)
            .arg(r.valLoss  >= 0 ? QString::number(r.valLoss, 'f',4) : "—")
            .arg(r.trainAcc * 100.0, 0,'f',1)
            .arg(r.valAcc   >= 0 ? QString::number(r.valAcc*100.0,'f',1) + "%" : "—")
            .arg(r.currentLR, 0, 'e', 2);
    } else {
        status = QString("Epoch %1  |  Train: %2  |  Val: %3  |  LR: %4")
            .arg(r.epoch)
            .arg(r.trainLoss, 0,'f',5)
            .arg(r.valLoss  >= 0 ? QString::number(r.valLoss, 'f',5) : "—")
            .arg(r.currentLR, 0, 'e', 2);
    }
    m_statusLbl->setText(status);
}

void TrainingPanel::onTrainingFinished(double finalVal) {
    setRunning(false);
    // Shrink X axis to actual epochs run (e.g. when early-stopped)
    if (!m_history.empty()) {
        const int lastEp = m_history.back().epoch;
        m_lossAxisX->setRange(0, lastEp);
        m_accAxisX ->setRange(0, lastEp);
    }
    m_statusLbl->setText(QString("Training complete.  Final val loss: %1  (%2 epochs)")
        .arg(finalVal,0,'f',5).arg(m_history.size()));
    m_exportLogBtn->setEnabled(!m_history.empty());
    emit trainingCompleted();
}

void TrainingPanel::onTrainingError(QString msg) {
    setRunning(false);
    QMessageBox::critical(this, "Training Error", msg);
    m_statusLbl->setText("Error: " + msg);
}

// ─── K-Fold Cross-Validation ──────────────────────────────────────────────────
void TrainingPanel::onCVClicked() {
    if (!m_net || !m_ds) return;
    if (m_running) return;

    // We need access to NetworkConfig — but the network is already built so
    // we infer its structure from the existing layers.
    NetworkConfig netCfg;
    netCfg.task = m_net->taskType();
    auto& layers = m_net->layers();
    // Hidden = all layers except the last
    for (int i = 0; i < static_cast<int>(layers.size()) - 1; ++i) {
        netCfg.hiddenSizes.push_back(layers[i].outputSize());
        netCfg.hiddenActivations.push_back(layers[i].activation());
        netCfg.dropoutRate = layers[i].dropoutRate(); // last value wins
    }

    TrainerConfig trCfg;
    trCfg.optimizer       = m_optimCombo->currentIndex()==0 ? OptimizerType::Adam : OptimizerType::SGD;
    trCfg.learningRate    = m_lrSpin->value();
    trCfg.lrSchedule      = static_cast<LRSchedule>(m_lrSchedCombo->currentData().toInt());
    trCfg.epochs          = m_epochsSpin->value();
    trCfg.batchSize       = m_batchSpin->value();

    const int k = m_kfoldSpin->value();
    setRunning(true);
    m_statusLbl->setText(QString("Running %1-fold CV...").arg(k));
    m_progress->setRange(0, k);
    m_progress->setValue(0);
    m_progress->show();
    m_cvResultLbl->hide();

    m_cvThread = new QThread(this);
    m_cv       = new CrossValidator;
    m_cv->setK(k);
    m_cv->setDataset(m_ds);
    m_cv->setNetworkConfig(netCfg);
    m_cv->setTrainerConfig(trCfg);
    m_cv->moveToThread(m_cvThread);

    connect(m_cvThread, &QThread::started,             m_cv,       &CrossValidator::run);
    connect(m_cv,       &CrossValidator::foldCompleted, this,      &TrainingPanel::onFoldCompleted, Qt::QueuedConnection);
    connect(m_cv,       &CrossValidator::cvFinished,    this,      &TrainingPanel::onCVFinished,    Qt::QueuedConnection);
    connect(m_cv,       &CrossValidator::cvError,       this,      &TrainingPanel::onCVError,       Qt::QueuedConnection);
    connect(m_cv,       &CrossValidator::cvFinished,    m_cvThread,&QThread::quit);
    connect(m_cv,       &CrossValidator::cvError,       m_cvThread,&QThread::quit);
    connect(m_cvThread, &QThread::finished,             m_cv,      &QObject::deleteLater);

    m_cvThread->start();
}

void TrainingPanel::onFoldCompleted(FoldResult fr) {
    m_progress->setValue(fr.fold);
    QString acc = fr.finalAcc >= 0
        ? QString(" | acc %1%").arg(fr.finalAcc * 100.0, 0, 'f', 1) : "";
    m_statusLbl->setText(QString("CV — fold %1 done: loss %2%3")
        .arg(fr.fold).arg(fr.finalLoss, 0,'f',4).arg(acc));
}

void TrainingPanel::onCVFinished(CVSummary s) {
    setRunning(false);
    QString accStr = s.meanAcc >= 0
        ? QString("Mean Accuracy: <b>%1%</b> &plusmn; %2%<br>")
            .arg(s.meanAcc * 100.0, 0, 'f', 2).arg(s.stdAcc * 100.0, 0, 'f', 2)
        : "";
    QString html = QString(
        "<b>%1-Fold Cross-Validation Results</b><br>"
        "Mean Loss: <b>%2</b> &plusmn; %3<br>%4"
        "Folds: ")
        .arg(s.k).arg(s.meanLoss, 0, 'f', 5).arg(s.stdLoss, 0, 'f', 5).arg(accStr);
    QStringList parts;
    for (const auto& f : s.folds)
        parts << QString("[%1: %2]").arg(f.fold).arg(f.finalLoss, 0,'f',4);
    html += parts.join(" ");
    m_cvResultLbl->setText(html);
    m_cvResultLbl->show();
    m_statusLbl->setText("Cross-validation complete.");
}

void TrainingPanel::onCVError(QString msg) {
    setRunning(false);
    QMessageBox::critical(this, "Cross-Validation Error", msg);
    m_statusLbl->setText("CV Error: " + msg);
}

// ─── Export log to CSV ───────────────────────────────────────────────────────
void TrainingPanel::onExportLogClicked() {
    if (m_history.empty()) return;
    const QString path = QFileDialog::getSaveFileName(
        this, "Export Training Log", "training_log.csv",
        "CSV Files (*.csv);;All files (*)");
    if (path.isEmpty()) return;

    QFile f(path);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::critical(this, "Export Error", "Cannot write file: " + path);
        return;
    }
    QTextStream out(&f);
    out << "epoch,train_loss,val_loss,train_acc,val_acc,learning_rate\n";
    for (const auto& r : m_history) {
        out << r.epoch << ','
            << r.trainLoss << ','
            << (r.valLoss   >= 0 ? QString::number(r.valLoss,   'g', 8) : "") << ','
            << (r.trainAcc  >= 0 ? QString::number(r.trainAcc,  'g', 8) : "") << ','
            << (r.valAcc    >= 0 ? QString::number(r.valAcc,    'g', 8) : "") << ','
            << r.currentLR << '\n';
    }
    m_statusLbl->setText("Log exported: " + path);
}

void TrainingPanel::setRunning(bool r) {
    m_running = r;
    m_trainBtn->setText(r ? "\u25A0  Stop" : "\u25B6  Train");
    m_trainBtn->setStyleSheet(r
        ? "QPushButton{background:#C0392B;color:white;border-radius:5px;font-weight:600;font-size:13px;padding:0 16px;}"
        : "QPushButton:enabled{background:#1D9E75;color:white;border-radius:5px;font-weight:600;font-size:13px;padding:0 16px;}"
          "QPushButton:disabled{background:palette(midlight);border-radius:5px;}");
    for (auto* w : {(QWidget*)m_optimCombo,(QWidget*)m_lrSpin,(QWidget*)m_lrSchedCombo,
                    (QWidget*)m_epochsSpin,(QWidget*)m_batchSpin,(QWidget*)m_valSpin,
                    (QWidget*)m_earlySpin,(QWidget*)m_kfoldSpin,(QWidget*)m_wBoundSpin,
                    (QWidget*)m_cvBtn})
        w->setEnabled(!r);
    if (!r) m_progress->hide();
}

} // namespace NeuralStudio
