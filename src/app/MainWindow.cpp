#include "MainWindow.h"
#include "ui/panels/DataPanel.h"
#include "ui/panels/NetworkPanel.h"
#include "ui/panels/TrainingPanel.h"
#include "ui/panels/EvaluationPanel.h"
#include "ui/panels/PredictionPanel.h"
#include "ui/panels/BatchPanel.h"
#include "ui/panels/PreTrainNNCPanel.h"
#include "ui/panels/EvolBoundsPanel.h"
#include "ui/panels/ArchSearchPanel.h"
#include "ui/panels/GEStudioPanel.h"
#include "core/dataset/DatasetLoader.h"
#include "core/nn/NeuralNetwork.h"
#include "core/nn/ModelSerializer.h"
#include "core/export/CppExporter.h"
#include "core/export/OnnxExporter.h"

#include <QSplitter>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QMenuBar>
#include <QMenu>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QFile>
#include <QFileInfo>
#include <QAction>
#include <QMessageBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QApplication>
#include <QSettings>
#include <QStatusBar>
#include <QScrollArea>

namespace NeuralStudio {

static constexpr int kMaxRecent = 8;

// Stack widget indices
//   0  DataPanel          ┐
//   1  NetworkPanel       │  Mode 1: Single Workflow
//   2  TrainingPanel      │
//   3  EvaluationPanel    │
//   4  PredictionPanel    ┘
//   5  BatchPanel              Mode 2: Batch Run
//   6  PreTrainNNCPanel        Mode 3: Pre-Train + NNC
//   7  EvolBoundsPanel         Mode 4: Evol. Bounds
//   8  ArchSearchPanel         Mode 5: Architecture Search
//   9  GEStudioPanel           Mode 6: Grammatical Evolution Studio

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowTitle("NeuralStudio");
    setWindowIcon(QIcon(":/icons/neuralstudio.svg"));
    setMinimumSize(1080, 700);
    resize(1360, 840);
    buildUi();
    buildMenuBar();
    statusBar()->showMessage("Ready");
}

// ─── buildUi ─────────────────────────────────────────────────────────────────
void MainWindow::buildUi() {
    auto* splitter = new QSplitter(Qt::Horizontal, this);
    splitter->setHandleWidth(1);
    setCentralWidget(splitter);
    splitter->addWidget(buildSidebar());
    splitter->setCollapsible(0, false);

    m_stack = new QStackedWidget;
    m_dataPanel       = new DataPanel;
    m_networkPanel    = new NetworkPanel;
    m_trainingPanel   = new TrainingPanel;
    m_evaluationPanel = new EvaluationPanel;
    m_predictionPanel = new PredictionPanel;
    m_batchPanel      = new BatchPanel;
    m_preTrainPanel   = new PreTrainNNCPanel;
    m_evolBoundsPanel = new EvolBoundsPanel;
    m_archSearchPanel = new ArchSearchPanel;
    m_geStudioPanel   = new GEStudioPanel;

    // Wrap the panels that may exceed the sidebar height in a scroll area
    auto wrap = [](QWidget* w) -> QWidget* {
        auto* s = new QScrollArea;
        s->setWidget(w); s->setWidgetResizable(true);
        s->setFrameShape(QFrame::NoFrame);
        return s;
    };

    m_stack->addWidget(m_dataPanel);             // 0
    m_stack->addWidget(m_networkPanel);          // 1
    m_stack->addWidget(m_trainingPanel);         // 2
    m_stack->addWidget(m_evaluationPanel);       // 3
    m_stack->addWidget(m_predictionPanel);       // 4
    m_stack->addWidget(m_batchPanel);            // 5
    m_stack->addWidget(wrap(m_preTrainPanel));   // 6
    m_stack->addWidget(wrap(m_evolBoundsPanel)); // 7
    m_stack->addWidget(wrap(m_archSearchPanel)); // 8
    m_stack->addWidget(wrap(m_geStudioPanel));   // 9

    splitter->addWidget(m_stack);
    splitter->setCollapsible(1, false);
    splitter->setSizes({230, 1130});

    // ── Workflow connections ──────────────────────────────────────────────
    connect(m_dataPanel, &DataPanel::datasetLoaded, this, &MainWindow::onDatasetLoaded);
    connect(m_dataPanel, &DataPanel::datasetLoaded, m_networkPanel,    &NetworkPanel::onDatasetLoaded);
    connect(m_dataPanel, &DataPanel::datasetLoaded, m_trainingPanel,   &TrainingPanel::onDatasetLoaded);
    connect(m_dataPanel, &DataPanel::testDatasetLoaded,
            m_trainingPanel, &TrainingPanel::onTestDatasetLoaded);
    connect(m_dataPanel, &DataPanel::datasetLoaded, m_evaluationPanel, &EvaluationPanel::onDatasetLoaded);
    connect(m_dataPanel, &DataPanel::datasetLoaded, m_predictionPanel, &PredictionPanel::onDatasetLoaded);
    connect(m_dataPanel, &DataPanel::testDatasetLoaded,
            m_evaluationPanel, &EvaluationPanel::onTestDatasetLoaded);

    connect(m_networkPanel, &NetworkPanel::networkBuilt, this,              &MainWindow::onNetworkBuilt);
    connect(m_networkPanel, &NetworkPanel::networkBuilt, m_trainingPanel,   &TrainingPanel::onNetworkBuilt);
    connect(m_networkPanel, &NetworkPanel::networkBuilt, m_evaluationPanel, &EvaluationPanel::onNetworkBuilt);
    connect(m_networkPanel, &NetworkPanel::networkBuilt, m_predictionPanel, &PredictionPanel::onNetworkBuilt);
    connect(m_networkPanel, &NetworkPanel::rbfBuilt,     m_trainingPanel,   &TrainingPanel::onRBFBuilt);
    connect(m_networkPanel, &NetworkPanel::rbfBuilt,     this,              &MainWindow::onRBFNetworkBuilt);
    connect(m_trainingPanel, &TrainingPanel::rbfTrainingCompleted, this, &MainWindow::onRBFTrainingCompleted);
    connect(m_trainingPanel, &TrainingPanel::trainingCompleted, this, &MainWindow::onTrainingCompleted);
    connect(m_trainingPanel, &TrainingPanel::trainingCompleted, m_evaluationPanel, &EvaluationPanel::onTrainingCompleted);
    connect(m_trainingPanel, &TrainingPanel::trainingCompleted, m_predictionPanel, &PredictionPanel::onTrainingCompleted);

    // ── Mode 2: Batch Run — lock navigation while a batch is executing ────
    connect(m_batchPanel, &BatchPanel::runningStateChanged,
            this, &MainWindow::lockSidebar);

    // ── Mode 3: Pre-Train + NNC — self-contained, NO navigation change ───
    connect(m_preTrainPanel, &PreTrainNNCPanel::runningStateChanged,
            this, &MainWindow::lockSidebar);
    connect(m_preTrainPanel, &PreTrainNNCPanel::trainingCompleted,
            this, [this](std::shared_ptr<NeuralNetwork> net, std::shared_ptr<Dataset>) {
                m_net = net;
                // Quietly notify Evaluation/Prediction without navigating away
                m_evaluationPanel->onNetworkBuilt(net);
                m_predictionPanel->onNetworkBuilt(net);
                setSidebarEnabled(3, true);
                setSidebarEnabled(4, true);
                statusBar()->showMessage(
                    "Pre-Train + NNC complete. Network ready  |  "
                    + net->summary());
            });

    // ── Mode 4: Evol. Bounds — self-contained, NO navigation change ──────
    connect(m_evolBoundsPanel, &EvolBoundsPanel::runningStateChanged,
            this, &MainWindow::lockSidebar);
    connect(m_evolBoundsPanel, &EvolBoundsPanel::trainingCompleted,
            this, [this](std::shared_ptr<NeuralNetwork> net, std::shared_ptr<Dataset>) {
                m_net = net;
                m_evaluationPanel->onNetworkBuilt(net);
                m_predictionPanel->onNetworkBuilt(net);
                setSidebarEnabled(3, true);
                setSidebarEnabled(4, true);
                statusBar()->showMessage(
                    "Evolutionary Bounds training complete. Network ready  |  "
                    + net->summary());
            });

    connect(m_archSearchPanel, &ArchSearchPanel::runningStateChanged,
            this, &MainWindow::lockSidebar);
    connect(m_archSearchPanel, &ArchSearchPanel::applyToMode1,
            this, &MainWindow::onApplyArchToMode1);

    // ── Mode 6: GE Studio — self-contained, NO navigation change ────────────
    connect(m_geStudioPanel, &GEStudioPanel::runningStateChanged,
            this, &MainWindow::lockSidebar);
    connect(m_geStudioPanel, &GEStudioPanel::trainingCompleted,
            this, [this](std::shared_ptr<NeuralNetwork> net, std::shared_ptr<Dataset>) {
                m_net = net;
                m_evaluationPanel->onNetworkBuilt(net);
                m_predictionPanel->onNetworkBuilt(net);
                setSidebarEnabled(3, true);
                setSidebarEnabled(4, true);
                statusBar()->showMessage(
                    "GE Studio complete. Best network ready  |  " + net->summary());
            });

    connect(m_stepBtns, &QButtonGroup::idClicked, this, &MainWindow::onStepChanged);
}

// ─── onApplyArchToMode1 ──────────────────────────────────────────────────────
void MainWindow::onApplyArchToMode1(QString datasetPath,
                                     QVector<int> hiddenSizes,
                                     QVector<QString> actNames) {
    // 1. Load train + companion test dataset in Mode 1's Data panel
    if (!datasetPath.isEmpty())
        m_dataPanel->loadFile(datasetPath);

    // 2. Belt-and-suspenders: explicitly forward the test dataset to all
    //    downstream panels that need it, in case the signal chain missed it.
    //    DataPanel::testDataset() returns bundle.test.get() (or null).
    if (const Dataset* testDs = m_dataPanel->testDataset()) {
        m_evaluationPanel->onTestDatasetLoaded(testDs);
        m_trainingPanel->onTestDatasetLoaded(testDs);
    }

    // 3. Configure the Network panel with the found architecture
    m_networkPanel->applyArchitecture(hiddenSizes, actNames);

    // 4. Navigate to Mode 1 → Network tab
    onStepChanged(1);
    if (auto* btn = m_stepBtns->button(1))
        btn->setChecked(true);

    // 5. Confirm (non-blocking status in status bar rather than blocking dialog)
    QString layers;
    for (int i = 0; i < hiddenSizes.size(); ++i) {
        if (i > 0) layers += " \u2192 ";
        layers += QString("%1\u00b7%2")
                  .arg(hiddenSizes[i])
                  .arg(i < actNames.size() ? actNames[i] : "?");
    }
    const QString companion = m_dataPanel->testDataset()
        ? QString("  (%1 + test set loaded)").arg(QFileInfo(datasetPath).fileName())
        : QString("  (%1)").arg(QFileInfo(datasetPath).fileName());

    QMessageBox::information(this, "Architecture Applied to Mode 1",
        QString(
            "\u2713 Dataset and architecture configured in Mode 1:\n\n"
            "  Architecture: %1\n"
            "  Dataset: %2\n\n"
            "Next steps:\n"
            "  1. Click \u2018Build Network\u2019 in the Network tab.\n"
            "  2. Proceed to \u25b6 Training.\n"
            "  3. Evaluate in \u25b6 Evaluation (Train Set + Test Set tabs).")
        .arg(layers)
        .arg(QFileInfo(datasetPath).fileName() + companion));
}

// ─── buildSidebar ─────────────────────────────────────────────────────────────
QWidget* MainWindow::buildSidebar() {
    m_sidebar = new QWidget;
    m_sidebar->setFixedWidth(230);
    m_sidebar->setStyleSheet("background: palette(window);");

    auto* outer = new QVBoxLayout(m_sidebar);
    outer->setContentsMargins(0, 0, 0, 0);
    outer->setSpacing(0);

    // App logo / title bar
    auto* header = new QWidget;
    header->setStyleSheet("background: palette(base); border-bottom: 1px solid palette(mid);");
    auto* hLay = new QVBoxLayout(header);
    hLay->setContentsMargins(14, 14, 14, 10);

    auto* appName = new QLabel("NeuralStudio");
    { QFont f = appName->font(); f.setPointSize(15); f.setBold(true); appName->setFont(f); }
    hLay->addWidget(appName);
    outer->addWidget(header);

    // Scrollable content
    auto* scroll = new QScrollArea;
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    auto* content = new QWidget;
    auto* lay = new QVBoxLayout(content);
    lay->setContentsMargins(8, 8, 8, 8);
    lay->setSpacing(2);

    m_stepBtns = new QButtonGroup(this);
    m_stepBtns->setExclusive(true);

    // ── Mode 1: Single Workflow ───────────────────────────────────────────
    addModeHeader(lay, "① SINGLE WORKFLOW", "#58A6FF");
    addWorkflowStep(lay, 0, "Data",       true);
    addWorkflowStep(lay, 1, "Network",    false);
    addWorkflowStep(lay, 2, "Training",   false);
    addWorkflowStep(lay, 3, "Evaluation", false);
    addWorkflowStep(lay, 4, "Predict",    false);

    addSeparator(lay);

    // ── Mode 2: Batch Run ─────────────────────────────────────────────────
    // Modes 2–6: one clickable button each (no separate header label).
    addModeButton(lay, 5, "② BATCH RUN", "#E3B341");

    addSeparator(lay);

    // ── Mode 3: Pre-Train + NNC ───────────────────────────────────────────
    addModeButton(lay, 6, "③ PRE-TRAIN + NNC", "#F0883E");

    addSeparator(lay);

    // ── Mode 4: Evol. Bounds ──────────────────────────────────────────────
    addModeButton(lay, 7, "④ EVOL. BOUNDS", "#BC8CFF");

    addSeparator(lay);

    // ── Mode 5: Architecture Search ──────────────────
    addModeButton(lay, 8, "⑤ ARCH. SEARCH", "#3DC9B0");

    addSeparator(lay);

    // ── Mode 6: Grammatical Evolution Studio ──────────────────────────────
    addModeButton(lay, 9, "⑥ GE STUDIO (BNF)", "#FF7B72");

    lay->addStretch();
    scroll->setWidget(content);
    outer->addWidget(scroll);

    // Select first button
    auto* first = qobject_cast<QPushButton*>(m_stepBtns->button(0));
    if (first) first->setChecked(true);
    return m_sidebar;
}

// ─── Sidebar helpers ──────────────────────────────────────────────────────────

QLabel* MainWindow::addModeHeader(QVBoxLayout* lay, const QString& text,
                                   const QString& color) {
    auto* l = new QLabel(text);
    QFont f = l->font(); f.setPointSize(9); f.setBold(true); l->setFont(f);
    const QString col = color.isEmpty() ? "palette(mid)" : color;
    l->setStyleSheet(QString("color: %1; padding: 8px 6px 2px 6px;").arg(col));
    lay->addWidget(l);
    return l;
}

// Workflow sub-step (Data, Network, etc.) — indented, can be disabled
QPushButton* MainWindow::addWorkflowStep(QVBoxLayout* lay, int idx,
                                          const QString& label, bool enabled) {
    auto* btn = new QPushButton(label);
    btn->setCheckable(true);
    btn->setEnabled(enabled);
    btn->setFixedHeight(32);
    btn->setStyleSheet(
        "QPushButton {"
        "  text-align: left; padding-left: 28px; border: none;"
        "  border-radius: 5px; font-size: 13px; background: transparent;"
        "}"
        "QPushButton:hover:enabled { background: palette(midlight); }"
        "QPushButton:checked {"
        "  background: #58A6FF22;"
        "  border-left: 3px solid #58A6FF;"
        "  color: #58A6FF; font-weight: 600;"
        "}"
        "QPushButton:disabled { color: palette(mid); }");
    m_stepBtns->addButton(btn, idx);
    lay->addWidget(btn);
    return btn;
}

// Mode selector button (Batch Run, Pre-Train, Evol. Bounds) — always enabled
QPushButton* MainWindow::addModeButton(QVBoxLayout* lay, int idx,
                                        const QString& label,
                                        const QString& accentColor) {
    auto* btn = new QPushButton(label);
    btn->setCheckable(true);
    btn->setEnabled(true);
    btn->setFixedHeight(36);
    const QString col = accentColor.isEmpty() ? "#AAAAAA" : accentColor;
    // Text is always shown in the mode's accent colour (it replaces the old
    // coloured header label), bold like the "① SINGLE WORKFLOW" header.
    btn->setStyleSheet(QString(
        "QPushButton {"
        "  text-align: left; padding-left: 6px; border: none;"
        "  border-radius: 6px; font-size: 12px; background: transparent;"
        "  font-weight: 700; color: %2;"
        "}"
        "QPushButton:hover { background: %1; color: %2; }"
        "QPushButton:checked {"
        "  background: %1; color: %2; font-weight: 700;"
        "  border-left: 3px solid %2;"
        "}"
        "QPushButton:disabled { color: palette(mid); }")
        .arg(col + "22")   // 22 = ~13% opacity hex
        .arg(col));
    m_stepBtns->addButton(btn, idx);
    lay->addWidget(btn);
    return btn;
}

QFrame* MainWindow::addSeparator(QVBoxLayout* lay) {
    auto* f = new QFrame;
    f->setFrameShape(QFrame::HLine);
    f->setStyleSheet("color: palette(mid); margin: 6px 2px;");
    f->setFixedHeight(1);
    lay->addWidget(f);
    return f;
}

// ─── setSidebarEnabled / activateStep ────────────────────────────────────────
void MainWindow::setSidebarEnabled(int fromIdx, bool enabled) {
    for (auto* btn : m_stepBtns->buttons()) {
        if (m_stepBtns->id(btn) >= fromIdx)
            btn->setEnabled(enabled);
    }
}

void MainWindow::lockSidebar(bool lock) {
    if (lock) {
        // Save current enabled state of every button, then disable all
        m_sidebarStates.clear();
        for (auto* btn : m_stepBtns->buttons()) {
            const int id = m_stepBtns->id(btn);
            m_sidebarStates[id] = btn->isEnabled();
            btn->setEnabled(false);
        }
    } else {
        // Restore previously saved states
        for (auto* btn : m_stepBtns->buttons()) {
            const int id = m_stepBtns->id(btn);
            if (m_sidebarStates.contains(id))
                btn->setEnabled(m_sidebarStates.value(id));
        }
        m_sidebarStates.clear();
    }
}

void MainWindow::activateStep(int idx) {
    auto* btn = qobject_cast<QPushButton*>(m_stepBtns->button(idx));
    if (btn) btn->setChecked(true);
    m_stack->setCurrentIndex(idx);
}

// ─── Workflow slot handlers ───────────────────────────────────────────────────
void MainWindow::onDatasetLoaded(const Dataset* ds) {
    if (!ds) return;
    setSidebarEnabled(1, true);
    statusBar()->showMessage(QString("Loaded: %1  |  %2 samples  |  %3 inputs")
        .arg(ds->name).arg(ds->sampleCount).arg(ds->inputCount));
    activateStep(1);
}

void MainWindow::onNetworkBuilt(std::shared_ptr<NeuralNetwork> net) {
    m_net = net;
    setSidebarEnabled(2, true);
    statusBar()->showMessage("Network: " + net->summary());
    activateStep(2);
}

void MainWindow::onRBFNetworkBuilt(std::shared_ptr<RBFNetwork> net) {
    m_rbfNet = net;
    setSidebarEnabled(2, true);
    statusBar()->showMessage(QString("RBF Network: k=%1 centres, ready to train.")
                             .arg(net->config().nCenters));
    activateStep(2);
}

void MainWindow::onRBFTrainingCompleted(std::shared_ptr<RBFNetwork> net,
                                        double /*trainErr*/, double /*valErr*/) {
    m_rbfNet = net;
    m_evaluationPanel->onRBFBuilt(net);
    m_predictionPanel->onRBFBuilt(net);
    setSidebarEnabled(3, true);
    setSidebarEnabled(4, true);
    m_evaluationPanel->onTrainingCompleted();
    activateStep(3);
}

void MainWindow::onTrainingCompleted() {
    setSidebarEnabled(3, true);
    setSidebarEnabled(4, true);
    statusBar()->showMessage("Training complete. Evaluation + Prediction ready.");
    activateStep(3);
}

void MainWindow::onStepChanged(int index) { m_stack->setCurrentIndex(index); }

// ─── Menu bar ─────────────────────────────────────────────────────────────────
void MainWindow::buildMenuBar() {
    QMenu* fileMenu = menuBar()->addMenu("&File");
    fileMenu->addAction("&Open Dataset...", QKeySequence::Open, this, &MainWindow::onOpenFile);
    fileMenu->addSeparator();
    fileMenu->addAction("Open &Project...", this, &MainWindow::onOpenProject);
    fileMenu->addAction("Save Pro&ject...", this, &MainWindow::onSaveProject);
    m_recentMenu = fileMenu->addMenu("Open &Recent");
    rebuildRecentMenu();
    fileMenu->addSeparator();
    fileMenu->addAction("&Save Model...", QKeySequence::Save, this, &MainWindow::onSaveModel);
    fileMenu->addAction("&Load Model...", this, &MainWindow::onLoadModel);
    fileMenu->addSeparator();
    fileMenu->addAction("&Export C++ Header...", this, &MainWindow::onExportCpp);
    fileMenu->addAction("Export O&NNX Model...", this, &MainWindow::onExportOnnx);
    fileMenu->addSeparator();
    fileMenu->addAction("&Quit", QKeySequence::Quit, qApp, &QApplication::quit);

    QMenu* helpMenu = menuBar()->addMenu("&Help");
    helpMenu->addAction("&About", this, &MainWindow::onAbout);
}

void MainWindow::onOpenFile() {
    const QString path = QFileDialog::getOpenFileName(
        this, "Open Dataset", {}, DatasetLoader::fileDialogFilter());
    if (!path.isEmpty()) loadDatasetFile(path);
}

void MainWindow::loadDatasetFile(const QString& path) {
    activateStep(0);
    m_dataPanel->loadFile(path);
    addToRecent(path);
}

// ─── Recent files ─────────────────────────────────────────────────────────────
void MainWindow::rebuildRecentMenu() {
    m_recentMenu->clear();
    QSettings settings;
    QStringList files = settings.value("recentFiles").toStringList();
    QStringList valid;
    for (const QString& p : files) if (QFileInfo::exists(p)) valid << p;
    if (valid != files) settings.setValue("recentFiles", valid);
    if (valid.isEmpty()) {
        QAction* none = m_recentMenu->addAction("(no recent files)");
        none->setEnabled(false); return;
    }
    for (const QString& p : valid) {
        QAction* act = m_recentMenu->addAction(QFileInfo(p).fileName());
        act->setData(p); act->setToolTip(p);
        connect(act, &QAction::triggered, this, &MainWindow::onRecentFileTriggered);
    }
    m_recentMenu->addSeparator();
    m_recentMenu->addAction("&Clear Recent", this, &MainWindow::onClearRecent);
}

void MainWindow::addToRecent(const QString& path) {
    QSettings settings;
    QStringList files = settings.value("recentFiles").toStringList();
    files.removeAll(path); files.prepend(path);
    while (files.size() > kMaxRecent) files.removeLast();
    settings.setValue("recentFiles", files);
    rebuildRecentMenu();
}

void MainWindow::onClearRecent() {
    QSettings settings; settings.setValue("recentFiles", QStringList{});
    rebuildRecentMenu();
}

void MainWindow::onRecentFileTriggered() {
    auto* act = qobject_cast<QAction*>(sender());
    if (!act) return;
    loadDatasetFile(act->data().toString());
}

// ─── Model save / load / export ───────────────────────────────────────────────
void MainWindow::onSaveModel() {
    if (!m_net || !m_net->isBuilt()) {
        QMessageBox::information(this, "Save Model", "No trained model to save."); return;
    }
    const QString path = QFileDialog::getSaveFileName(
        this, "Save Model", "model.nsmodel", ModelSerializer::fileFilter());
    if (path.isEmpty()) return;
    try {
        ModelSerializer::save(*m_net, path);
        statusBar()->showMessage("Model saved: " + path);
    } catch (const std::exception& ex) {
        QMessageBox::critical(this, "Save Error", ex.what());
    }
}

void MainWindow::onLoadModel() {
    const QString path = QFileDialog::getOpenFileName(
        this, "Load Model", {}, ModelSerializer::fileFilter());
    if (path.isEmpty()) return;
    try {
        auto net = std::make_shared<NeuralNetwork>();
        ModelSerializer::load(*net, path);
        m_net = net;
        emit m_networkPanel->networkBuilt(net);
        setSidebarEnabled(2, true); setSidebarEnabled(3, true); setSidebarEnabled(4, true);
        statusBar()->showMessage("Model loaded: " + net->summary());
    } catch (const std::exception& ex) {
        QMessageBox::critical(this, "Load Error", ex.what());
    }
}

// ─── Project save / load (.nsproj) ────────────────────────────────────────────
//  A project file captures the SESSION SETUP (which files, which architecture,
//  which mode queues) — not trained weights; those belong to .nsmodel files.
//  Format: JSON, versioned for forward compatibility.
// ─── Export ONNX ──────────────────────────────────────────────────────────────
void MainWindow::onExportOnnx() {
    if (!m_net || !m_net->isBuilt()) {
        QMessageBox::information(this, "Export ONNX",
            "No trained model to export. Train a network in Mode 1 first.");
        return;
    }
    const QString path = QFileDialog::getSaveFileName(
        this, "Export ONNX Model", "model.onnx", OnnxExporter::fileFilter());
    if (path.isEmpty()) return;
    try {
        OnnxExporter::save(*m_net, path);
        QMessageBox::information(this, "Export ONNX",
            "Model exported.\n\n"
            "Use it in Python with onnxruntime:\n\n"
            "  import onnxruntime as ort, numpy as np\n"
            "  sess = ort.InferenceSession(\"" + QFileInfo(path).fileName() + "\")\n"
            "  x = np.array([[...raw features...]], dtype=np.float32)\n"
            "  y = sess.run(None, {\"input\": x})[0]\n\n"
            "Feed RAW feature values \u2014 input normalization is embedded in "
            "the graph. Multiclass models output class probabilities (softmax).");
        statusBar()->showMessage("ONNX model exported: " + QFileInfo(path).fileName());
    } catch (const std::exception& ex) {
        QMessageBox::critical(this, "Export Error", ex.what());
    }
}

void MainWindow::onSaveProject() {
    const QString path = QFileDialog::getSaveFileName(
        this, "Save Project", "session.nsproj",
        "NeuralStudio Project (*.nsproj)");
    if (path.isEmpty()) return;

    QJsonObject root;
    root["format"]  = "nsproj";
    root["version"] = 1;

    // Mode 1: dataset + architecture
    QJsonObject m1;
    m1["dataset"] = m_dataPanel->currentFilePath();
    QVector<int> sizes; QVector<QString> acts;
    if (m_networkPanel->currentArchitecture(sizes, acts)) {
        QJsonArray sArr, aArr;
        for (int v : sizes) sArr.append(v);
        for (const QString& a : acts) aArr.append(a);
        m1["hiddenSizes"]       = sArr;
        m1["hiddenActivations"] = aArr;
    }
    root["mode1"] = m1;

    // Modes 2–5: dataset queues
    auto queueToJson = [](const QStringList& q){
        QJsonArray arr; for (const QString& p : q) arr.append(p); return arr;
    };
    root["mode2Queue"] = queueToJson(m_batchPanel->queuePaths());
    root["mode3Queue"] = queueToJson(m_preTrainPanel->queuePaths());
    root["mode4Queue"] = queueToJson(m_evolBoundsPanel->queuePaths());
    root["mode5Queue"] = queueToJson(m_archSearchPanel->queuePaths());
    root["mode6Queue"] = queueToJson(m_geStudioPanel->queuePaths());

    QFile f(path);
    if (!f.open(QIODevice::WriteOnly)) {
        QMessageBox::critical(this, "Save Project", "Cannot write: " + path);
        return;
    }
    f.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
    statusBar()->showMessage("Project saved: " + QFileInfo(path).fileName());
}

void MainWindow::onOpenProject() {
    const QString path = QFileDialog::getOpenFileName(
        this, "Open Project", {}, "NeuralStudio Project (*.nsproj)");
    if (path.isEmpty()) return;

    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) {
        QMessageBox::critical(this, "Open Project", "Cannot read: " + path);
        return;
    }
    QJsonParseError perr;
    const QJsonDocument doc = QJsonDocument::fromJson(f.readAll(), &perr);
    if (perr.error != QJsonParseError::NoError || !doc.isObject()) {
        QMessageBox::critical(this, "Open Project",
            "Not a valid project file:\n" + perr.errorString());
        return;
    }
    const QJsonObject root = doc.object();
    if (root["format"].toString() != "nsproj") {
        QMessageBox::critical(this, "Open Project", "Not a NeuralStudio project file.");
        return;
    }

    // Restore queues first (cheap, no side effects); skip files that no
    // longer exist on disk and report them at the end.
    auto jsonToQueue = [](const QJsonValue& v, QStringList& missing){
        QStringList out;
        for (const auto& item : v.toArray()) {
            const QString p = item.toString();
            if (p.isEmpty()) continue;
            if (QFileInfo::exists(p)) out << p; else missing << p;
        }
        return out;
    };
    QStringList missing;
    m_batchPanel     ->setQueuePaths(jsonToQueue(root["mode2Queue"], missing));
    m_preTrainPanel  ->setQueuePaths(jsonToQueue(root["mode3Queue"], missing));
    m_evolBoundsPanel->setQueuePaths(jsonToQueue(root["mode4Queue"], missing));
    m_archSearchPanel->setQueuePaths(jsonToQueue(root["mode5Queue"], missing));
    m_geStudioPanel  ->setQueuePaths(jsonToQueue(root["mode6Queue"], missing));

    // Mode 1: dataset then architecture (same order as Apply-to-Mode-1)
    const QJsonObject m1 = root["mode1"].toObject();
    const QString dsPath = m1["dataset"].toString();
    if (!dsPath.isEmpty()) {
        if (QFileInfo::exists(dsPath)) {
            m_dataPanel->loadFile(dsPath);
            if (const Dataset* testDs = m_dataPanel->testDataset()) {
                m_evaluationPanel->onTestDatasetLoaded(testDs);
                m_trainingPanel->onTestDatasetLoaded(testDs);
            }
        } else missing << dsPath;
    }
    if (m1.contains("hiddenSizes")) {
        QVector<int> sizes; QVector<QString> acts;
        for (const auto& v : m1["hiddenSizes"].toArray())       sizes.push_back(v.toInt());
        for (const auto& v : m1["hiddenActivations"].toArray()) acts.push_back(v.toString());
        if (!sizes.isEmpty()) m_networkPanel->applyArchitecture(sizes, acts);
    }

    if (!missing.isEmpty())
        QMessageBox::warning(this, "Open Project",
            QString("Loaded, but %1 file(s) referenced by the project no longer "
                    "exist and were skipped:\n\n%2")
                .arg(missing.size()).arg(missing.join("\n")));

    statusBar()->showMessage("Project loaded: " + QFileInfo(path).fileName());
}

void MainWindow::onExportCpp() {
    if (!m_net || !m_net->isBuilt()) {
        QMessageBox::information(this, "Export C++", "No trained model to export."); return;
    }
    const QString path = QFileDialog::getSaveFileName(
        this, "Export C++ Header", "neuralstudio_model.h", CppExporter::fileFilter());
    if (path.isEmpty()) return;
    try {
        CppExporter::exportHeader(*m_net, path);
        QMessageBox::information(this, "Export Complete",
            QString("Model exported to:\n%1\n\nUsage:\n  #include \"%2\"\n"
                    "  auto result = NeuralStudioModel::predict(inputs);")
                .arg(path, QFileInfo(path).fileName()));
        statusBar()->showMessage("C++ export: " + path);
    } catch (const std::exception& ex) {
        QMessageBox::critical(this, "Export Error", ex.what());
    }
}

// ─── About ────────────────────────────────────────────────────────────────────
void MainWindow::onAbout() {
    QMessageBox::about(this, "About NeuralStudio",
        "<h3>NeuralStudio &nbsp; v1.0.6</h3>"
        "<p>C++20 + Qt 6 neural network training workbench.</p>"

        "<b>Authors</b>"
        "<ul>"
        "<li>Vasileios Charilogis</li>"
        "<li>Ioannis G. Tsoulos</li>"
        "<li>Anna Maria Gianni</li>"
        "</ul>"
        "<p><i>OptimTeam &nbsp;·&nbsp; Department of Informatics &amp; Telecommunications, "
        "University of Ioannina</i></p>");
}

} // namespace NeuralStudio
