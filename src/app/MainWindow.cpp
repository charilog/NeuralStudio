#include "MainWindow.h"
#include "ui/panels/DataPanel.h"
#include "ui/panels/NetworkPanel.h"
#include "ui/panels/TrainingPanel.h"
#include "ui/panels/EvaluationPanel.h"
#include "ui/panels/PredictionPanel.h"
#include "ui/panels/BatchPanel.h"
#include "core/dataset/DatasetLoader.h"
#include "core/nn/NeuralNetwork.h"
#include "core/nn/ModelSerializer.h"
#include "core/export/CppExporter.h"

#include <QSplitter>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QMessageBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QApplication>
#include <QSettings>
#include <QStatusBar>

namespace NeuralStudio {

static constexpr int kMaxRecent = 8;

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowTitle("NeuralStudio  0.8.0");
    setWindowIcon(QIcon(":/icons/neuralstudio.svg"));
    setMinimumSize(1080, 700);
    resize(1340, 820);
    buildUi();
    buildMenuBar();
    statusBar()->showMessage("Ready");
}

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
    m_stack->addWidget(m_dataPanel);       // 0
    m_stack->addWidget(m_networkPanel);    // 1
    m_stack->addWidget(m_trainingPanel);   // 2
    m_stack->addWidget(m_evaluationPanel); // 3
    m_stack->addWidget(m_predictionPanel); // 4
    m_stack->addWidget(m_batchPanel);      // 5 (Tools)
    splitter->addWidget(m_stack);
    splitter->setCollapsible(1, false);
    splitter->setSizes({210, 1130});

    connect(m_dataPanel, &DataPanel::datasetLoaded, this, &MainWindow::onDatasetLoaded);
    connect(m_dataPanel, &DataPanel::datasetLoaded, m_networkPanel,    &NetworkPanel::onDatasetLoaded);
    connect(m_dataPanel, &DataPanel::datasetLoaded, m_trainingPanel,   &TrainingPanel::onDatasetLoaded);
    connect(m_dataPanel, &DataPanel::datasetLoaded, m_evaluationPanel, &EvaluationPanel::onDatasetLoaded);
    connect(m_dataPanel, &DataPanel::datasetLoaded, m_predictionPanel, &PredictionPanel::onDatasetLoaded);

    connect(m_dataPanel, &DataPanel::testDatasetLoaded,
            m_evaluationPanel, &EvaluationPanel::onTestDatasetLoaded);

    connect(m_networkPanel, &NetworkPanel::networkBuilt, this, &MainWindow::onNetworkBuilt);
    connect(m_networkPanel, &NetworkPanel::networkBuilt, m_trainingPanel,   &TrainingPanel::onNetworkBuilt);
    connect(m_networkPanel, &NetworkPanel::networkBuilt, m_evaluationPanel, &EvaluationPanel::onNetworkBuilt);
    connect(m_networkPanel, &NetworkPanel::networkBuilt, m_predictionPanel, &PredictionPanel::onNetworkBuilt);
    // v0.8.0: RBF connections
    connect(m_networkPanel, &NetworkPanel::rbfBuilt, m_trainingPanel, &TrainingPanel::onRBFBuilt);
    connect(m_networkPanel, &NetworkPanel::rbfBuilt, this, &MainWindow::onRBFNetworkBuilt);
    connect(m_trainingPanel, &TrainingPanel::rbfTrainingCompleted, this, &MainWindow::onRBFTrainingCompleted);

    connect(m_trainingPanel, &TrainingPanel::trainingCompleted, this, &MainWindow::onTrainingCompleted);
    connect(m_trainingPanel, &TrainingPanel::trainingCompleted, m_evaluationPanel, &EvaluationPanel::onTrainingCompleted);
    connect(m_trainingPanel, &TrainingPanel::trainingCompleted, m_predictionPanel, &PredictionPanel::onTrainingCompleted);

    connect(m_stepBtns, &QButtonGroup::idClicked, this, &MainWindow::onStepChanged);
}

QWidget* MainWindow::buildSidebar() {
    m_sidebar = new QWidget;
    m_sidebar->setFixedWidth(210);
    auto* lay = new QVBoxLayout(m_sidebar);
    lay->setContentsMargins(12, 16, 12, 16);
    lay->setSpacing(4);

    auto* title = new QLabel("NeuralStudio");
    QFont tf = title->font(); tf.setPointSize(15); tf.setBold(true); title->setFont(tf);
    title->setContentsMargins(4, 0, 0, 12);
    lay->addWidget(title);

    auto sectionLbl = [&](const QString& t) -> QLabel* {
        auto* l = new QLabel(t);
        QFont f = l->font(); f.setPointSize(9); l->setFont(f);
        l->setStyleSheet("color:palette(mid);padding:8px 4px 2px;");
        return l;
    };

    m_stepBtns = new QButtonGroup(this);
    m_stepBtns->setExclusive(true);

    lay->addWidget(sectionLbl("WORKFLOW"));
    addStepButton(lay, 0, "\u2460  Data",       true);
    addStepButton(lay, 1, "\u2461  Network",    false);
    addStepButton(lay, 2, "\u2462  Training",   false);
    addStepButton(lay, 3, "\u2463  Evaluation", false);
    addStepButton(lay, 4, "\u2464  Predict",    false);

    lay->addWidget(sectionLbl("TOOLS"));
    addStepButton(lay, 5, "\u25EB  Batch Run",  true);

    lay->addStretch();

    auto* ver = new QLabel("v0.8.0");
    ver->setStyleSheet("color:#1D9E75;font-size:12px;font-weight:700;");
    ver->setAlignment(Qt::AlignCenter);
    lay->addWidget(ver);

    qobject_cast<QPushButton*>(m_stepBtns->button(0))->setChecked(true);
    return m_sidebar;
}

QPushButton* MainWindow::addStepButton(QVBoxLayout* lay, int idx,
                                        const QString& label, bool enabled) {
    auto* btn = new QPushButton(label);
    btn->setCheckable(true);
    btn->setEnabled(enabled);
    btn->setFixedHeight(38);
    btn->setStyleSheet(
        "QPushButton{text-align:left;padding-left:12px;border:none;border-radius:6px;"
        "font-size:13px;background:transparent;}"
        "QPushButton:hover:enabled{background:palette(midlight);}"
        "QPushButton:checked{background:palette(highlight);color:palette(highlighted-text);font-weight:600;}"
        "QPushButton:disabled{color:palette(mid);}");
    m_stepBtns->addButton(btn, idx);
    lay->addWidget(btn);
    return btn;
}

void MainWindow::setSidebarEnabled(int fromIdx, bool enabled) {
    for (auto* btn : m_stepBtns->buttons()) {
        if (m_stepBtns->id(btn) >= fromIdx) btn->setEnabled(enabled);
    }
}

void MainWindow::activateStep(int idx) {
    auto* btn = qobject_cast<QPushButton*>(m_stepBtns->button(idx));
    if (btn) btn->setChecked(true);
    m_stack->setCurrentIndex(idx);
}

void MainWindow::onDatasetLoaded(const Dataset* ds) {
    if (!ds) return;
    setSidebarEnabled(1, true);
    statusBar()->showMessage(QString("Loaded: %1  |  %2 samples  |  %3 inputs")
        .arg(ds->name).arg(ds->sampleCount).arg(ds->inputCount));
    activateStep(1);
}

void MainWindow::onRBFTrainingCompleted(std::shared_ptr<RBFNetwork> net,
                                        double /*trainErr*/, double /*valErr*/) {
    m_rbfNet = net;
    m_evaluationPanel->onRBFBuilt(net);
    m_predictionPanel->onRBFBuilt(net);
    setSidebarEnabled(3, true);
    setSidebarEnabled(4, true);
    // Pre-populate evaluation results (while still on Training panel)
    m_evaluationPanel->onTrainingCompleted();
    // Navigate to Evaluation
    activateStep(3);
}

void MainWindow::onRBFNetworkBuilt(std::shared_ptr<RBFNetwork> net) {
    m_rbfNet = net;
    setSidebarEnabled(2, true);   // enable Training step
    statusBar()->showMessage(QString("RBF Network: k=%1 centres, ready to train.")
                             .arg(net->config().nCenters));
    activateStep(2);   // navigate to Training
}

void MainWindow::onNetworkBuilt(std::shared_ptr<NeuralNetwork> net) {
    m_net = net;
    setSidebarEnabled(2, true);
    statusBar()->showMessage("Network: " + net->summary());
    activateStep(2);
}

void MainWindow::onTrainingCompleted() {
    setSidebarEnabled(3, true);
    setSidebarEnabled(4, true);
    statusBar()->showMessage("Training complete. Evaluation + Prediction ready.");
    activateStep(3);
}

void MainWindow::onStepChanged(int index) { m_stack->setCurrentIndex(index); }

void MainWindow::buildMenuBar() {
    QMenu* fileMenu = menuBar()->addMenu("&File");
    fileMenu->addAction("&Open Dataset...", QKeySequence::Open,
                        this, &MainWindow::onOpenFile);

    // Recent submenu
    m_recentMenu = fileMenu->addMenu("Open &Recent");
    rebuildRecentMenu();

    fileMenu->addSeparator();
    fileMenu->addAction("&Save Model...", QKeySequence::Save, this, &MainWindow::onSaveModel);
    fileMenu->addAction("&Load Model...",                      this, &MainWindow::onLoadModel);
    fileMenu->addSeparator();
    fileMenu->addAction("&Export C++ Header...",               this, &MainWindow::onExportCpp);
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

// ─── Recent files ────────────────────────────────────────────────────────────
void MainWindow::rebuildRecentMenu() {
    m_recentMenu->clear();
    QSettings settings;
    QStringList files = settings.value("recentFiles").toStringList();

    // Filter out non-existing
    QStringList valid;
    for (const QString& p : files) if (QFileInfo::exists(p)) valid << p;
    if (valid != files) settings.setValue("recentFiles", valid);

    if (valid.isEmpty()) {
        QAction* none = m_recentMenu->addAction("(no recent files)");
        none->setEnabled(false);
        return;
    }
    for (const QString& p : valid) {
        QAction* act = m_recentMenu->addAction(QFileInfo(p).fileName());
        act->setData(p);
        act->setToolTip(p);
        connect(act, &QAction::triggered, this, &MainWindow::onRecentFileTriggered);
    }
    m_recentMenu->addSeparator();
    m_recentMenu->addAction("&Clear Recent", this, &MainWindow::onClearRecent);
}

void MainWindow::addToRecent(const QString& path) {
    QSettings settings;
    QStringList files = settings.value("recentFiles").toStringList();
    files.removeAll(path);
    files.prepend(path);
    while (files.size() > kMaxRecent) files.removeLast();
    settings.setValue("recentFiles", files);
    rebuildRecentMenu();
}

void MainWindow::onClearRecent() {
    QSettings settings;
    settings.setValue("recentFiles", QStringList{});
    rebuildRecentMenu();
}

void MainWindow::onRecentFileTriggered() {
    auto* act = qobject_cast<QAction*>(sender());
    if (!act) return;
    loadDatasetFile(act->data().toString());
}

void MainWindow::onSaveModel() {
    if (!m_net || !m_net->isBuilt()) {
        QMessageBox::information(this, "Save Model", "No trained model to save.");
        return;
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
        setSidebarEnabled(2, true);
        setSidebarEnabled(3, true);
        setSidebarEnabled(4, true);
        statusBar()->showMessage("Model loaded: " + net->summary());
    } catch (const std::exception& ex) {
        QMessageBox::critical(this, "Load Error", ex.what());
    }
}

void MainWindow::onExportCpp() {
    if (!m_net || !m_net->isBuilt()) {
        QMessageBox::information(this, "Export C++", "No trained model to export.");
        return;
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

void MainWindow::onAbout() {
    QMessageBox::about(this, "About NeuralStudio",
        "<h3>NeuralStudio v0.6.0</h3>"
        "<p>C++20 + Qt 6 machine learning workbench.</p>"
        "<b>Phase 6 additions:</b><ul>"
        "<li>Batch Run mode — train many datasets with one architecture template</li>"
        "<li>Native XLSX export (zero external dependencies)</li>"
        "<li>Per-dataset metrics table with classification (Err %) and regression (MAE/RMSE/R²)</li>"
        "<li>Drag &amp; drop multi-file queue</li>"
        "</ul>");
}

} // namespace NeuralStudio
