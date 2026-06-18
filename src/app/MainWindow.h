#pragma once
#include <QMainWindow>
#include <QStackedWidget>
#include <QButtonGroup>
#include <QPushButton>
#include <QVBoxLayout>
#include <QLabel>
#include <memory>

QT_BEGIN_NAMESPACE
class QMenu;
QT_END_NAMESPACE

namespace NeuralStudio {

class DataPanel; class NetworkPanel; class TrainingPanel;
class EvaluationPanel; class PredictionPanel; class BatchPanel;
class Dataset; class NeuralNetwork; class RBFNetwork;

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);

private slots:
    void onDatasetLoaded(const Dataset* ds);
    void onNetworkBuilt(std::shared_ptr<NeuralNetwork> net);
    void onRBFNetworkBuilt(std::shared_ptr<RBFNetwork> net);       // v0.8.0
    void onRBFTrainingCompleted(std::shared_ptr<RBFNetwork> net, double trainErr, double valErr);
    void onTrainingCompleted();
    void onStepChanged(int index);
    void onOpenFile();
    void onSaveModel();
    void onLoadModel();
    void onExportCpp();
    void onAbout();
    void onClearRecent();
    void onRecentFileTriggered();

private:
    QWidget*        m_sidebar  = nullptr;
    QStackedWidget* m_stack    = nullptr;
    QButtonGroup*   m_stepBtns = nullptr;

    DataPanel*        m_dataPanel       = nullptr;
    NetworkPanel*     m_networkPanel    = nullptr;
    TrainingPanel*    m_trainingPanel   = nullptr;
    EvaluationPanel*  m_evaluationPanel = nullptr;
    PredictionPanel*  m_predictionPanel = nullptr;
    BatchPanel*       m_batchPanel      = nullptr;

    std::shared_ptr<NeuralNetwork> m_net;
    std::shared_ptr<RBFNetwork>    m_rbfNet;   // v0.8.0

    QMenu* m_recentMenu = nullptr;

    void buildUi();
    void buildMenuBar();
    void loadDatasetFile(const QString& path);  // central entry point
    void rebuildRecentMenu();
    void addToRecent(const QString& path);
    QWidget*     buildSidebar();
    QPushButton* addStepButton(QVBoxLayout* lay, int idx,
                               const QString& label, bool enabled);
    void setSidebarEnabled(int fromIdx, bool enabled);
    void activateStep(int idx);
};

} // namespace NeuralStudio
