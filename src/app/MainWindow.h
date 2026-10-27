#pragma once
#include <QMainWindow>
#include <QStackedWidget>
#include <QButtonGroup>
#include <QMap>
#include <QPushButton>
#include <QVBoxLayout>
#include <QLabel>
#include <QFrame>
#include <memory>

QT_BEGIN_NAMESPACE
class QMenu;
QT_END_NAMESPACE

namespace NeuralStudio {

class DataPanel; class NetworkPanel; class TrainingPanel;
class EvaluationPanel; class PredictionPanel; class BatchPanel;
class PreTrainNNCPanel; class EvolBoundsPanel; class ArchSearchPanel; class GEStudioPanel;
class Dataset; class NeuralNetwork; class RBFNetwork;

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);

private slots:
    void onDatasetLoaded(const Dataset* ds);
    void onNetworkBuilt(std::shared_ptr<NeuralNetwork> net);
    void onApplyArchToMode1(QString datasetPath, QVector<int> hiddenSizes, QVector<QString> actNames);
    void onRBFNetworkBuilt(std::shared_ptr<RBFNetwork> net);
    void onRBFTrainingCompleted(std::shared_ptr<RBFNetwork> net, double trainErr, double valErr);
    void onTrainingCompleted();
    void onStepChanged(int index);
    void onOpenFile();
    void onSaveModel();
    void onLoadModel();
    void onSaveProject();   // whole-session state → .nsproj (JSON)
    void onOpenProject();   // restore session from .nsproj
    void onExportCpp();
    void onExportOnnx();   // deployment export (onnxruntime-compatible)
    void onAbout();
    void onClearRecent();
    void onRecentFileTriggered();

private:
    QWidget*        m_sidebar  = nullptr;
    QStackedWidget* m_stack    = nullptr;
    QButtonGroup*   m_stepBtns     = nullptr;
    QMap<int,bool>  m_sidebarStates;   // saved enabled states for lockSidebar

    DataPanel*        m_dataPanel       = nullptr;
    NetworkPanel*     m_networkPanel    = nullptr;
    TrainingPanel*    m_trainingPanel   = nullptr;
    EvaluationPanel*  m_evaluationPanel = nullptr;
    PredictionPanel*  m_predictionPanel = nullptr;
    BatchPanel*       m_batchPanel      = nullptr;
    PreTrainNNCPanel* m_preTrainPanel   = nullptr;
    EvolBoundsPanel*  m_evolBoundsPanel = nullptr;
    ArchSearchPanel*  m_archSearchPanel = nullptr;
    GEStudioPanel*    m_geStudioPanel   = nullptr;

    std::shared_ptr<NeuralNetwork> m_net;
    std::shared_ptr<RBFNetwork>    m_rbfNet;

    QMenu* m_recentMenu = nullptr;

    void buildUi();
    void buildMenuBar();
    void loadDatasetFile(const QString& path);
    void rebuildRecentMenu();
    void addToRecent(const QString& path);
    QWidget*     buildSidebar();

    // Sidebar helpers
    QPushButton* addWorkflowStep(QVBoxLayout* lay, int idx,
                                 const QString& label, bool enabled);
    QPushButton* addModeButton  (QVBoxLayout* lay, int idx,
                                 const QString& label,
                                 const QString& accentColor = QString());
    QFrame*      addSeparator   (QVBoxLayout* lay);
    QLabel*      addModeHeader  (QVBoxLayout* lay, const QString& text,
                                 const QString& color = QString());

    void setSidebarEnabled(int fromIdx, bool enabled);
    void lockSidebar(bool lock);   // saves/restores all button states
    void activateStep(int idx);
};

} // namespace NeuralStudio
