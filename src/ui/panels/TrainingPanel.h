#pragma once
#include "core/nn/NeuralNetwork.h"
#include "core/nn/Trainer.h"
#include "core/nn/RBFNetwork.h"
#include "core/nn/RBFTrainer.h"
#include "core/optimizers/MetaTrainer.h"
#include "core/validation/CrossValidator.h"
#include "core/dataset/Dataset.h"
#include <QWidget>
#include <QLabel>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QComboBox>
#include <QPushButton>
#include <QProgressBar>
#include <QTabWidget>
#include <QThread>
#include <QPointer>
#include <memory>
#include <vector>

QT_BEGIN_NAMESPACE
class QChart; class QChartView; class QLineSeries; class QValueAxis;
QT_END_NAMESPACE

namespace NeuralStudio {

class TrainingPanel : public QWidget {
    Q_OBJECT
public:
    explicit TrainingPanel(QWidget* parent = nullptr);

public slots:
    void onNetworkBuilt(std::shared_ptr<NeuralNetwork> net);
    void onRBFBuilt(std::shared_ptr<RBFNetwork> net);   // v0.8.0
    void onDatasetLoaded(const Dataset* ds);

signals:
    void trainingCompleted();
    void requestStop();
    void rbfTrainingCompleted(std::shared_ptr<RBFNetwork> net, double trainErr, double valErr);

private slots:
    void onTrainClicked();
    void onCVClicked();
    void onExportLogClicked();
    void onOptimizerChanged(int index);      // show/hide gradient-specific controls
    void onEpochCompleted(EpochResult result);
    void onTrainingFinished(double finalVal);
    void onTrainingError(QString msg);
    void onFoldCompleted(FoldResult fr);
    void onCVFinished(CVSummary sum);
    void onCVError(QString msg);

private:
    std::shared_ptr<NeuralNetwork> m_net;
    std::shared_ptr<RBFNetwork>    m_rbfNet;   // v0.8.0
    const Dataset* m_ds = nullptr;
    QPointer<QThread>  m_thread;
    QPointer<Trainer>  m_trainer;
    QPointer<MetaTrainer> m_metaTrainer;
    QPointer<QThread>  m_cvThread;
    QPointer<CrossValidator> m_cv;
    bool m_running = false;

    // Config widgets
    QComboBox*      m_optimCombo  = nullptr;
    QDoubleSpinBox* m_lrSpin      = nullptr;
    QComboBox*      m_lrSchedCombo= nullptr;
    QSpinBox*       m_epochsSpin  = nullptr;
    QSpinBox*       m_batchSpin   = nullptr;  // = Population for meta-optimizers
    QDoubleSpinBox* m_valSpin     = nullptr;
    QSpinBox*       m_earlySpin   = nullptr;
    QSpinBox*       m_kfoldSpin   = nullptr;
    QDoubleSpinBox* m_wBoundSpin  = nullptr;  // weight bound (evolutionary only)
    QLabel*         m_batchLbl    = nullptr;  // dynamic: "Batch" ↔ "Population"
    QLabel*         m_lrLbl       = nullptr;
    QLabel*         m_schedLbl    = nullptr;
    QLabel*         m_wBoundLbl   = nullptr;

    QPushButton*    m_trainBtn    = nullptr;
    QPushButton*    m_cvBtn       = nullptr;
    QPushButton*    m_exportLogBtn= nullptr;
    QLabel*         m_statusLbl   = nullptr;
    QProgressBar*   m_progress    = nullptr;
    QLabel*         m_cvResultLbl = nullptr;

    QTabWidget*  m_chartTabs   = nullptr;
    QLabel*      m_rbfSummaryLbl = nullptr;   // shown after RBF training, hidden for MLP

    // Loss chart
    QChart*      m_lossChart   = nullptr;
    QChartView*  m_lossView    = nullptr;
    QLineSeries* m_trainSeries = nullptr;
    QLineSeries* m_valSeries   = nullptr;
    QValueAxis*  m_lossAxisX   = nullptr;
    QValueAxis*  m_lossAxisY   = nullptr;

    // Accuracy chart
    QChart*      m_accChart    = nullptr;
    QChartView*  m_accView     = nullptr;
    QLineSeries* m_trainAccSer = nullptr;
    QLineSeries* m_valAccSer   = nullptr;
    QValueAxis*  m_accAxisX    = nullptr;
    QValueAxis*  m_accAxisY    = nullptr;

    // Training history for CSV export
    std::vector<EpochResult> m_history;

    void buildUi();
    void styleChart(QChart* c);
    void setRunning(bool r);
    void resetCharts();
    bool isGradientDescent() const;  // Adam/SGD/AdamW/RMSProp/Nadam/AdaGrad
    bool isEvolutionary() const;     // population-based meta-optimizers
};

} // namespace NeuralStudio
