#pragma once
#include "core/nn/NeuralNetwork.h"
#include "core/nn/RBFNetwork.h"
#include "core/dataset/Dataset.h"
#include <QWidget>
#include <QLabel>
#include <QDoubleSpinBox>
#include <QPushButton>
#include <QScrollArea>
#include <memory>
#include <vector>

namespace NeuralStudio {

// ─── PredictionPanel ──────────────────────────────────────────────────────────
//  Phase 4: manual single-sample prediction.
//  Generates one spinbox per input feature (pre-filled with column mean).
//  Pressing "Predict" runs the trained network and shows the result.
// ─────────────────────────────────────────────────────────────────────────────
class PredictionPanel : public QWidget {
    Q_OBJECT
public:
    explicit PredictionPanel(QWidget* parent = nullptr);

public slots:
    void onDatasetLoaded(const Dataset* ds);
    void onNetworkBuilt(std::shared_ptr<NeuralNetwork> net);
    void onRBFBuilt(std::shared_ptr<RBFNetwork> net);      // v0.8.0
    void onTrainingCompleted();

private slots:
    void onPredictClicked();
    void onLoadFirstSampleClicked();
    void onResetClicked();

private:
    const Dataset*                 m_ds  = nullptr;
    std::shared_ptr<NeuralNetwork> m_net;
    std::shared_ptr<RBFNetwork>    m_rbfNet;   // v0.8.0
    std::vector<QDoubleSpinBox*>   m_inputSpins;

    QWidget*      m_inputsHost   = nullptr;
    QPushButton*  m_predictBtn   = nullptr;
    QPushButton*  m_loadFirstBtn = nullptr;
    QPushButton*  m_resetBtn     = nullptr;
    QLabel*       m_resultLbl    = nullptr;
    QLabel*       m_statusLbl    = nullptr;

    void buildUi();
    void rebuildInputFields();
    void resetToMeans();
};

} // namespace NeuralStudio
