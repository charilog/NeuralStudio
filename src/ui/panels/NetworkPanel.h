#pragma once
#include "core/dataset/Dataset.h"
#include "core/nn/NeuralNetwork.h"
#include "core/nn/RBFNetwork.h"
#include <QWidget>
#include <QLabel>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QComboBox>
#include <QVBoxLayout>
#include <QPushButton>
#include <QScrollArea>
#include <QStackedWidget>
#include <QButtonGroup>
#include <memory>
#include <vector>

namespace NeuralStudio {

class NetworkPanel : public QWidget {
    Q_OBJECT
public:
    explicit NetworkPanel(QWidget* parent = nullptr);
    std::shared_ptr<NeuralNetwork> network()    const { return m_net;    }
    std::shared_ptr<RBFNetwork>    rbfNetwork() const { return m_rbfNet; }
    bool isRBFMode() const;

public slots:
    void onDatasetLoaded(const Dataset* ds);

signals:
    void networkBuilt(std::shared_ptr<NeuralNetwork> net);  // MLP
    void rbfBuilt(std::shared_ptr<RBFNetwork>    net);      // RBF

private slots:
    void onNumLayersChanged(int n);
    void onBuildClicked();        // MLP build
    void onBuildRBFClicked();     // RBF build
    void onTypeToggled();         // MLP ↔ RBF switcher

private:
    const Dataset* m_ds = nullptr;
    std::shared_ptr<NeuralNetwork> m_net;
    std::shared_ptr<RBFNetwork>    m_rbfNet;

    // ── Type switcher (MLP | RBF) ─────────────────────────────────────────
    QPushButton*    m_mlpBtn      = nullptr;
    QPushButton*    m_rbfBtn      = nullptr;
    QStackedWidget* m_stack       = nullptr;   // page 0 = MLP, page 1 = RBF

    // ── MLP widgets ───────────────────────────────────────────────────────
    QComboBox*      m_taskCombo    = nullptr;
    QSpinBox*       m_layersSpin   = nullptr;
    QDoubleSpinBox* m_dropoutSpin  = nullptr;
    QWidget*        m_layersWidget = nullptr;
    QVBoxLayout*    m_layersLayout = nullptr;
    QPushButton*    m_buildBtn     = nullptr;
    QLabel*         m_summaryLbl   = nullptr;
    QLabel*         m_infoLbl      = nullptr;
    struct LayerRow { QSpinBox* neurons; QComboBox* activation; };
    std::vector<LayerRow> m_layerRows;

    // ── RBF widgets ───────────────────────────────────────────────────────
    QSpinBox*       m_rbfCentersSpin  = nullptr;
    QComboBox*      m_rbfKernelCombo  = nullptr;
    QComboBox*      m_rbfWidthCombo   = nullptr;
    QDoubleSpinBox* m_rbfWidthScale   = nullptr;
    QDoubleSpinBox* m_rbfLambdaSpin   = nullptr;
    QSpinBox*       m_rbfKmeansIter   = nullptr;
    QComboBox*      m_rbfTaskCombo    = nullptr;
    QPushButton*    m_buildRBFBtn     = nullptr;
    QLabel*         m_rbfInfoLbl      = nullptr;

    void buildUi();
    void rebuildLayerRows(int count);
    void updateRBFInfo();
};

} // namespace NeuralStudio
