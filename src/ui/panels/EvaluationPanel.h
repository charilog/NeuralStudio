#pragma once
#include "core/nn/NeuralNetwork.h"
#include "core/nn/RBFNetwork.h"
#include "core/dataset/Dataset.h"
#include <QWidget>
#include <QLabel>
#include <QTableWidget>
#include <QPushButton>
#include <QTabWidget>
#include <memory>

namespace NeuralStudio {

class EvaluationPanel : public QWidget {
    Q_OBJECT
public:
    explicit EvaluationPanel(QWidget* parent = nullptr);

public slots:
    void onNetworkBuilt(std::shared_ptr<NeuralNetwork> net);
    void onRBFBuilt(std::shared_ptr<RBFNetwork> net);      // v0.8.0
    void onDatasetLoaded(const Dataset* ds);
    void onTestDatasetLoaded(const Dataset* testDs);  // auto from companion
    void onTrainingCompleted();

private slots:
    void onEvaluateClicked();
    void onLoadTestClicked();

private:
    std::shared_ptr<NeuralNetwork> m_net;
    std::shared_ptr<RBFNetwork>    m_rbfNet;   // v0.8.0
    const Dataset* m_trainDs = nullptr;
    std::shared_ptr<Dataset> m_testDs;

    // ── Widgets ──────────────────────────────────────────────────────────────
    QPushButton* m_evalBtn       = nullptr;
    QPushButton* m_loadTestBtn   = nullptr;
    QLabel*      m_statusLbl     = nullptr;
    QLabel*      m_testFileLbl   = nullptr;
    QTabWidget*  m_tabs          = nullptr;

    // Per-tab contents (train and test use same widget types)
    struct SetWidgets {
        QLabel*       metricsLbl  = nullptr;
        QTableWidget* confMatrix  = nullptr;
        QTableWidget* predTable   = nullptr;
    };
    SetWidgets m_trainW, m_testW;

    void buildUi();
    SetWidgets buildResultTab(const QString& tabName);
    void evaluateOnSet(const Dataset* ds, SetWidgets& w, const QString& title);
    void evaluateRBFOnSet(const Dataset* ds, SetWidgets& w, const QString& title);
    void clearResults();   // resets all result tables/labels when network changes
    void evaluateClassification(const Dataset* ds, SetWidgets& w);
    void evaluateMultiClass    (const Dataset* ds, SetWidgets& w);
    void evaluateRegression    (const Dataset* ds, SetWidgets& w);
};

} // namespace NeuralStudio
