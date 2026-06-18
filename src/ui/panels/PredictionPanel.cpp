#include "PredictionPanel.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QFrame>
#include <algorithm>

namespace NeuralStudio {

PredictionPanel::PredictionPanel(QWidget* parent) : QWidget(parent) { buildUi(); }

void PredictionPanel::buildUi() {
    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(20, 16, 20, 16);
    root->setSpacing(10);

    auto* title = new QLabel("Manual Prediction");
    { QFont f = title->font(); f.setPointSize(14); f.setBold(true); title->setFont(f); }
    root->addWidget(title);

    m_statusLbl = new QLabel("Train a network first.");
    m_statusLbl->setStyleSheet("QLabel{background:palette(midlight);border-radius:4px;padding:6px 12px;font-size:12px;}");
    root->addWidget(m_statusLbl);

    // Toolbar
    auto* tb = new QHBoxLayout;
    m_loadFirstBtn = new QPushButton("Load 1st Sample");
    m_loadFirstBtn->setFixedHeight(28);
    m_loadFirstBtn->setEnabled(false);
    m_resetBtn = new QPushButton("Reset to Means");
    m_resetBtn->setFixedHeight(28);
    m_resetBtn->setEnabled(false);
    tb->addWidget(m_loadFirstBtn);
    tb->addWidget(m_resetBtn);
    tb->addStretch();
    root->addLayout(tb);

    // Inputs area (scrollable grid)
    auto* group = new QGroupBox("Input Features");
    auto* gLay  = new QVBoxLayout(group);
    auto* scroll = new QScrollArea;
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    m_inputsHost = new QWidget;
    scroll->setWidget(m_inputsHost);
    gLay->addWidget(scroll);
    root->addWidget(group, 1);

    // Predict button + result
    m_predictBtn = new QPushButton("Predict");
    m_predictBtn->setFixedHeight(38);
    m_predictBtn->setEnabled(false);
    m_predictBtn->setStyleSheet(
        "QPushButton:enabled{background:#1D9E75;color:white;border-radius:5px;font-weight:600;font-size:13px;}"
        "QPushButton:disabled{background:palette(midlight);border-radius:5px;}");
    root->addWidget(m_predictBtn);

    m_resultLbl = new QLabel;
    m_resultLbl->setWordWrap(true);
    m_resultLbl->setStyleSheet(
        "QLabel{background:palette(midlight);border-radius:4px;padding:14px 16px;"
        "font-size:13px;font-family:monospace;}");
    m_resultLbl->setMinimumHeight(80);
    m_resultLbl->setText("(no prediction yet)");
    root->addWidget(m_resultLbl);

    connect(m_predictBtn,   &QPushButton::clicked, this, &PredictionPanel::onPredictClicked);
    connect(m_loadFirstBtn, &QPushButton::clicked, this, &PredictionPanel::onLoadFirstSampleClicked);
    connect(m_resetBtn,     &QPushButton::clicked, this, &PredictionPanel::onResetClicked);
}

void PredictionPanel::onDatasetLoaded(const Dataset* ds) {
    m_ds = ds;
    rebuildInputFields();
    m_loadFirstBtn->setEnabled(ds != nullptr);
    m_resetBtn->setEnabled(ds != nullptr);
}

void PredictionPanel::onNetworkBuilt(std::shared_ptr<NeuralNetwork> net) {
    m_net = net;
    m_rbfNet.reset();   // switching to MLP: forget RBF
    m_resultLbl->setText("");
    if (m_net && m_ds) m_statusLbl->setText("Network ready. Enter values and click Predict.");
    m_predictBtn->setEnabled(m_net && m_net->isBuilt());
}

void PredictionPanel::onTrainingCompleted() {
    m_predictBtn->setEnabled(m_net && m_net->isBuilt());
    m_statusLbl->setText("Ready to predict (training complete).");
}

void PredictionPanel::rebuildInputFields() {
    // Clear old
    if (m_inputsHost->layout()) {
        QLayoutItem* it;
        while ((it = m_inputsHost->layout()->takeAt(0))) {
            if (it->widget()) it->widget()->deleteLater();
            delete it;
        }
        delete m_inputsHost->layout();
    }
    m_inputSpins.clear();
    if (!m_ds) return;

    auto* grid = new QGridLayout(m_inputsHost);
    grid->setContentsMargins(8, 8, 8, 8);
    grid->setHorizontalSpacing(16);
    grid->setVerticalSpacing(6);

    // Two-column layout: label/spinbox pairs
    const int n      = m_ds->inputCount;
    const int perCol = (n + 1) / 2;

    for (int i = 0; i < n; ++i) {
        const auto& s = m_ds->inputStats[i];
        auto* lbl = new QLabel(m_ds->inputNames[i] + ":");
        lbl->setMinimumWidth(110);

        auto* spin = new QDoubleSpinBox;
        // Allow values somewhat outside training range for what-if scenarios
        const double margin = std::max(1.0, (s.max - s.min) * 0.5);
        spin->setRange(s.min - margin, s.max + margin);
        spin->setDecimals(4);
        spin->setValue(s.mean);
        spin->setSingleStep(std::max(0.001, (s.max - s.min) / 100.0));
        spin->setMinimumWidth(110);
        spin->setToolTip(QString("Range: [%1, %2]\nMean: %3")
            .arg(s.min, 0, 'f', 3).arg(s.max, 0, 'f', 3).arg(s.mean, 0, 'f', 3));

        const int col = (i < perCol) ? 0 : 2;
        const int row = (i < perCol) ? i : i - perCol;
        grid->addWidget(lbl,  row, col);
        grid->addWidget(spin, row, col + 1);
        m_inputSpins.push_back(spin);
    }
}

void PredictionPanel::onLoadFirstSampleClicked() {
    if (!m_ds || m_ds->sampleCount == 0) return;
    for (int i = 0; i < m_ds->inputCount; ++i)
        m_inputSpins[i]->setValue(m_ds->inputAt(0, i));
    m_statusLbl->setText("Loaded first sample values.");
}

void PredictionPanel::onResetClicked() {
    if (!m_ds) return;
    for (int i = 0; i < m_ds->inputCount; ++i)
        m_inputSpins[i]->setValue(m_ds->inputStats[i].mean);
    m_statusLbl->setText("Reset to column means.");
}

void PredictionPanel::onPredictClicked() {
    // ── RBF prediction (v0.8.0) ───────────────────────────────────────────────
    if (m_rbfNet && m_rbfNet->isBuilt() && !m_inputSpins.empty()) {
        std::vector<double> input(m_inputSpins.size());
        for (size_t i = 0; i < m_inputSpins.size(); ++i)
            input[i] = m_inputSpins[i]->value();
        auto output = m_rbfNet->predict(input);
        QString html = "<b>RBF Prediction Result</b><br><br>";
        if (m_rbfNet->taskType() == TaskType::BinaryClassification) {
            double p = output[0]; int cls = p >= 0.5 ? 1 : 0;
            html += QString("Class: <b>%1</b>  |  P(class=1): <b>%2</b>").arg(cls).arg(p,0,'f',4);
        } else if (m_rbfNet->taskType() == TaskType::MultiClassClassification) {
            int best = int(std::max_element(output.begin(),output.end())-output.begin());
            html += QString("Predicted class: <b>%1</b>  |  P: <b>%2</b>").arg(best).arg(output[best],0,'f',4);
            for (int i=0;i<int(output.size());i++)
                html += QString("<br>Class %1: %2").arg(i).arg(output[i],0,'f',4);
        } else {
            html += QString("Prediction: <b>%1</b>").arg(output[0],0,'f',6);
        }
        m_resultLbl->setText(html);
        return;
    }

    if (!m_net || !m_net->isBuilt() || m_inputSpins.empty()) return;

    std::vector<double> input(m_inputSpins.size());
    for (size_t i = 0; i < m_inputSpins.size(); ++i)
        input[i] = m_inputSpins[i]->value();

    auto output = m_net->predict(input);
    QString html = "<b>Prediction Result</b><br><br>";

    if (m_net->taskType() == TaskType::BinaryClassification) {
        const double p = output[0];
        const int    pred = p >= 0.5 ? 1 : 0;
        const double conf = pred == 1 ? p : (1.0 - p);
        html += QString("Predicted class: <b style='color:%1'>%2</b><br>"
                        "Probability (class=1): <b>%3</b><br>"
                        "Confidence: <b>%4%</b>")
            .arg(pred == 1 ? "#C0392B" : "#1D9E75")
            .arg(pred)
            .arg(p, 0, 'f', 4)
            .arg(conf * 100.0, 0, 'f', 2);

    } else if (m_net->taskType() == TaskType::MultiClassClassification) {
        const auto& cv  = m_net->classValues();
        int bestIdx = static_cast<int>(
            std::max_element(output.begin(), output.end()) - output.begin());
        html += QString("Predicted class: <b style='color:#185fa5'>%1</b><br><br>"
                        "Probabilities per class:<br>")
            .arg(cv[bestIdx], 0, 'f', 2);
        for (size_t i = 0; i < output.size(); ++i) {
            const bool win = ((int)i == bestIdx);
            html += QString("&nbsp;&nbsp;Class %1: %2%3%4<br>")
                .arg(cv[i], 0, 'f', 2)
                .arg(win ? "<b>" : "")
                .arg(QString::number(output[i] * 100.0, 'f', 2) + "%")
                .arg(win ? "</b>" : "");
        }
    } else { // Regression
        html += "Predicted output:<br>";
        for (size_t i = 0; i < output.size(); ++i)
            html += QString("&nbsp;&nbsp;Output %1: <b>%2</b><br>")
                .arg(i + 1).arg(output[i], 0, 'f', 5);
    }

    m_resultLbl->setText(html);
}

// ── v0.8.0: RBF model ────────────────────────────────────────────────────────
void PredictionPanel::onRBFBuilt(std::shared_ptr<RBFNetwork> net) {
    m_rbfNet = net;
    m_net.reset();      // switching to RBF: forget MLP
    m_resultLbl->setText("");
    if (m_ds) m_statusLbl->setText("RBF network ready. Enter values and click Predict.");
    m_predictBtn->setEnabled(true);
}

} // namespace NeuralStudio
