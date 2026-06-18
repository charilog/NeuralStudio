#include "EvaluationPanel.h"
#include "core/dataset/DatasetLoader.h"
#include "utils/Logger.h"

#include <QApplication>
#include <QRegularExpression>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QHeaderView>
#include <QTableWidgetItem>
#include <QFileDialog>
#include <QFileInfo>
#include <QMessageBox>
#include <QSplitter>
#include <cmath>
#include <algorithm>
#include <vector>

namespace NeuralStudio {

EvaluationPanel::EvaluationPanel(QWidget* parent) : QWidget(parent) { buildUi(); }

void EvaluationPanel::buildUi() {
    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(20, 16, 20, 16);
    root->setSpacing(10);

    auto* title = new QLabel("Model Evaluation");
    { QFont f = title->font(); f.setPointSize(14); f.setBold(true); title->setFont(f); }
    root->addWidget(title);

    // Button row
    auto* btnRow = new QHBoxLayout;
    m_evalBtn = new QPushButton("\u2713  Evaluate on Train");
    m_evalBtn->setFixedHeight(36);
    m_evalBtn->setEnabled(true);
    m_evalBtn->setStyleSheet(
        "QPushButton{background:#185fa5;color:white;border-radius:5px;"
        "font-weight:600;font-size:13px;}"
        "QPushButton:hover{background:#1a6fc0;}"
        "QPushButton:pressed{background:#1050a0;}");

    m_loadTestBtn = new QPushButton("Load Test File\u2026");
    m_loadTestBtn->setFixedHeight(36);
    m_loadTestBtn->setEnabled(false);
    m_loadTestBtn->setStyleSheet(
        "QPushButton:enabled{border:1px solid palette(mid);border-radius:5px;padding:0 12px;}"
        "QPushButton:disabled{color:palette(mid);border:1px solid palette(midlight);"
        "border-radius:5px;padding:0 12px;}");

    m_testFileLbl = new QLabel("No test file loaded.");
    m_testFileLbl->setStyleSheet("font-size:11px;color:palette(mid);");

    btnRow->addWidget(m_evalBtn);
    btnRow->addWidget(m_loadTestBtn);
    btnRow->addWidget(m_testFileLbl, 1);
    root->addLayout(btnRow);

    // ── Prominent result banner (replaces the old invisible status label) ─────
    m_statusLbl = new QLabel(
        "Press \u2713 Evaluate on Train to compute metrics.");
    m_statusLbl->setWordWrap(true);
    m_statusLbl->setAlignment(Qt::AlignCenter);
    m_statusLbl->setMinimumHeight(52);
    m_statusLbl->setStyleSheet(
        "QLabel{background:#1C2030;border-radius:6px;border:1px solid #2A3050;"
        "padding:10px 16px;font-size:13px;color:#7080A0;}");
    root->addWidget(m_statusLbl);

    // Tabs
    m_tabs = new QTabWidget;
    m_trainW = buildResultTab("Train Set");
    m_testW  = buildResultTab("Test Set");
    root->addWidget(m_tabs, 1);

    connect(m_evalBtn,     &QPushButton::clicked, this, &EvaluationPanel::onEvaluateClicked);
    connect(m_loadTestBtn, &QPushButton::clicked, this, &EvaluationPanel::onLoadTestClicked);
}

EvaluationPanel::SetWidgets EvaluationPanel::buildResultTab(const QString& tabName) {
    SetWidgets w;
    auto* tab  = new QWidget;
    auto* vlay = new QVBoxLayout(tab);
    vlay->setContentsMargins(0, 8, 0, 0);
    vlay->setSpacing(8);

    w.metricsLbl = new QLabel;
    w.metricsLbl->setWordWrap(true);
    w.metricsLbl->setStyleSheet("font-family:monospace;font-size:12px;padding:6px 2px;");
    vlay->addWidget(w.metricsLbl);

    auto* cmGroup = new QGroupBox("Confusion Matrix");
    auto* cmLay   = new QVBoxLayout(cmGroup);
    w.confMatrix  = new QTableWidget;
    w.confMatrix->setEditTriggers(QAbstractItemView::NoEditTriggers);
    w.confMatrix->setMaximumHeight(160);
    w.confMatrix->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    // Fixed row heights so the matrix stays compact regardless of cell content
    w.confMatrix->verticalHeader()->setSectionResizeMode(QHeaderView::Fixed);
    w.confMatrix->verticalHeader()->setDefaultSectionSize(36);
    cmLay->addWidget(w.confMatrix);
    vlay->addWidget(cmGroup);

    auto* predGroup = new QGroupBox("Predictions (first 100 samples)");
    auto* predLay   = new QVBoxLayout(predGroup);
    w.predTable = new QTableWidget;
    w.predTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    w.predTable->setAlternatingRowColors(true);
    w.predTable->setStyleSheet(
        "QTableWidget { background:#1f1f2a; alternate-background-color:#262635; "
        "gridline-color:#3a3a4a; }");
    w.predTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    w.predTable->verticalHeader()->setDefaultSectionSize(22);
    predLay->addWidget(w.predTable);
    vlay->addWidget(predGroup, 1);

    m_tabs->addTab(tab, tabName);
    return w;
}

// ─── Slot handlers ────────────────────────────────────────────────────────────
void EvaluationPanel::onDatasetLoaded(const Dataset* ds)              { m_trainDs = ds; }
void EvaluationPanel::onNetworkBuilt(std::shared_ptr<NeuralNetwork> net) {
    m_net = net;
    m_rbfNet.reset();   // switching to MLP: forget any RBF state
    clearResults();
    m_statusLbl->setText("Network built \u2014 train it first, then evaluate.");
    m_statusLbl->setStyleSheet(
        "QLabel{background:#1C2030;border-radius:6px;border:1px solid #2A3050;"
        "padding:10px 16px;font-size:13px;color:#7080A0;}");
}

void EvaluationPanel::onTestDatasetLoaded(const Dataset* testDs) {
    if (!testDs) {
        m_testDs.reset();
        m_testFileLbl->setText("No companion test file found.");
        m_testFileLbl->setStyleSheet("font-size:11px;color:palette(mid);");
        return;
    }
    // Take a non-owning shared_ptr (aliasing constructor would be cleaner but
    // the DataPanel owns the dataset; we just hold a pointer-like reference).
    // Simplest: store a shared_ptr with no-op deleter.
    m_testDs = std::shared_ptr<Dataset>(const_cast<Dataset*>(testDs), [](Dataset*){});
    m_testFileLbl->setText(
        QString("✓ Auto-loaded: %1 (%2 samples)")
            .arg(QFileInfo(testDs->sourcePath).fileName())
            .arg(testDs->sampleCount));
    m_testFileLbl->setStyleSheet("font-size:11px;color:#1D9E75;font-weight:600;");

    // If a trained model already exists, evaluate immediately
    if (m_rbfNet && m_rbfNet->isBuilt())
        evaluateRBFOnSet(m_testDs.get(), m_testW, "Test Set");
    else if (m_net && m_net->isBuilt())
        evaluateOnSet(m_testDs.get(), m_testW, "Test Set");
}

void EvaluationPanel::clearResults() {
    for (auto* w : {&m_trainW, &m_testW}) {
        w->metricsLbl->setText("\u2014");
        w->predTable->clearContents();
        w->predTable->setRowCount(0);
        w->predTable->setColumnCount(0);
        w->confMatrix->clearContents();
        w->confMatrix->setRowCount(0);
        w->confMatrix->setColumnCount(0);
    }
}

void EvaluationPanel::onTrainingCompleted() {
    m_loadTestBtn->setEnabled(true);
    onEvaluateClicked();   // auto-evaluate immediately after training
}

void EvaluationPanel::onEvaluateClicked() {
    // ── Guard checks ──────────────────────────────────────────────────────────
    if (!m_trainDs) {
        m_statusLbl->setText("\u26A0  Load a dataset first.");
    m_statusLbl->setStyleSheet(
        "QLabel{background:#3D1616;border-radius:6px;border:1px solid #703030;"
        "padding:10px 16px;font-size:13px;font-weight:600;color:#FF8080;}");
        return;
    }

    const bool hasRBF = m_rbfNet && m_rbfNet->isBuilt();
    const bool hasMLP = m_net  && m_net->isBuilt();

    if (!hasRBF && !hasMLP) {
        m_statusLbl->setText("\u26A0  Build and train a network first.");
    m_statusLbl->setStyleSheet(
        "QLabel{background:#3D1616;border-radius:6px;border:1px solid #703030;"
        "padding:10px 16px;font-size:13px;font-weight:600;color:#FF8080;}");
        return;
    }

    // ── RBF evaluation ────────────────────────────────────────────────────────
    if (hasRBF) {
        m_statusLbl->setText("Evaluating RBF\u2026");
    m_statusLbl->setStyleSheet(
        "QLabel{background:#1A2A3D;border-radius:6px;border:1px solid #2A4060;"
        "padding:10px 16px;font-size:13px;font-weight:600;color:#5BA0E0;}");
        evaluateRBFOnSet(m_trainDs, m_trainW, "Train Set");
        if (m_testDs) evaluateRBFOnSet(m_testDs.get(), m_testW, "Test Set");
        m_tabs->setCurrentIndex(0);
        return;
    }

    // ── MLP evaluation ────────────────────────────────────────────────────────
    m_statusLbl->setText("Evaluating MLP\u2026");
    m_statusLbl->setStyleSheet(
        "QLabel{background:#1A2A3D;border-radius:6px;border:1px solid #2A4060;"
        "padding:10px 16px;font-size:13px;font-weight:600;color:#5BA0E0;}");
    if (!m_net) {   // extra safety
        m_statusLbl->setText("\u26A0  MLP network is null — rebuild and retrain.");
        return;
    }
    evaluateOnSet(m_trainDs, m_trainW, "Train Set");
    if (m_testDs) evaluateOnSet(m_testDs.get(), m_testW, "Test Set");
    m_tabs->setCurrentIndex(0);
}

void EvaluationPanel::onLoadTestClicked() {
    const QString path = QFileDialog::getOpenFileName(
        this, "Open Test Dataset", {}, DatasetLoader::fileDialogFilter());
    if (path.isEmpty()) return;
    try {
        m_testDs = DatasetLoader::load(path);
        m_testFileLbl->setText(QFileInfo(path).fileName() +
            QString("  (%1 samples)").arg(m_testDs->sampleCount));
        m_testFileLbl->setStyleSheet("font-size:11px;color:#1D9E75;");
        if (m_net && m_net->isBuilt())
            evaluateOnSet(m_testDs.get(), m_testW, "Test Set");
        m_tabs->setCurrentIndex(1);
    } catch (const std::exception& ex) {
        QMessageBox::critical(this, "Load Error", ex.what());
    }
}

// ─── Dispatcher ──────────────────────────────────────────────────────────────
void EvaluationPanel::evaluateOnSet(const Dataset* ds, SetWidgets& w, const QString& title) {
    auto setErr = [this](const QString& msg) {
        m_statusLbl->setText(msg);
        m_statusLbl->setStyleSheet(
            "QLabel{background:#3D1616;border-radius:6px;border:1px solid #703030;"
            "padding:12px 16px;font-size:13px;font-weight:600;color:#FF8080;}");
    };
    if (!ds)    { setErr("\u26A0  No dataset loaded.");                        return; }
    if (!m_net) { setErr("\u26A0  No MLP network \u2014 rebuild and retrain."); return; }

    switch (m_net->taskType()) {
        case TaskType::BinaryClassification:     evaluateClassification(ds, w); break;
        case TaskType::MultiClassClassification: evaluateMultiClass(ds, w);     break;
        default:                                 evaluateRegression(ds, w);     break;
    }

    // Show key metrics directly in the banner (strip HTML from metricsLbl)
    QString metrics = w.metricsLbl->text();
    metrics.remove(QRegularExpression("<[^>]*>")).replace("&nbsp;", " ").replace("&sup2;","\u00B2");
    metrics = metrics.simplified();

    m_statusLbl->setTextFormat(Qt::RichText);
    m_statusLbl->setText(
        "<b>\u2713  " + title + " \u2014 " + QString::number(ds->sampleCount) + " samples</b>"
        "<br><span style=\'font-size:12px;font-weight:normal;color:#80C0A0;\'>" + metrics + "</span>");
    m_statusLbl->setStyleSheet(
        "QLabel{background:#163D2B;border-radius:6px;border:1px solid #2A6040;"
        "padding:12px 16px;font-size:14px;font-weight:600;color:#4EDDAD;}");
}

// ─── Binary classification ────────────────────────────────────────────────────
void EvaluationPanel::evaluateClassification(const Dataset* ds, SetWidgets& w) {
    const int n = ds->sampleCount;
    const double mid = 0.5 * (m_net->outputMin() + m_net->outputMax());
    int tp=0,tn=0,fp=0,fn=0;

    const int show = std::min(n, 100);
    w.predTable->clearContents();
    w.predTable->setRowCount(show);
    w.predTable->setColumnCount(3);
    w.predTable->setHorizontalHeaderLabels({"Target","Predicted","Correct?"});

    for (int r = 0; r < n; ++r) {
        int tc = (ds->outputAt(r,0) > mid) ? 1 : 0;
        auto pred = m_net->predict(ds->inputRow(r));
        int pc    = (pred[0] >= 0.5) ? 1 : 0;
        if (tc==1&&pc==1) ++tp; else if(tc==0&&pc==0) ++tn;
        else if(tc==0&&pc==1) ++fp; else ++fn;
        if (r < show) {
            bool ok = (tc==pc);
            auto* t0 = new QTableWidgetItem(QString::number(tc)); t0->setTextAlignment(Qt::AlignCenter);
            auto* t1 = new QTableWidgetItem(QString("%1 (p=%2)").arg(pc).arg(pred[0],0,'f',3)); t1->setTextAlignment(Qt::AlignCenter);
            auto* t2 = new QTableWidgetItem(ok?"✓":"✗"); t2->setTextAlignment(Qt::AlignCenter);
            t2->setBackground(ok ? QColor(200,240,210) : QColor(255,210,210));
            t2->setForeground(QColor(20, 40, 30));
            w.predTable->setItem(r,0,t0); w.predTable->setItem(r,1,t1); w.predTable->setItem(r,2,t2);
        }
    }
    double acc  = (tp+tn)/(double)n;
    double prec = (tp+fp)>0 ? tp/(double)(tp+fp) : 0;
    double rec  = (tp+fn)>0 ? tp/(double)(tp+fn) : 0;
    double f1   = (prec+rec)>0 ? 2*prec*rec/(prec+rec) : 0;

    w.metricsLbl->setText(QString("Accuracy: <b>%1%</b>  &nbsp; Precision: <b>%2</b>  &nbsp; Recall: <b>%3</b>  &nbsp; F1: <b>%4</b>  &nbsp; (N=%5)")
        .arg(acc*100,0,'f',2).arg(prec,0,'f',4).arg(rec,0,'f',4).arg(f1,0,'f',4).arg(n));

    w.confMatrix->setRowCount(2); w.confMatrix->setColumnCount(2);
    w.confMatrix->setHorizontalHeaderLabels({"Pred:0","Pred:1"});
    w.confMatrix->setVerticalHeaderLabels({"True:0","True:1"});
    auto cm=[&](int r,int c,int v,bool d){
        auto* it=new QTableWidgetItem(QString::number(v));
        it->setTextAlignment(Qt::AlignCenter);
        QFont f=it->font(); f.setBold(true); f.setPointSize(13); it->setFont(f);
        const QBrush bgB(d ? QColor(200,240,210) : QColor(255,220,220));
        const QBrush fgB(QColor(20, 40, 30));
        it->setBackground(bgB);
        it->setData(Qt::BackgroundRole, bgB);
        it->setForeground(fgB);
        it->setData(Qt::ForegroundRole, fgB);
        w.confMatrix->setItem(r,c,it);
    };
    cm(0,0,tn,true); cm(0,1,fp,false); cm(1,0,fn,false); cm(1,1,tp,true);
}

// ─── Multi-class ──────────────────────────────────────────────────────────────
void EvaluationPanel::evaluateMultiClass(const Dataset* ds, SetWidgets& w) {
    const int n = ds->sampleCount;
    const auto& cv = m_net->classValues();
    const int nc   = static_cast<int>(cv.size());

    // nClasses × nClasses confusion matrix
    std::vector<std::vector<int>> cm(nc, std::vector<int>(nc, 0));
    int correct = 0;

    const int show = std::min(n, 100);
    w.predTable->clearContents();
    w.predTable->setRowCount(show);
    w.predTable->setColumnCount(3);
    w.predTable->setHorizontalHeaderLabels({"True Class","Pred Class","Correct?"});

    for (int r = 0; r < n; ++r) {
        int tc   = m_net->classIndex(ds->outputAt(r, 0));
        auto prob = m_net->predict(ds->inputRow(r));
        int pc   = static_cast<int>(std::max_element(prob.begin(), prob.end()) - prob.begin());
        cm[tc][pc]++;
        if (tc == pc) ++correct;

        if (r < show) {
            bool ok = (tc==pc);
            auto* t0=new QTableWidgetItem(QString::number(cv[tc],'f',1)); t0->setTextAlignment(Qt::AlignCenter);
            auto* t1=new QTableWidgetItem(QString("%1 (p=%2)").arg(cv[pc],0,'f',1).arg(prob[pc],0,'f',3)); t1->setTextAlignment(Qt::AlignCenter);
            auto* t2=new QTableWidgetItem(ok?"✓":"✗"); t2->setTextAlignment(Qt::AlignCenter);
            t2->setBackground(ok?QColor(200,240,210):QColor(255,210,210));
            t2->setForeground(QColor(20, 40, 30));
            w.predTable->setItem(r,0,t0); w.predTable->setItem(r,1,t1); w.predTable->setItem(r,2,t2);
        }
    }

    double acc = correct/(double)n;
    w.metricsLbl->setText(QString("Accuracy: <b>%1%</b>  &nbsp; Correct: <b>%2/%3</b>  &nbsp; Classes: <b>%4</b>")
        .arg(acc*100,0,'f',2).arg(correct).arg(n).arg(nc));

    // Confusion matrix
    w.confMatrix->setRowCount(nc); w.confMatrix->setColumnCount(nc);
    QStringList hdrs;
    for (int i=0;i<nc;++i) hdrs << QString("Pred:%1").arg(cv[i],0,'f',1);
    w.confMatrix->setHorizontalHeaderLabels(hdrs);
    QStringList vhdrs;
    for (int i=0;i<nc;++i) vhdrs << QString("True:%1").arg(cv[i],0,'f',1);
    w.confMatrix->setVerticalHeaderLabels(vhdrs);

    for (int r=0;r<nc;++r) for (int c=0;c<nc;++c) {
        auto* it=new QTableWidgetItem(QString::number(cm[r][c]));
        it->setTextAlignment(Qt::AlignCenter);
        QFont f=it->font(); f.setBold(r==c); f.setPointSize(12); it->setFont(f);
        QColor bg = (r==c) ? QColor(200,240,210)
                          : (cm[r][c]>0 ? QColor(255,220,220) : QColor());
        if (bg.isValid()) {
            const QBrush bgB(bg), fgB(QColor(20, 40, 30));
            it->setBackground(bgB);
            it->setData(Qt::BackgroundRole, bgB);
            it->setForeground(fgB);
            it->setData(Qt::ForegroundRole, fgB);
        }
        w.confMatrix->setItem(r,c,it);
    }
}

// ─── Regression ──────────────────────────────────────────────────────────────
void EvaluationPanel::evaluateRegression(const Dataset* ds, SetWidgets& w) {
    const int n = ds->sampleCount;
    const int show = std::min(n,100);
    w.predTable->clearContents();
    w.predTable->setRowCount(show);
    w.predTable->setColumnCount(3);
    w.predTable->setHorizontalHeaderLabels({"Target","Predicted","Error"});

    double ssRes=0,ssTot=0,sumT=0;
    for(int r=0;r<n;++r) sumT+=ds->outputAt(r,0);
    double meanT=sumT/n, mse=0;
    for(int r=0;r<n;++r){
        double t=ds->outputAt(r,0);
        auto pred=m_net->predict(ds->inputRow(r));
        double err=pred[0]-t;
        mse+=err*err; ssRes+=err*err; ssTot+=(t-meanT)*(t-meanT);
        if(r<show){
            auto* a=new QTableWidgetItem(QString::number(t,   'f',4)); a->setTextAlignment(Qt::AlignRight|Qt::AlignVCenter);
            auto* b=new QTableWidgetItem(QString::number(pred[0],'f',4)); b->setTextAlignment(Qt::AlignRight|Qt::AlignVCenter);
            auto* c=new QTableWidgetItem(QString::number(err,  'f',4)); c->setTextAlignment(Qt::AlignRight|Qt::AlignVCenter);
            w.predTable->setItem(r,0,a); w.predTable->setItem(r,1,b); w.predTable->setItem(r,2,c);
        }
    }
    mse/=n;
    double rmse=std::sqrt(mse), r2=(ssTot>1e-12)?1.0-ssRes/ssTot:0.0;
    w.metricsLbl->setText(QString("MSE: <b>%1</b>  &nbsp; RMSE: <b>%2</b>  &nbsp; R\u00B2: <b>%3</b>  &nbsp; (N=%4)")
        .arg(mse,0,'f',6).arg(rmse,0,'f',6).arg(r2,0,'f',4).arg(n));
    w.confMatrix->setRowCount(0); // hide confusion matrix for regression
}

// ── v0.8.0: RBF model ────────────────────────────────────────────────────────
void EvaluationPanel::onRBFBuilt(std::shared_ptr<RBFNetwork> net) {
    m_rbfNet = net;
    m_net.reset();      // switching to RBF: forget any MLP state
    clearResults();
    m_statusLbl->setText("RBF network built \u2014 train it first, then evaluate.");
    m_statusLbl->setStyleSheet(
        "QLabel{background:#1C2030;border-radius:6px;border:1px solid #2A3050;"
        "padding:10px 16px;font-size:13px;color:#7080A0;}");
}

void EvaluationPanel::evaluateRBFOnSet(const Dataset* ds, SetWidgets& w, const QString& title) {
    if (!ds || !m_rbfNet || !m_rbfNet->isBuilt()) return;

    const int n     = ds->sampleCount;
    const int nOut  = m_rbfNet->nOutputs();
    const TaskType task = m_rbfNet->taskType();

    // ── Setup table BEFORE the loop ───────────────────────────────────────────
    w.predTable->clearContents();
    const int show = std::min(n, 100);
    w.predTable->setRowCount(show);

    if (task == TaskType::Regression) {
        const int cols = std::min(nOut, 3) * 2;  // Target+Predicted per output (max 3 outputs shown)
        QStringList hdr;
        for (int o = 0; o < std::min(nOut, 3); ++o)
            hdr << QString("Target%1").arg(nOut>1?QString::number(o):"")
                << QString("Pred%1").arg(nOut>1?QString::number(o):"");
        w.predTable->setColumnCount(cols);
        w.predTable->setHorizontalHeaderLabels(hdr);
    } else {
        w.predTable->setColumnCount(3);
        w.predTable->setHorizontalHeaderLabels({"Target", "Predicted", "Correct?"});
    }

    // ── Loop ─────────────────────────────────────────────────────────────────
    double totalSqErr = 0.0;
    int correct = 0;
    const int nClasses = (task == TaskType::BinaryClassification) ? 2 : nOut;
    // Confusion matrix accumulator (nClasses × nClasses)
    std::vector<std::vector<int>> confData(nClasses, std::vector<int>(nClasses, 0));

    for (int r = 0; r < n; ++r) {
        auto pred = m_rbfNet->predict(ds->inputs[r]);

        if (task == TaskType::Regression) {
            for (int o = 0; o < nOut; ++o) {
                double e = pred[o] - ds->outputs[r][o];
                totalSqErr += e * e;
            }
            if (r < show) {
                for (int o = 0; o < std::min(nOut, 3); ++o) {
                    int col = o * 2;
                    w.predTable->setItem(r, col,   new QTableWidgetItem(
                        QString::number(ds->outputs[r][o], 'f', 4)));
                    w.predTable->setItem(r, col+1, new QTableWidgetItem(
                        QString::number(pred[o], 'f', 4)));
                }
            }
        } else {
            // Classification: compute predicted and true class index
            int pc, tc;
            if (nOut == 1) {
                pc = (pred[0] >= 0.5) ? 1 : 0;
                tc = (ds->outputs[r][0] >= 0.5) ? 1 : 0;
            } else {
                pc = int(std::max_element(pred.begin(), pred.end()) - pred.begin());
                tc = int(std::max_element(ds->outputs[r].begin(), ds->outputs[r].end())
                         - ds->outputs[r].begin());
            }
            if (pc == tc) ++correct;
            // Accumulate confusion matrix
            if (tc >= 0 && tc < nClasses && pc >= 0 && pc < nClasses)
                confData[tc][pc]++;
            for (int o = 0; o < nOut; ++o) {
                double e = pred[o] - ds->outputs[r][o];
                totalSqErr += e * e;
            }
            if (r < show) {
                bool ok = (pc == tc);
                auto* t0 = new QTableWidgetItem(QString::number(tc)); t0->setTextAlignment(Qt::AlignCenter);
                auto* t1 = new QTableWidgetItem(QString::number(pc)); t1->setTextAlignment(Qt::AlignCenter);
                auto* t2 = new QTableWidgetItem(ok ? "\u2713" : "\u2717"); t2->setTextAlignment(Qt::AlignCenter);
                t2->setForeground(ok ? QColor("#1D9E75") : QColor("#C0392B"));
                w.predTable->setItem(r, 0, t0);
                w.predTable->setItem(r, 1, t1);
                w.predTable->setItem(r, 2, t2);
            }
        }
    }

    // ── Confusion Matrix (classification only) ────────────────────────────────
    if (task != TaskType::Regression) {
        w.confMatrix->setRowCount(nClasses);
        w.confMatrix->setColumnCount(nClasses);
        QStringList hdrs;
        for (int c = 0; c < nClasses; ++c) hdrs << QString("Pred:%1").arg(c);
        w.confMatrix->setHorizontalHeaderLabels(hdrs);
        QStringList vhdrs;
        for (int c = 0; c < nClasses; ++c) vhdrs << QString("True:%1").arg(c);
        w.confMatrix->setVerticalHeaderLabels(vhdrs);
        w.confMatrix->horizontalHeader()->show();
        w.confMatrix->verticalHeader()->show();
        for (int tr = 0; tr < nClasses; ++tr) {
            for (int tc2 = 0; tc2 < nClasses; ++tc2) {
                auto* it = new QTableWidgetItem(QString::number(confData[tr][tc2]));
                it->setTextAlignment(Qt::AlignCenter);
                bool onDiag = (tr == tc2);
                it->setBackground(onDiag ? QColor("#163D2B") : QColor("#2A1A1A"));
                it->setForeground(onDiag ? QColor("#4EDDAD") : QColor("#E07070"));
                w.confMatrix->setItem(tr, tc2, it);
            }
        }
    } else {
        // Regression: clear confusion matrix
        w.confMatrix->setRowCount(0);
        w.confMatrix->setColumnCount(0);
    }

    // ── Summary ───────────────────────────────────────────────────────────────
    double mse = totalSqErr / (double(n) * nOut);   // mean over samples AND outputs
    QString summary;
    if (task == TaskType::Regression) {
        summary = QString("MSE: %1  |  RMSE: %2  |  %3 outputs  |  RBF k=%4")
            .arg(mse, 0,'f',6).arg(std::sqrt(mse), 0,'f',6)
            .arg(nOut).arg(m_rbfNet->nCenters());
    } else {
        double acc = double(correct) / n;
        summary = QString("Accuracy: %1%  |  Correct: %2/%3  |  RBF k=%4")
            .arg(acc * 100, 0,'f',2).arg(correct).arg(n).arg(m_rbfNet->nCenters());
    }
    w.metricsLbl->setText(summary);

    // Show metrics directly in the banner
    m_statusLbl->setTextFormat(Qt::RichText);
    m_statusLbl->setText(
        "<b>\u2713  " + title + " \u2014 " + QString::number(n) + " samples  (RBF k="
        + QString::number(m_rbfNet->nCenters()) + ")</b>"
        "<br><span style='font-size:12px;font-weight:normal;color:#80C0A0;'>"
        + summary.remove(QRegularExpression("<[^>]*>")).replace("&nbsp;"," ").simplified()
        + "</span>");
    m_statusLbl->setStyleSheet(
        "QLabel{background:#163D2B;border-radius:6px;border:1px solid #2A6040;"
        "padding:12px 16px;font-size:14px;font-weight:600;color:#4EDDAD;}");
}

} // namespace NeuralStudio
