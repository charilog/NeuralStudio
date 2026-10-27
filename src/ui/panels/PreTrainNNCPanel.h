#pragma once
#include <QCheckBox>
// PreTrainNNCPanel.h — v5: N-runs (mean ± std), version 1.0.0
#include "core/pretrain/PreTrainWorker.h"
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QTabWidget>
#include <QLabel>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QComboBox>
#include <QLineEdit>
#include <QPushButton>
#include <QProgressBar>
#include <QGroupBox>
#include <QTableWidget>
#include <QScrollArea>
#include <QThread>
#include <QPointer>
#include <QStringList>
#include <QVector>
#include <QMap>
#include <QList>
#include <QElapsedTimer>
#include <QTimer>
#include <vector>
#include <memory>

QT_BEGIN_NAMESPACE
class QChart; class QChartView; class QLineSeries; class QValueAxis;
class QDragEnterEvent; class QDropEvent;
QT_END_NAMESPACE

namespace NeuralStudio {

// ── PhaseBox: reusable config widget for one training phase ──────────────────
class PhaseBox : public QGroupBox {
    Q_OBJECT
public:
    explicit PhaseBox(int num, bool allowNone, QWidget* parent = nullptr);
    PhaseConfig config() const;
    static void fillOptimizerCombo(QComboBox* c, bool allowNone);
private:
    QComboBox*      m_optCombo = nullptr;
    QSpinBox*       m_popSpin  = nullptr;
    QSpinBox*       m_iterSpin = nullptr;
    QDoubleSpinBox* m_wbSpin   = nullptr;
};

// ── Panel ─────────────────────────────────────────────────────────────────────
class PreTrainNNCPanel : public QWidget {
    Q_OBJECT
public:
    explicit PreTrainNNCPanel(QWidget* parent = nullptr);

    // Project save/load: dataset queue access
    QStringList queuePaths() const { return m_queue; }
    void setQueuePaths(const QStringList& paths) {
        if (m_running) return;
        m_queue.clear(); addToQueue(paths);
    }

protected:
    void dragEnterEvent(QDragEnterEvent* e) override;
    void dropEvent     (QDropEvent* e)       override;

signals:
    void runningStateChanged(bool running);
    void trainingCompleted(std::shared_ptr<NeuralNetwork> net,
                           std::shared_ptr<Dataset>       ds);

private slots:
    void onAddFiles();
    void onClearQueue();
    void onRemoveSelected();
    void onRunClicked();
    void onStopClicked();
    // Worker
    void onDatasetStarted  (int dsIdx, int totalDs, QString dsName);
    void onPhaseStarted    (int phase, int total, QString name);
    void onIterationDone   (int phase, int iter, int total, double loss, double vl);
    void onPhaseFinished   (int phase, double finalLoss);
    void onDatasetFinished (QString dsName, QStringList configs,
                            QVector<double> trainLoss, QVector<double> trainErr,
                            QVector<double> valErr, QVector<double> testErr, bool isCls);
    void onAllDone         (std::shared_ptr<NeuralNetwork> net,
                            std::shared_ptr<Dataset> ds);
    void onError           (QString msg);
    // Exports
    void onExportLossPng();
    void onExportEvalXlsx();
    void onExportCompXlsx();
    void onExportBarPng();

private:
    // ── Dataset queue ──────────────────────────────────────────────────────
    QStringList   m_queue;
    QTableWidget* m_queueTable  = nullptr;
    QPushButton*  m_addBtn      = nullptr;
    QPushButton*  m_clearBtn    = nullptr;
    QPushButton*  m_removeBtn   = nullptr;

    // ── Config widgets ─────────────────────────────────────────────────────
    QGroupBox*      m_netGrp     = nullptr;   // "Network & Training" box (right of queue)
    QLineEdit*      m_hiddenEdit = nullptr;
    QComboBox*      m_actCombo   = nullptr;
    QDoubleSpinBox* m_valSpin    = nullptr;
    QSpinBox*       m_iwSpin     = nullptr;
    QCheckBox*      m_equalChk   = nullptr;   // baseline with the same total budget
    PhaseBox*       m_ph1        = nullptr;
    PhaseBox*       m_ph2        = nullptr;
    PhaseBox*       m_ph3        = nullptr;

    // ── Progress ──────────────────────────────────────────────────────────
    QPushButton*               m_runBtn            = nullptr;
    QPushButton*               m_stopBtn           = nullptr;
    QSpinBox*                  m_nRunsSpin         = nullptr;
    QLabel*                    m_phaseNameLbl      = nullptr;
    QLabel*                    m_statusLbl         = nullptr;
    QWidget*                   m_progressContainer = nullptr;
    std::vector<QProgressBar*> m_bars;
    std::vector<QLabel*>       m_barLbls;
    std::vector<int>           m_phaseIters;
    std::vector<int>           m_xOff;

    // ── Tab 0: Loss ────────────────────────────────────────────────────────
    QTabWidget*           m_tabs          = nullptr;
    QVBoxLayout*          m_lossLayout    = nullptr;
    QScrollArea*          m_lossScroll    = nullptr;
    QList<QChart*>        m_lossCharts;
    QList<QChartView*>    m_lossViews;
    QChart*               m_curLossChart  = nullptr;
    QValueAxis*           m_curLossAxisX  = nullptr;
    QValueAxis*           m_curLossAxisY  = nullptr;
    QList<QLineSeries*>   m_curLossSeries;

    // ── Tab 1: Evaluation ─────────────────────────────────────────────────
    QTableWidget*  m_evalTable = nullptr;

    // ── Tab 2: Comparison ─────────────────────────────────────────────────
    QTableWidget*  m_compTable = nullptr;

    // ── Tab 3: Bar chart ─────────────────────────────────────────────────
    QChart*        m_barChart  = nullptr;
    QChartView*    m_barView   = nullptr;
    QScrollArea*   m_barScroll = nullptr;
    struct BarEntry {
        QString     dsName;
        QStringList configs;
        QVector<double> trainErr;
        bool isClassification = false;
    };
    QList<BarEntry> m_barEntries;

    // ── N-runs accumulator ────────────────────────────────────────────────
    struct DsAccum {
        QStringList              configs;
        bool                     isCls   = false;
        QVector<QVector<double>> trainErr;    // [cfgIdx][runIdx]
        QVector<QVector<double>> valErr;
        QVector<QVector<double>> trainLoss;
        QVector<QVector<double>> testErr;     // companion test file (-1 = none)
        QVector<int>             fromStd;     // runs in which this phase started from the Standard weights
    };
    QMap<QString, DsAccum> m_accumResults;

    // ── State ─────────────────────────────────────────────────────────────
    bool    m_running       = false;
    int     m_dsDone        = 0;
    int     m_currentRun    = 0;
    int     m_totalRuns     = 1;
    bool    m_stopRequested = false;
    int     m_activePhases  = 3;
    int     m_currentDsIdx  = 0;
    int     m_totalDs       = 0;
    QString m_currentDsName;

    QElapsedTimer   m_clock;
    QTimer*         m_tickTimer = nullptr;
    QLabel*         m_timeLbl   = nullptr;

    PreTrainConfig                m_lastConfig;
    std::shared_ptr<NeuralNetwork> m_lastNet;
    std::shared_ptr<Dataset>       m_lastDs;

    QPointer<QThread>        m_thread;
    QPointer<PreTrainWorker> m_worker;

    // ── Helpers ───────────────────────────────────────────────────────────
    void buildUi();
    void buildUi_queueAndNetwork(QVBoxLayout* root);   // queue + network side by side
    QGroupBox* buildQueueGroup ();               // returns the box; caller places it
    void buildNetworkGroup();                    // builds m_netGrp
    void buildConfigArea (QVBoxLayout* root);
    void buildTabWidget  (QVBoxLayout* root);
    void addToQueue(const QStringList& paths);
    void refreshQueueTable();
    void rebuildProgress(int nPhases);
    void startNewLossChart(const QString& dsName, int nPhases);
    void clearLossCharts();
    void styleChart(QChart* c, bool dark);
    void onTimerTick();
    void setRunning(bool r);
    void rebuildBarChart();

    PreTrainConfig buildConfig()  const;
    void           launchWorker(const PreTrainConfig& cfg);
    void           finalizeAllRuns();
    void           refreshInfoLabel(const QString& phaseInfo);

    bool exportStackedPng(const QList<QChart*>& charts,
                          QWidget* parent, const QString& defaultName);
};

} // namespace NeuralStudio
