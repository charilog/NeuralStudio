#pragma once
// EvolBoundsPanel.h — v8: N-runs (mean ± std), version 1.0.0
#include <QElapsedTimer>
#include <QTimer>
#include "core/bounds/EvolBoundsWorker.h"
#include "ui/panels/PreTrainNNCPanel.h"
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QTabWidget>
#include <QLabel>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QComboBox>
#include <QLineEdit>
#include <QCheckBox>
#include <QPushButton>
#include <QProgressBar>
#include <QStackedWidget>
#include <QGroupBox>
#include <QTableWidget>
#include <QScrollArea>
#include <QThread>
#include <QPointer>
#include <QStringList>
#include <QVector>
#include <QMap>
#include <QList>
#include <vector>
#include <memory>

QT_BEGIN_NAMESPACE
class QChart; class QChartView;
class QLineSeries; class QAreaSeries; class QValueAxis;
class QDragEnterEvent; class QDropEvent;
QT_END_NAMESPACE

namespace NeuralStudio {

class EvolBoundsPanel : public QWidget {
    Q_OBJECT
public:
    explicit EvolBoundsPanel(QWidget* parent = nullptr);

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
    void onOpt2Changed(int idx);
    void onRunClicked();
    void onStopClicked();
    // Worker
    void onDatasetStarted  (int dsIdx, int totalDs, QString dsName);
    void onPhaseStarted    (int phase, int total, QString name);
    void onIterationDone   (int phase, int iter, int total, double loss, double vl);
    void onPhaseFinished   (int phase, double finalLoss);
    void onDatasetBoundsReady(QString dsName, QVector<double> lb,
                               QVector<double> rb, double initBound);
    void onDatasetFinished (QString dsName, QStringList configs,
                            QVector<double> trainLoss, QVector<double> trainErr,
                            QVector<double> valErr, QVector<double> testErr, bool isCls);
    void onAllDone         (std::shared_ptr<NeuralNetwork> net,
                            std::shared_ptr<Dataset> ds);
    void onError           (QString msg);
    // Exports
    void onExportLossPng();
    void onExportBoundsPng();
    void onExportEvalXlsx();
    void onExportCompXlsx();

private:
    // ── Dataset queue ──────────────────────────────────────────────────────
    QStringList   m_queue;
    QTableWidget* m_queueTable  = nullptr;
    QPushButton*  m_addBtn      = nullptr;
    QPushButton*  m_clearBtn    = nullptr;
    QPushButton*  m_removeBtn   = nullptr;

    // ── Config widgets ─────────────────────────────────────────────────────
    QLineEdit*      m_hiddenEdit = nullptr;
    QComboBox*      m_actCombo   = nullptr;
    QDoubleSpinBox* m_valSpin    = nullptr;
    QComboBox*      m_opt1Combo  = nullptr;
    QSpinBox*       m_nc1Spin    = nullptr;
    QSpinBox*       m_ng1Spin    = nullptr;
    QDoubleSpinBox* m_lamSpin    = nullptr;
    QDoubleSpinBox* m_aSpin      = nullptr;
    QDoubleSpinBox* m_fSpin      = nullptr;
    QDoubleSpinBox* m_ibSpin     = nullptr;
    QComboBox*      m_opt2Combo  = nullptr;
    QSpinBox*       m_nc2Spin    = nullptr;
    QSpinBox*       m_ng2Spin    = nullptr;
    QStackedWidget* m_geStack    = nullptr;
    QSpinBox*       m_nsGeSpin   = nullptr;
    QSpinBox*       m_nlGeSpin   = nullptr;
    QDoubleSpinBox* m_plGeSpin   = nullptr;
    QComboBox*      m_opt3Combo  = nullptr;
    QSpinBox*       m_nc3Spin    = nullptr;
    QSpinBox*       m_ng3Spin    = nullptr;
    QCheckBox*      m_bfgsChk    = nullptr;
    QCheckBox*      m_equalChk   = nullptr;   // baseline with the same total budget

    // ── Progress ──────────────────────────────────────────────────────────
    QPushButton*               m_runBtn            = nullptr;
    QPushButton*               m_stopBtn           = nullptr;
    // Config groups locked while a run is in progress (see setRunning)
    QGroupBox*                 m_netGrp            = nullptr;
    QGroupBox*                 m_ph1Grp            = nullptr;
    QGroupBox*                 m_ph2Grp            = nullptr;
    QGroupBox*                 m_ph3Grp            = nullptr;
    QSpinBox*                  m_nRunsSpin         = nullptr;
    QLabel*                    m_phaseNameLbl      = nullptr;
    QLabel*                    m_statusLbl         = nullptr;
    QWidget*                   m_progressContainer = nullptr;
    std::vector<QProgressBar*> m_bars;
    std::vector<QLabel*>       m_barLbls;
    std::vector<int>           m_phaseIters;
    std::vector<int>           m_xOff;

    // ── Tab widget ──────────────────────────────────────────────────────────
    QTabWidget*           m_tabs         = nullptr;

    // Tab 0: Loss
    QVBoxLayout*          m_lossLayout   = nullptr;
    QList<QChart*>        m_lossCharts;
    QList<QChartView*>    m_lossViews;
    QChart*               m_curLossChart  = nullptr;
    QValueAxis*           m_curLossAxisX  = nullptr;
    QValueAxis*           m_curLossAxisY  = nullptr;
    QList<QLineSeries*>   m_curLossSeries;

    // Tab 1: Bounds (only from last run — bounds are deterministic for fixed Phase1 seed)
    QVBoxLayout*          m_boundsLayout  = nullptr;
    QList<QChart*>        m_boundsCharts;
    QList<QChartView*>    m_boundsViews;

    // Tab 2: Evaluation
    QTableWidget*  m_evalTable  = nullptr;

    // Tab 3: Comparison
    QTableWidget*  m_compTable  = nullptr;

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

    EvolBoundsConfig               m_lastConfig;
    std::shared_ptr<NeuralNetwork> m_lastNet;
    std::shared_ptr<Dataset>       m_lastDs;

    QPointer<QThread>          m_thread;
    QPointer<EvolBoundsWorker> m_worker;

    // ── Helpers ───────────────────────────────────────────────────────────
    void buildUi();
    void buildUi_queueAndNetwork(QVBoxLayout* root);   // queue + network side by side
    QGroupBox* buildQueueGroup ();               // returns the box; caller places it
    void buildNetworkGroup();                    // builds m_netGrp
    void buildConfigArea (QVBoxLayout* root);
    void buildTabWidget  (QVBoxLayout* root);

    void addToQueue(const QStringList& paths);
    void refreshQueueTable();

    void styleChart(QChart* c, bool dark);
    void onTimerTick();
    void setRunning(bool r);
    void rebuildProgress(int nPhases);
    void startNewLossChart(const QString& dsName, int nPhases);
    void clearLossCharts();

    QChart* makeBoundsChart(const QString& dsName,
                             const QVector<double>& lb,
                             const QVector<double>& rb,
                             double initBound);
    bool exportStackedPng(const QList<QChart*>& charts,
                          QWidget* parent, const QString& defaultName);

    EvolBoundsConfig buildConfig()  const;
    void             launchWorker(const EvolBoundsConfig& cfg);
    void             finalizeAllRuns();
    void             refreshInfoLabel(const QString& phaseInfo);
};

} // namespace NeuralStudio
