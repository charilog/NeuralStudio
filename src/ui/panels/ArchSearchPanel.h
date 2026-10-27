#pragma once
// ArchSearchPanel.h — Mode 5: Evolutionary Architecture Search  v1.0.0
#include "core/arch/ArchSearchWorker.h"
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
class QDragEnterEvent; class QDropEvent;
QT_END_NAMESPACE

namespace NeuralStudio {

class ArchSearchPanel : public QWidget {
    Q_OBJECT
public:
    explicit ArchSearchPanel(QWidget* parent = nullptr);

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
    // Emitted when user clicks "Apply to Mode 1": carries path + best architecture.
    void applyToMode1(QString datasetPath, QVector<int> hiddenSizes, QVector<QString> actNames);

private slots:
    void onAddFiles();
    void onClearQueue();
    void onRemoveSelected();
    void onRunClicked();
    void onStopClicked();
    // Worker
    void onDatasetStarted (int dsIdx, int totalDs, QString dsName);
    void onEvalDone       (int evalNum, int totalEvals, double fitness,
                           QString archDesc, bool cacheHit);
    void onGenerationDone (int gen, int totalGens,
                           double bestFitness, QString bestDesc,
                           int uniqueEvals, int cacheHits);
    void onDatasetFinished(QString dsName,
                           QVector<ArchResult> topResults,
                           QVector<double>     convergence,
                           bool                isClassification);
    void onAllDone();
    void onError(QString msg);
    // Exports
    void onExportResultsXlsx();

private:
    // ── Dataset queue ──────────────────────────────────────────────────────
    QStringList   m_queue;
    QTableWidget* m_queueTable  = nullptr;
    QPushButton*  m_addBtn      = nullptr;
    QPushButton*  m_clearBtn    = nullptr;
    QPushButton*  m_removeBtn   = nullptr;

    // ── Search space ───────────────────────────────────────────────────────
    QGroupBox*      m_searchSpaceGroup = nullptr;   // disabled while running
    QSpinBox*       m_minLayersSpin  = nullptr;
    QSpinBox*       m_maxLayersSpin  = nullptr;
    QLineEdit*      m_sizesEdit      = nullptr;   // comma-separated candidate sizes
    QCheckBox*      m_actRelu        = nullptr;
    QCheckBox*      m_actTanh        = nullptr;
    QCheckBox*      m_actSigmoid     = nullptr;
    QCheckBox*      m_actLinear      = nullptr;
    QDoubleSpinBox* m_valSpin        = nullptr;
    QSpinBox*       m_topKSpin       = nullptr;

    // ── Outer optimizer (architecture search) ─────────────────────────────
    QGroupBox* m_outerOptGroup = nullptr;   // disabled while running
    QComboBox* m_outerOptCombo = nullptr;
    QSpinBox*  m_outerPopSpin  = nullptr;
    QSpinBox*  m_outerGensSpin = nullptr;

    // ── Inner optimizer (training each candidate) ─────────────────────────
    QGroupBox*      m_innerOptGroup  = nullptr;   // disabled while running
    QComboBox*      m_innerOptCombo  = nullptr;
    QSpinBox*       m_innerPopSpin   = nullptr;
    QSpinBox*       m_innerItersSpin = nullptr;
    QDoubleSpinBox* m_innerWBSpin    = nullptr;
    QSpinBox*       m_innerPolishSpin = nullptr;   // L-BFGS polish evaluations (0 = off)

    // ── Progress controls ──────────────────────────────────────────────────
    QPushButton*  m_runBtn     = nullptr;
    QPushButton*  m_stopBtn    = nullptr;
    QPushButton*  m_applyBtn   = nullptr;   // visible only after a successful search
    QSpinBox*     m_nRunsSpin  = nullptr;

    // ── Per-dataset best result for "Apply to Mode 1" ──────────────────────
    struct DatasetApplyInfo {
        QString    datasetPath;
        QString    dsName;
        bool       isCls = false;
        ArchResult bestResult;
    };
    QList<DatasetApplyInfo> m_applyInfos;   // one per dataset, filled in finalizeAllRuns
    QLabel*       m_infoLbl    = nullptr;   // combined run/dataset/gen/eval info

    // ── Progress bars — fastest (top) → slowest (bottom) ────────────────────
    // Eval is always visible. Gen/Dataset/Run only appear when their loop
    // actually iterates more than once (otherwise they would just sit at
    // 0% or 100% the whole time and add visual noise). Overall is always
    // visible and is the bottom-most, slowest-moving bar.
    QProgressBar* m_evalBar    = nullptr;  // 0→100% per generation     (always shown)
    QWidget*      m_genRow     = nullptr;  // wraps m_genBar + label, hidden if G==1
    QProgressBar* m_genBar     = nullptr;  // 0→100% per dataset
    QWidget*      m_dsRow      = nullptr;  // wraps m_dsBar + label, hidden if D==1
    QProgressBar* m_dsBar      = nullptr;  // 0→100% per run
    QWidget*      m_runRow     = nullptr;  // wraps m_runBar + label, hidden if N==1
    QProgressBar* m_runBar     = nullptr;  // 0→100% across all runs
    QProgressBar* m_overallBar = nullptr;  // 0→100% total              (always shown)

    QLabel*       m_timeLbl    = nullptr;
    QElapsedTimer m_clock;
    QTimer*       m_tickTimer  = nullptr;

    // ── Tab widget ─────────────────────────────────────────────────────────
    QTabWidget* m_tabs = nullptr;

    // Tab 0: Top-K results table
    QTableWidget* m_resultsTable = nullptr;

    // Tab 1: Comparison (N>1 runs: mean ± std; N=1: same as Tab 0)
    QTableWidget* m_compTable = nullptr;

    // ── N-runs state (same pattern as Modes 3 & 4) ────────────────────────
    struct DsAccum {
        QVector<QVector<ArchResult>> runResults;
        bool isCls = false;
    };
    QMap<QString, DsAccum> m_accumResults;
    ArchSearchConfig        m_lastConfig;
    int  m_currentRun    = 0;
    int  m_totalRuns     = 1;
    bool m_stopRequested = false;

    // ── Runtime state ──────────────────────────────────────────────────────
    bool    m_running        = false;
    int     m_dsDone         = 0;
    int     m_currentDsIdx   = 0;
    int     m_totalDs        = 0;
    QString m_currentDsName;
    int     m_currentGen     = 0;
    int     m_totalGens      = 0;
    int     m_currentEval    = 0;
    int     m_totalEvals     = 0;
    int     m_uniqueEvals    = 0;
    int     m_cacheHits      = 0;
    double  m_bestFitness    = 1e9;
    QString m_bestDesc;
    bool    m_isCls          = false;

    QPointer<QThread>          m_thread;
    QPointer<ArchSearchWorker> m_worker;

    // ── Helpers ───────────────────────────────────────────────────────────
    void buildUi();
    void buildUi_queueAndSearchSpace(QVBoxLayout* root);   // queue + search space side by side
    QGroupBox* buildQueueGroup   ();               // returns the box; caller places it
    QGroupBox* buildSearchSpace  ();               // returns the box; caller places it
    void buildOptimizerArea(QVBoxLayout* root);
    void buildProgressArea (QVBoxLayout* root);
    void buildTabWidget    (QVBoxLayout* root);

    void addToQueue(const QStringList& paths);
    void refreshQueueTable();

    void onTimerTick();
    void refreshInfoLabel();
    void setRunning(bool r);
    void updateBarVisibility();   // shows/hides Gen/Dataset/Run rows based on G/D/N

    ArchSearchConfig buildConfig() const;
    void             launchWorker(const ArchSearchConfig& cfg);
    void             finalizeAllRuns();

    void populateResultsTable(const QString& dsName,
                              const QVector<ArchResult>& results,
                              bool isCls, bool append);
    void rebuildComparisonTable();
};

} // namespace NeuralStudio
