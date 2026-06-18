#pragma once
#include "core/batch/BatchJob.h"
#include "core/batch/BatchRunner.h"
#include <QWidget>
#include <QLabel>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QPushButton>
#include <QProgressBar>
#include <QTableWidget>
#include <QTabWidget>
#include <QLineEdit>
#include <QComboBox>
#include <QThread>
#include <QPointer>
#include <QTimer>
#include <QElapsedTimer>
#include <vector>

namespace NeuralStudio {

// Forward declarations for statistical tabs (defined in WilcoxonTab.h / FriedmanTab.h)
class WilcoxonTab;
class FriedmanTab;

// ─── BatchPanel ───────────────────────────────────────────────────────────────
//  Runs a grid of jobs: every queued dataset × every checked optimizer.
//  Results table has one row per (dataset, optimizer) combination, allowing
//  direct comparison of algorithms on the same data.
// ─────────────────────────────────────────────────────────────────────────────
class BatchPanel : public QWidget {
    Q_OBJECT
public:
    explicit BatchPanel(QWidget* parent = nullptr);

protected:
    void dragEnterEvent(QDragEnterEvent* e) override;
    void dropEvent(QDropEvent* e)           override;

private slots:
    void onAddFiles();
    void onClearQueue();
    void onRunBatch();
    void onStopBatch();
    void onExportXlsx();
    void onTimerTick();
    void onSelectAll();
    void onSelectNone();
    void onSelectGradient();
    void onSelectEvolutionary();

    void onJobStarted  (int index, QString name);
    void onJobProgress (int index, int epoch, int totalEpochs, double trainLoss);
    void onJobCompleted(int index, BatchJobResult result);
    void onAllJobsFinished();
    void onBatchError  (QString msg);

private:
    // ── Dataset queue ────────────────────────────────────────────────────────
    QTableWidget* m_queueTable = nullptr;
    QStringList   m_queue;            // dataset paths

    // ── Optimizer checkboxes ─────────────────────────────────────────────────
    struct OptimizerEntry {
        QString    id;
        QString    label;
        bool       gradient;    // Adam/SGD/AdamW/RMSProp/Nadam/AdaGrad
        bool       quasiNewton; // L-BFGS/BFGS/CG/LM
        bool       singlePoint; // SA/NM (derivative-free single-point)
        QCheckBox* check = nullptr;
    };
    std::vector<OptimizerEntry> m_optimEntries;

    // ── Architecture + training config ───────────────────────────────────────
    QLineEdit*      m_hiddenEdit  = nullptr;
    QComboBox*      m_activCombo  = nullptr;
    QDoubleSpinBox* m_dropoutSpin = nullptr;
    QDoubleSpinBox* m_lrSpin      = nullptr;
    QSpinBox*       m_epochsSpin  = nullptr;
    QSpinBox*       m_batchSpin   = nullptr;   // Batch for gradient, Population for evolutionary
    QDoubleSpinBox* m_valSpin     = nullptr;
    QSpinBox*       m_earlySpin   = nullptr;
    QDoubleSpinBox* m_wBoundSpin  = nullptr;

    // ── Buttons + status ─────────────────────────────────────────────────────
    QPushButton*  m_addBtn    = nullptr;
    QPushButton*  m_clearBtn  = nullptr;
    QPushButton*  m_runBtn    = nullptr;
    QPushButton*  m_stopBtn   = nullptr;
    QLabel*       m_statusLbl = nullptr;
    QProgressBar* m_overall   = nullptr;
    QLabel*       m_timeLbl   = nullptr;

    // ── Results area (tabbed) ────────────────────────────────────────────────
    QTabWidget*    m_tabWidget    = nullptr;
    QTableWidget*  m_resultsTable = nullptr;
    QPushButton*   m_exportBtn    = nullptr;   // lives inside Results tab
    WilcoxonTab*   m_wilcoxon     = nullptr;
    FriedmanTab*   m_friedman     = nullptr;

    // ── State ────────────────────────────────────────────────────────────────
    std::vector<BatchJobResult> m_results;
    QPointer<QThread>     m_thread;
    QPointer<BatchRunner> m_runner;
    bool                  m_running = false;
    QElapsedTimer         m_batchClock;
    QTimer*               m_tickTimer = nullptr;
    int                   m_jobsDone  = 0;

    // ── Helpers ──────────────────────────────────────────────────────────────
    void         buildUi();
    void         refreshQueueTable();
    void         appendResultRow(const BatchJobResult& r);
    void         setRunning(bool r);
    void         setResultsSortingEnabled(bool on);
    QStringList  selectedOptimizerIds() const;   // checked optimizers
    BatchJobConfig makeJobTemplate(const QString& dsPath,
                                   const QString& optimId) const;
};

} // namespace NeuralStudio
