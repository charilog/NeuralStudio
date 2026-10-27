#pragma once
// GEStudioPanel.h — Mode 6: Grammatical Evolution Studio
#include "core/ge/GEStudioWorker.h"
#include "core/ge/GEBatch.h"
#include "core/ge/GEReport.h"
#include "core/ge/GEExperiment.h"
#include <QPointer>
#include <QWidget>
#include <QStringList>
#include <QVector>
#include <memory>

class QTableWidget; class QPushButton; class QComboBox; class QSpinBox;
class QDoubleSpinBox; class QCheckBox; class QLineEdit; class QLabel;
class QPlainTextEdit; class QProgressBar; class QTabWidget; class QGroupBox;
class QChartView; class QChart; class QDragEnterEvent; class QDropEvent;
class QVBoxLayout; class QThread; class QTextBrowser; class QToolButton;

namespace NeuralStudio {

class GEModelPicker;

class GEStudioPanel : public QWidget {
    Q_OBJECT
public:
    explicit GEStudioPanel(QWidget* parent = nullptr);
    ~GEStudioPanel() override;

    QStringList queuePaths() const { return m_queue; }
    void setQueuePaths(const QStringList& paths);

protected:
    void dragEnterEvent(QDragEnterEvent* e) override;
    void dropEvent     (QDropEvent* e)       override;

signals:
    void runningStateChanged(bool running);
    // Best network of the last dataset (only when it uses no constructed features).
    void trainingCompleted(std::shared_ptr<NeuralNetwork> net, std::shared_ptr<Dataset> ds);

private slots:
    void onAddFiles();
    void onRemoveSelected();
    void onClearQueue();
    void onPresetChanged(int idx);
    void onValidateGrammar();
    void onRun();
    void onStop();
    void onGenotypeChanged();
    void onDatasetStarted(int dsIdx, int totalDs, QString dsName, QString info);
    void onGenerationDone(int gen, int totalGens, double bestFitness, QString bestDesc,
                          double meanFitness, double diversity,
                          int uniqueEvals, int cacheHits, qint64 innerCalls);
    void onFinalStageStarted(int totalRows);
    void onFinalRowDone(int row, int totalRows, QString label);
    void onDatasetFinished(GEDatasetResult r);
    void onAllDone(std::shared_ptr<NeuralNetwork> net, std::shared_ptr<Dataset> ds);
    void onError(QString msg);
    void onExportXlsx();

    // Batch experiment (same command line as tools/ge_batch, run in-process, no console)
    void onBatchRun();
    void onBatchCheck();
    void onBatchBrowseDir();
    void onBatchOpenResults();
    void onBatchUsage();
    void onBatchStarted(int jobsToDo, int alreadyDone);
    void onBatchProgress(int jobIdx, int jobsToDo, double withinJob, QString label);
    void onBatchJobFinished(int jobsDone, int jobsToDo, QString line, bool ok);
    void onBatchMessage(QString text);
    void onBatchFinished(bool stopped, QString prefix);
    void onBatchError(QString msg);

    // one Run / Export for both pages
    void onRunClicked();
    void onExportClicked();
    void onSetupTabChanged(int idx);

    // Experiment page
    void onNewEmpty();
    void onNewBenchmark();
    void onOpen();
    void onEdit();
    void onSave();
    void onSaveAs();
    void onExportTables();
    void onExperimentModelsPicked();

private:
    void buildUi();
    QGroupBox* buildQueueGroup();
    QGroupBox* buildGrammarGroup();
    QGroupBox* buildEvolutionGroup();
    QGroupBox* buildTrainingGroup();
    QGroupBox* buildFinalGroup();
    QGroupBox* buildModelsGroup();
    QWidget*   buildResultTabs();
    QWidget*   buildExperimentPage();
    void       setExperiment(const GEExperiment& e, const QString& path, bool dirty);
    void       refreshExperimentSummary();
    QString    experimentPrefix() const;        // absolute results prefix of the current experiment
    bool       applyCommandIfEdited(QString& error);
    GEReportFilter reportFilter() const;
    void       updateActionStates();
    bool       experimentMode() const;
    QString    defaultWorkDir() const;
    void       saveBatchSettings() const;
    bool       loadOverallFromRaw(const QString& rawPath, bool quiet);
    void       fillOverallTable(QTableWidget* t, const QString& task);
    void       fillHypothesesTable();
    void       refreshOverallViews();
    void       showResultTabsFor(bool experiment);
    QString    workDir() const;
    void addToQueue(const QStringList& paths);
    void refreshQueueTable();
    void setRunning(bool running);
    void log(const QString& line);
    void rebuildChart();
    GEStudioConfig collectConfig() const;
    void updateProgress(double withinDataset);

    // queue
    QStringList   m_queue;
    QTableWidget* m_queueTable = nullptr;
    QPushButton  *m_addBtn = nullptr, *m_removeBtn = nullptr, *m_clearBtn = nullptr;

    // grammar
    QComboBox*      m_presetCombo = nullptr;
    QPlainTextEdit* m_grammarEdit = nullptr;
    QLabel*         m_grammarStatus = nullptr;
    QSpinBox       *m_maxLayersSpin = nullptr, *m_maxFeaturesSpin = nullptr, *m_featSizeSpin = nullptr;

    // evolution
    QGroupBox*      m_evoGroup = nullptr;
    QComboBox      *m_genotypeCombo = nullptr, *m_crossCombo = nullptr;
    QSpinBox       *m_popSpin = nullptr, *m_gensSpin = nullptr, *m_tournSpin = nullptr,
                   *m_ncrSpin = nullptr, *m_genomeSpin = nullptr, *m_seedSpin = nullptr,
                   *m_patienceSpin = nullptr;
    QDoubleSpinBox *m_replaceSpin = nullptr, *m_mutSpin = nullptr, *m_pcrSpin = nullptr,
                   *m_alphaSpin = nullptr, *m_eliteSpin = nullptr, *m_floorSpin = nullptr;
    QCheckBox*      m_transferChk = nullptr;
    QWidget        *m_codonBox = nullptr, *m_pgeBox = nullptr;

    // training
    QGroupBox*      m_trainGroup = nullptr;
    QDoubleSpinBox *m_boundSpin = nullptr, *m_valSpin = nullptr, *m_parsSpin = nullptr;
    QComboBox      *m_innerCombo = nullptr, *m_metricCombo = nullptr;
    QSpinBox       *m_innerPopSpin = nullptr, *m_innerItersSpin = nullptr;

    // final
    QGroupBox*      m_finalGroup = nullptr;
    QSpinBox       *m_topKSpin = nullptr, *m_finalPopSpin = nullptr, *m_finalItersSpin = nullptr;
    QCheckBox      *m_polishChk = nullptr;
    QLineEdit*      m_baseHiddenEdit = nullptr;
    QComboBox*      m_baseActCombo = nullptr;

    // run
    QPushButton   *m_runBtn = nullptr, *m_stopBtn = nullptr, *m_exportBtn = nullptr;
    QProgressBar*  m_bar = nullptr;
    QLabel*        m_statusLbl = nullptr;

    // results
    QTabWidget*     m_tabs = nullptr;
    QChartView*     m_chartView = nullptr;
    QTableWidget   *m_cmpTable = nullptr, *m_topTable = nullptr;
    QPlainTextEdit *m_probsView = nullptr, *m_logView = nullptr;
    QVector<GEDatasetResult> m_results;

    // models (click-to-select lists)
    GEModelPicker  *m_singleModels = nullptr, *m_expModels = nullptr;

    // setup pages: Single run | Experiment
    QTabWidget*     m_setupTabs = nullptr;
    QWidget*        m_singlePage = nullptr;
    QWidget*        m_expPage = nullptr;

    // experiment page
    QTextBrowser*   m_expSummary = nullptr;
    QToolButton    *m_newBtn = nullptr, *m_cmdToggle = nullptr;
    QPushButton    *m_openBtn = nullptr, *m_editBtn = nullptr, *m_saveBtn = nullptr, *m_saveAsBtn = nullptr;
    QWidget*        m_cmdBox = nullptr;
    GEExperiment    m_exp;
    QString         m_expPath;          // file the experiment was opened from / saved to ("" = not saved)
    bool            m_expDirty = false;
    QString         m_cmdGenerated;     // command text produced from m_exp (to detect manual edits)

    // batch
    QPlainTextEdit *m_batchCmdEdit = nullptr, *m_batchView = nullptr;
    QLineEdit      *m_batchDirEdit = nullptr;
    QPushButton    *m_batchCheckBtn = nullptr, *m_batchBrowseBtn = nullptr,
                   *m_batchOpenBtn = nullptr, *m_batchUsageBtn = nullptr;
    QPointer<QThread> m_batchThread;
    QPointer<GEBatchRunner>  m_batchRunner;
    QPointer<GEStudioWorker> m_worker;
    bool            m_batchMode = false;
    int             m_batchJobs = 0;
    QString         m_batchPrefix;
    QString         m_batchLastLine;
    QTableWidget   *m_clsTable = nullptr, *m_regTable = nullptr, *m_hypTable = nullptr;
    GEOverall       m_overall;
    QString         m_overallSource;

    bool    m_running = false;
    int     m_dsIdx = 0, m_totalDs = 1;
    int     m_finalRows = 1, m_finalDone = 0;
    double  m_lastGenFrac = 0;
    bool    m_inFinal = false;
    QPointer<QThread> m_thread;
};

} // namespace NeuralStudio
