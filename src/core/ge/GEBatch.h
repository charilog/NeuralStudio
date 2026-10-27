#pragma once
// GEBatch.h — the batch experiment engine of Mode 6 (folds x seeds x configurations),
// shared by the command-line tool tools/ge_batch.cpp and the GE Studio panel.
// Qt Core only. The same command line works in both places, e.g.
//
//   ge_batch --dir sample_data/tenfolding --datasets wine,airfoil --folds 1-10
//            --configs unified-ge,archbounds-ge --seeds 3 --out results/run1
//
// Paths may use '\' or '/' on every platform; relative paths are resolved against
// a base folder (the current folder for the CLI, the panel's working folder in the GUI).
#include <QList>
#include <QMap>
#include <QMutex>
#include <QObject>
#include <QString>
#include <QStringList>
#include <atomic>

namespace NeuralStudio {

class GEStudioWorker;

struct GEBatchOptions {
    QMap<QString, QString> args;   // every --key value pair (flags map to "1")
    QString     dir;               // absolute dataset folder
    QString     prefix;            // absolute output prefix (<prefix>_raw.csv, ...)
    QStringList datasets;
    QStringList devDatasets;       // --dev-datasets: left out of the hypothesis tests
    QList<int>  folds;
    QStringList configs;
    int         seeds = 1;
    bool        summarizeOnly = false;
    bool        noResume = false;
    bool        help = false;
    QStringList warnings;          // non-fatal problems (e.g. from an experiment file)
    QString     experimentPath;    // set when --experiment was used
};

// Split a command line into arguments. Handles "double" and 'single' quotes and
// keeps backslashes (Windows paths) untouched.
QStringList splitBatchCommandLine(const QString& cmd);

// Parse arguments (a leading program name such as .\build\Release\ge_batch.exe is
// ignored). Returns false and sets `error` on invalid input.
bool parseBatchArgs(const QStringList& args, const QString& baseDir, GEBatchOptions& out, QString& error);

QString batchUsage();

struct GEExperiment;
// Models a configuration must produce for these options (all of the mode, or the --methods subset).
QStringList geRequestedMethods(const GEBatchOptions& o, const QString& cfg);
// Experiment equivalent to the options; datasets are sorted by task reading their first fold
// (the ones that cannot be read are returned in `unreadable`).
bool geExperimentFromBatchOptions(const GEBatchOptions& o, GEExperiment& e, QStringList& unreadable);

class GEBatchRunner : public QObject {
    Q_OBJECT
public:
    explicit GEBatchRunner(QObject* parent = nullptr);
    void setOptions(const GEBatchOptions& o) { m_opt = o; }

public slots:
    void run();
    // Thread-safe: may be called from any thread (use Qt::DirectConnection).
    void requestStop();

signals:
    void message(QString text);                                    // log / summary text (may be multi-line)
    void batchStarted(int jobsToDo, int alreadyDone);
    void progress(int jobIdx, int jobsToDo, double withinJob, QString label);   // jobIdx is 0-based
    void jobFinished(int jobsDone, int jobsToDo, QString line, bool ok);
    void finished(bool stopped, QString prefix);
    void errorOccurred(QString msg);

private:
    GEBatchOptions    m_opt;
    std::atomic<bool> m_stop{false};
    QMutex            m_mx;
    GEStudioWorker*   m_current = nullptr;   // guarded by m_mx
};

} // namespace NeuralStudio
