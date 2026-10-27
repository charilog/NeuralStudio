#pragma once
// GEExperiment.h — experiment files of Mode 6 (JSON, Qt Core only).
//
// One file describes a whole batch experiment: where the data are, which
// classification and which regression datasets take part, which grammar
// configurations and which models (methods) are run, whether each model uses
// the BNF grammar, and all options of the batch engine. The GE Studio panel
// loads it into the batch command; ge_batch runs it with --experiment <file>.
//
// {
//   "name": "bench64",
//   "description": "...",
//   "data_dir": "sample_data/tenfolding",
//   "output": "results/bench64",
//   "folds": "1-10",
//   "seeds": 1,
//   "datasets": { "classification": ["pima", ...], "regression": ["airfoil", ...] },
//   "configs": ["unified-ge"],
//   "models": [ { "id": "standard_T", "bnf": false, "run": true }, ... ],
//   "options": { "total-budget": 600000, "gens": 40, ... }
// }
#include <QByteArray>
#include <QMap>
#include <QString>
#include <QStringList>
#include <QVector>

namespace NeuralStudio {

struct GEExperimentModel {
    QString id;
    bool    bnf = false;   // as declared in the file (checked against the method catalog)
    bool    run = true;
};

struct GEExperiment {
    QString     name;
    QString     description;
    QString     dataDir;
    QString     output;
    QString     folds = "1-10";
    int         seeds = 1;
    QStringList classification;
    QStringList regression;
    QStringList configs{"unified-ge"};
    QStringList development;              // datasets used while developing the method: in the tables, not in the hypothesis tests
    QVector<GEExperimentModel> models;
    QMap<QString, QString>     options;   // batch option name without "--" -> value

    QStringList datasets() const { return classification + regression; }
    QStringList runModels() const;        // ids with run == true
    bool        fixedBudget() const { return options.contains("total-budget") && options.value("total-budget").toLongLong() > 0; }
};

// Models produced by the batch engine in the given budget mode, in report order.
QVector<GEExperimentModel> geDefaultModels(bool fixedBudget);
// Ids of the rows a grammar configuration produces in the given budget mode.
QStringList geGeMethodIds(bool fixedBudget);
// The rows of the first version (default of a configuration) and the self-adaptive rows (fixed budget only).
QStringList geClassicGeMethodIds(bool fixedBudget);
QStringList geAdaptiveMethodIds();

// Rebuild an experiment from a results file (<prefix>_raw.csv). Uses <prefix>_experiment.json when it
// exists (exact settings); otherwise datasets, tasks, configurations, models, folds and seeds are read
// from the rows and the options are left at their defaults (reported in `warnings`).
bool geExperimentFromRaw(const QString& rawPath, GEExperiment& e, QString& error, QStringList& warnings);

// "1-10", "1,3,5" from a list of folds
QString geFoldsToText(QList<int> folds);

// The 64-dataset benchmark of the BNF paper (as available in sample_data/tenfolding).
GEExperiment geBenchmarkExperiment();

// Load / save. `warnings` gets non-fatal problems (unknown keys, BNF flags that do
// not match the model, models that cannot be switched off, ...).
bool geLoadExperiment(const QString& path, GEExperiment& e, QString& error, QStringList& warnings);
bool geSaveExperiment(const QString& path, const GEExperiment& e, QString& error);
QByteArray geExperimentToJson(const GEExperiment& e);
bool geExperimentFromJson(const QByteArray& json, const QString& nameFallback, GEExperiment& e, QString& error,
                          QStringList& warnings);

// Runs of the experiment already complete in `rows` (every requested model present) and the total.
struct GERawRow;
int geExperimentDone(const GEExperiment& e, const QVector<GERawRow>& rows, int& total);

// The equivalent ge_batch command line (without the program name).
QString geExperimentToCommand(const GEExperiment& e);

// Task of a dataset as NeuralStudio will treat it, from <dir>/<ds>_1.train
// ("classification", "regression" or "" if the file cannot be read).
QString geDetectDatasetTask(const QString& dir, const QString& ds, QString* detail = nullptr);

// Human-readable summary (models with BNF yes/no, dataset counts, options).
QString geExperimentSummary(const GEExperiment& e);

} // namespace NeuralStudio
