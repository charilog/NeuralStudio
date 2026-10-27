#pragma once
// GEExperimentDialog.h — edit a Mode 6 experiment: datasets (classification / regression),
// models (with / without BNF), grammar configurations, folds, seeds and options.
// Shows, for an experiment that already has results, how much of every dataset and model is done,
// so models and datasets can be added (only the missing runs are computed) or removed (their
// results stay in the file and simply leave the tables).
#include "core/ge/GEExperiment.h"
#include <QDialog>
#include <functional>
#include <QMap>
#include <QSet>
#include <QVector>

class QLineEdit; class QSpinBox; class QCheckBox; class QTableWidget; class QLabel; class QTabWidget;

namespace NeuralStudio {

class GEModelPicker;

class GEExperimentDialog : public QDialog {
    Q_OBJECT
public:
    GEExperimentDialog(const GEExperiment& e, const QString& workDir, QWidget* parent = nullptr);
    GEExperiment experiment() const;

protected:
    void accept() override;

private slots:
    void rescanDatasets();
    void applyFilter();
    void browseData();
    void browseOutput();
    void refreshModels();

private:
    struct DsInfo { QString name, task, detail; qint64 bytes = 0; };
    QString resolve(const QString& p) const;
    void    readResults();
    void    fillDatasets();
    void    setChecked(const std::function<bool(int row)>& pred);

    QString       m_workDir;
    GEExperiment  m_in;
    QTabWidget*   m_tabs = nullptr;
    QLineEdit    *m_dev = nullptr;
    QLineEdit    *m_name = nullptr, *m_desc = nullptr, *m_dataDir = nullptr, *m_output = nullptr, *m_folds = nullptr,
                 *m_filter = nullptr;
    QSpinBox*     m_seeds = nullptr;
    QVector<QCheckBox*> m_cfgChecks;
    GEModelPicker* m_models = nullptr;
    QTableWidget *m_datasets = nullptr, *m_options = nullptr;
    QLabel       *m_dsCount = nullptr, *m_resultsInfo = nullptr;
    QVector<DsInfo> m_all;
    QSet<QString>   m_selected;            // datasets ticked (kept across rescans / filters)
    QMap<QString, QSet<int>> m_doneFolds;  // dataset -> folds with any result
    QMap<QString, int>       m_rowsByModel;
    bool            m_hasResults = false;
};

} // namespace NeuralStudio
