#include "GEExperimentDialog.h"
#include "GEModelPicker.h"
#include "core/ge/Baselines.h"
#include "core/ge/GEReport.h"
#include <QCheckBox>
#include <QDateTime>
#include <QDialogButtonBox>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QHash>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QProgressDialog>
#include <QPushButton>
#include <QRegularExpression>
#include <QSpinBox>
#include <QTabWidget>
#include <QTableWidget>
#include <QVBoxLayout>

namespace NeuralStudio {

namespace {
// option -> (default shown as placeholder, meaning)
struct OptInfo { const char* key; const char* def; const char* help; };
const OptInfo kOpts[] = {
    {"total-budget", "0", "fixed total budget T per run (loss evaluations); 0 = equal-budget mode"},
    {"gens", "40", "GE generations"},
    {"pop", "40", "GE population"},
    {"patience", "15", "stop the search after this many generations without improvement"},
    {"inner-pop", "30", "population of the inner optimizer (training of each phenotype)"},
    {"inner-iters", "60", "iterations of the inner optimizer"},
    {"inner-polish", "1", "1 = inner training ends with bound-respecting L-BFGS"},
    {"inner-polish-evals", "300", "evaluations of that L-BFGS"},
    {"final-pop", "80", "population of the final training"},
    {"final-iters", "200", "iterations of the final training"},
    {"search-frac", "0.5", "fixed-T mode: share of T the search may use"},
    {"polish-frac", "0.5", "fixed-T mode: share of each row's budget given to L-BFGS"},
    {"lamarck", "3", "Lamarckian refinements per phenotype"},
    {"finalists", "5", "phenotypes refined and re-ranked after the search"},
    {"bound-mode", "fan", "fan (scale-aware bounds) | abs (±bound)"},
    {"fan-k", "3", "multiplier of the Glorot limit (fan mode)"},
    {"bound", "10", "±bound (abs mode)"},
    {"val-split", "0.2", "validation share of the train file used by the search"},
    {"metric", "loss", "fitness: loss | error"},
    {"optimizer", "de", "inner optimizer"},
    {"parsimony", "0", "fitness *= 1 + parsimony * params / 1000"},
    {"topk", "1", "GE ranks retrained with the final budget"},
    {"baseline-hidden", "16", "hidden layers of the Standard network"},
    {"baseline-act", "relu", "activation of the Standard network"},
};
const QStringList kConfigs = {"unified-ge", "archbounds-ge", "archonly-ge", "unified-pge", "unified-gelocal", "unified-sage", "unifiedpe-sage"};

QHash<QString, QPair<QString, QString>>& taskCache() {   // path|mtime -> (task, detail)
    static QHash<QString, QPair<QString, QString>> c;
    return c;
}

QTableWidgetItem* ro(const QString& t) {
    auto* it = new QTableWidgetItem(t);
    it->setFlags(it->flags() & ~Qt::ItemIsEditable);
    return it;
}
} // namespace

GEExperimentDialog::GEExperimentDialog(const GEExperiment& e, const QString& workDir, QWidget* parent)
    : QDialog(parent), m_workDir(workDir), m_in(e) {
    setWindowTitle("Experiment — " + (e.name.isEmpty() ? QString("new") : e.name));
    resize(980, 720);
    auto* root = new QVBoxLayout(this);

    // general
    auto* gen = new QGroupBox("General");
    auto* gf = new QFormLayout(gen);
    m_name = new QLineEdit(e.name);
    m_desc = new QLineEdit(e.description);
    m_dataDir = new QLineEdit(e.dataDir);
    m_output = new QLineEdit(e.output);
    m_folds = new QLineEdit(e.folds);
    m_seeds = new QSpinBox; m_seeds->setRange(1, 1000); m_seeds->setValue(e.seeds);
    auto withBrowse = [&](QLineEdit* ed, const char* slot) {
        auto* w = new QWidget; auto* h = new QHBoxLayout(w); h->setContentsMargins(0, 0, 0, 0);
        auto* b = new QPushButton("…"); b->setFixedWidth(32);
        h->addWidget(ed, 1); h->addWidget(b);
        connect(b, SIGNAL(clicked()), this, slot);
        return w;
    };
    gf->addRow("Name", m_name);
    gf->addRow("Description", m_desc);
    gf->addRow("Data folder", withBrowse(m_dataDir, SLOT(browseData())));
    gf->addRow("Results (prefix)", withBrowse(m_output, SLOT(browseOutput())));
    auto* fs = new QWidget; auto* fh = new QHBoxLayout(fs); fh->setContentsMargins(0, 0, 0, 0);
    fh->addWidget(m_folds, 1); fh->addWidget(new QLabel("Seeds")); fh->addWidget(m_seeds);
    gf->addRow("Folds", fs);
    auto* cfgW = new QWidget; auto* ch = new QHBoxLayout(cfgW); ch->setContentsMargins(0, 0, 0, 0);
    QStringList cfgs = kConfigs;
    for (const QString& c : e.configs) if (!cfgs.contains(c) && c != "baselines") cfgs << c;
    for (const QString& c : cfgs) {
        auto* cb = new QCheckBox(c);
        cb->setChecked(e.configs.contains(c));
        cb->setToolTip("grammar configuration <preset>-<genotype>; used by the models that use BNF (except GNN)");
        m_cfgChecks << cb; ch->addWidget(cb);
    }
    ch->addStretch();
    gf->addRow("Grammar configurations", cfgW);
    m_dev = new QLineEdit(e.development.join(", "));
    m_dev->setPlaceholderText("e.g. airfoil, concrete, housing");
    m_dev->setToolTip("Datasets used while developing / tuning the method. They stay in the tables but are left out of "
                      "the hypothesis tests (sheet Hypotheses), so the tests use confirmation datasets only.");
    gf->addRow("Development datasets", m_dev);
    m_resultsInfo = new QLabel; m_resultsInfo->setWordWrap(true); m_resultsInfo->setStyleSheet("color:palette(mid);");
    gf->addRow("", m_resultsInfo);
    root->addWidget(gen);

    m_tabs = new QTabWidget;
    // datasets
    auto* dsPage = new QWidget; auto* dl = new QVBoxLayout(dsPage);
    auto* dtop = new QHBoxLayout;
    m_filter = new QLineEdit; m_filter->setPlaceholderText("filter by name…");
    auto* bCls = new QPushButton("All classification");
    auto* bReg = new QPushButton("All regression");
    auto* bNone = new QPushButton("None");
    auto* bRescan = new QPushButton("Rescan folder");
    m_dsCount = new QLabel;
    dtop->addWidget(m_filter, 1); dtop->addWidget(bCls); dtop->addWidget(bReg); dtop->addWidget(bNone); dtop->addWidget(bRescan);
    dl->addLayout(dtop);
    m_datasets = new QTableWidget(0, 5);
    m_datasets->setHorizontalHeaderLabels({"Use", "Dataset", "Task", "Size (KB)", "Results"});
    m_datasets->horizontalHeader()->setStretchLastSection(true);
    m_datasets->verticalHeader()->setVisible(false);
    m_datasets->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_datasets->setSortingEnabled(true);
    dl->addWidget(m_datasets, 1);
    dl->addWidget(m_dsCount);
    m_tabs->addTab(dsPage, "Datasets");

    // models
    auto* mPage = new QWidget; auto* ml = new QVBoxLayout(mPage);
    m_models = new GEModelPicker;
    ml->addWidget(m_models, 1);
    auto* mnote = new QLabel("Adding a model computes only that model on the runs that lack it (a model that uses BNF repeats "
                             "the GE run of that fold so its ablations refer to the same phenotype). Removing a model keeps its "
                             "results in the file; it only leaves the tables.");
    mnote->setWordWrap(true); mnote->setStyleSheet("color:palette(mid);");
    ml->addWidget(mnote);
    m_tabs->addTab(mPage, "Models");

    // options
    auto* oPage = new QWidget; auto* ol = new QVBoxLayout(oPage);
    m_options = new QTableWidget(0, 3);
    m_options->setHorizontalHeaderLabels({"Option", "Value (empty = default)", "Meaning"});
    m_options->horizontalHeader()->setStretchLastSection(true);
    m_options->verticalHeader()->setVisible(false);
    for (const auto& o : kOpts) {
        const int r = m_options->rowCount(); m_options->insertRow(r);
        m_options->setItem(r, 0, ro(o.key));
        auto* v = new QTableWidgetItem(e.options.value(o.key));
        v->setToolTip(QString("default: %1").arg(o.def));
        m_options->setItem(r, 1, v);
        m_options->setItem(r, 2, ro(QString("%1 (default %2)").arg(o.help, o.def)));
    }
    m_options->resizeColumnsToContents();
    ol->addWidget(m_options, 1);
    auto* onote = new QLabel("Changing options of an experiment that already has GE results makes old and new GE rows "
                             "incomparable; use a new results prefix for that.");
    onote->setWordWrap(true); onote->setStyleSheet("color:palette(mid);");
    ol->addWidget(onote);
    m_tabs->addTab(oPage, "Options");
    root->addWidget(m_tabs, 1);

    auto* bb = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    root->addWidget(bb);
    connect(bb, &QDialogButtonBox::accepted, this, &GEExperimentDialog::accept);
    connect(bb, &QDialogButtonBox::rejected, this, &QDialog::reject);
    connect(m_filter, &QLineEdit::textChanged, this, &GEExperimentDialog::applyFilter);
    connect(bRescan, &QPushButton::clicked, this, &GEExperimentDialog::rescanDatasets);
    connect(bCls, &QPushButton::clicked, this, [this] { setChecked([this](int r) { return m_datasets->item(r, 2)->text() == "classification"; }); });
    connect(bReg, &QPushButton::clicked, this, [this] { setChecked([this](int r) { return m_datasets->item(r, 2)->text() == "regression"; }); });
    connect(bNone, &QPushButton::clicked, this, [this] { setChecked([](int) { return false; }); });
    connect(m_datasets, &QTableWidget::itemChanged, this, [this](QTableWidgetItem* it) {
        if (it->column() != 0) return;
        const QString n = m_datasets->item(it->row(), 1) ? m_datasets->item(it->row(), 1)->text() : QString();
        if (n.isEmpty()) return;
        if (it->checkState() == Qt::Checked) m_selected.insert(n); else m_selected.remove(n);
        m_dsCount->setText(QString("%1 datasets selected").arg(m_selected.size()));
    });

    for (const QString& d : e.datasets()) m_selected.insert(d);
    readResults();
    refreshModels();
    rescanDatasets();
}

QString GEExperimentDialog::resolve(const QString& p0) const {
    QString p = p0.trimmed(); p.replace('\\', '/');
    if (p.isEmpty()) return p;
    return QDir::isAbsolutePath(p) ? QDir::cleanPath(p) : QDir::cleanPath(QDir(m_workDir).absoluteFilePath(p));
}

void GEExperimentDialog::readResults() {
    m_doneFolds.clear(); m_rowsByModel.clear(); m_hasResults = false;
    const QString raw = resolve(m_output->text()) + "_raw.csv";
    QVector<GERawRow> rows; QString err;
    if (QFileInfo::exists(raw) && geLoadRawCsv(raw, rows, err)) {
        m_hasResults = true;
        for (const GERawRow& r : rows) { m_doneFolds[r.ds].insert(r.fold); ++m_rowsByModel[r.method]; }
        m_resultsInfo->setText(QString("Results found: %1 (%2 rows). Ticked datasets/models that already have results "
                                       "are not computed again.").arg(QDir::toNativeSeparators(raw)).arg(rows.size()));
    } else {
        m_resultsInfo->setText("No results yet for this prefix.");
    }
}

void GEExperimentDialog::refreshModels() {
    // keep the current ticks when only the results prefix changed
    const QStringList ticked = m_models->ids().isEmpty() ? m_in.runModels() : m_models->checked();
    QStringList ids;
    for (const auto& m : geDefaultModels(m_in.fixedBudget())) ids << m.id;
    m_models->setEntries(GEModelPicker::catalogEntries(ids));
    m_models->setChecked(ticked);
    QMap<QString, QString> rows;
    for (const QString& id : ids) rows[id] = QString::number(m_rowsByModel.value(id));
    m_models->setInfoColumn(m_hasResults ? QString("Result rows") : QString(), rows);
}

void GEExperimentDialog::rescanDatasets() {
    const QString dir = resolve(m_dataDir->text());
    m_all.clear();
    QDir d(dir);
    if (!d.exists()) {
        m_dsCount->setText("Data folder not found: " + QDir::toNativeSeparators(dir));
        fillDatasets();
        return;
    }
    QStringList names;
    for (const QString& f : d.entryList({"*_1.train"}, QDir::Files)) names << f.left(f.size() - QString("_1.train").size());
    for (const QString& n : m_selected) if (!names.contains(n)) names << n;   // keep selections that are not in the folder
    QProgressDialog pd("Reading datasets…", "Skip", 0, names.size(), this);
    pd.setWindowModality(Qt::WindowModal);
    pd.setMinimumDuration(400);
    int k = 0;
    for (const QString& n : names) {
        pd.setValue(k++);
        DsInfo info; info.name = n;
        const QFileInfo fi(QString("%1/%2_1.train").arg(dir, n));
        info.bytes = fi.exists() ? fi.size() : 0;
        if (m_in.classification.contains(n)) info.task = "classification";
        else if (m_in.regression.contains(n)) info.task = "regression";
        else if (!fi.exists() || fi.size() == 0) info.task = "missing";
        else if (!pd.wasCanceled()) {
            const QString key = fi.absoluteFilePath() + "|" + QString::number(fi.lastModified().toSecsSinceEpoch());
            if (!taskCache().contains(key)) {
                QString detail; const QString t = geDetectDatasetTask(dir, n, &detail);
                taskCache()[key] = {t.isEmpty() ? QString("unreadable") : t, detail};
            }
            info.task = taskCache()[key].first; info.detail = taskCache()[key].second;
        } else info.task = "?";
        m_all << info;
    }
    pd.setValue(names.size());
    fillDatasets();
}

void GEExperimentDialog::fillDatasets() {
    m_datasets->blockSignals(true);
    m_datasets->setSortingEnabled(false);
    m_datasets->setRowCount(0);
    QStringList folds;
    const QList<int> fl = [&] { QList<int> o; for (const QString& p : m_folds->text().split(',', Qt::SkipEmptyParts)) {
        const QStringList ab = p.split('-'); if (ab.size() == 2) for (int i = ab[0].toInt(); i <= ab[1].toInt(); ++i) o << i; else o << p.toInt(); } return o; }();
    for (const DsInfo& i : m_all) {
        const int r = m_datasets->rowCount(); m_datasets->insertRow(r);
        auto* c = new QTableWidgetItem; c->setFlags(Qt::ItemIsUserCheckable | Qt::ItemIsEnabled | Qt::ItemIsSelectable);
        c->setCheckState(m_selected.contains(i.name) ? Qt::Checked : Qt::Unchecked);
        m_datasets->setItem(r, 0, c);
        m_datasets->setItem(r, 1, ro(i.name));
        auto* t = ro(i.task); t->setToolTip(i.detail);
        if (i.task != "classification" && i.task != "regression") t->setForeground(QColor("#c0392b"));
        m_datasets->setItem(r, 2, t);
        auto* s = new QTableWidgetItem; s->setData(Qt::DisplayRole, static_cast<int>((i.bytes + 1023) / 1024));
        s->setFlags(s->flags() & ~Qt::ItemIsEditable);
        m_datasets->setItem(r, 3, s);
        int done = 0; for (int f : fl) if (m_doneFolds.value(i.name).contains(f)) ++done;
        m_datasets->setItem(r, 4, ro(m_hasResults && m_doneFolds.contains(i.name) ? QString("%1 / %2 folds").arg(done).arg(fl.size()) : QString()));
    }
    m_datasets->setSortingEnabled(true);
    m_datasets->sortItems(1);
    m_datasets->blockSignals(false);
    m_datasets->resizeColumnsToContents();
    applyFilter();
    m_dsCount->setText(QString("%1 datasets selected, %2 in the folder").arg(m_selected.size()).arg(m_all.size()));
}

void GEExperimentDialog::applyFilter() {
    const QString f = m_filter->text().trimmed();
    for (int r = 0; r < m_datasets->rowCount(); ++r)
        m_datasets->setRowHidden(r, !f.isEmpty() && !m_datasets->item(r, 1)->text().contains(f, Qt::CaseInsensitive));
}

void GEExperimentDialog::setChecked(const std::function<bool(int)>& pred) {
    for (int r = 0; r < m_datasets->rowCount(); ++r) {
        if (m_datasets->isRowHidden(r)) continue;
        m_datasets->item(r, 0)->setCheckState(pred(r) ? Qt::Checked : Qt::Unchecked);
    }
}

void GEExperimentDialog::browseData() {
    const QString d = QFileDialog::getExistingDirectory(this, "Data folder", resolve(m_dataDir->text()));
    if (d.isEmpty()) return;
    QString rel = QDir(m_workDir).relativeFilePath(d);
    m_dataDir->setText(rel.startsWith("..") ? d : rel);
    rescanDatasets();
}

void GEExperimentDialog::browseOutput() {
    const QString f = QFileDialog::getSaveFileName(this, "Results prefix (files <prefix>_raw.csv, …)",
                                                   resolve(m_output->text()), "All files (*)");
    if (f.isEmpty()) return;
    QString p = f; if (p.endsWith("_raw.csv")) p.chop(8);
    QString rel = QDir(m_workDir).relativeFilePath(p);
    m_output->setText(rel.startsWith("..") ? p : rel);
    readResults(); refreshModels(); fillDatasets();
}

GEExperiment GEExperimentDialog::experiment() const {
    GEExperiment e = m_in;
    e.name = m_name->text().trimmed();
    e.description = m_desc->text().trimmed();
    e.development.clear();
    for (QString d : m_dev->text().split(QRegularExpression("[,\\s]+"), Qt::SkipEmptyParts)) e.development << d.trimmed();
    e.dataDir = m_dataDir->text().trimmed().replace('\\', '/');
    e.output = m_output->text().trimmed().replace('\\', '/');
    e.folds = m_folds->text().trimmed();
    e.seeds = m_seeds->value();
    e.configs.clear();
    for (auto* cb : m_cfgChecks) if (cb->isChecked()) e.configs << cb->text();
    e.options.clear();
    for (int r = 0; r < m_options->rowCount(); ++r) {
        const QString v = m_options->item(r, 1)->text().trimmed();
        if (!v.isEmpty()) e.options[m_options->item(r, 0)->text()] = v;
    }
    // models: keep the ticks by id; if the budget mode changed, map the ids that exist in both modes
    QSet<QString> run;
    for (const QString& id : m_models->checked()) run.insert(id);
    e.models = geDefaultModels(e.fixedBudget());
    for (auto& m : e.models) {
        QString id = m.id;
        if (!run.contains(id)) {   // T <-> eq equivalents
            if (id.endsWith("_T") || id.contains("_T_")) id.replace("_T", "_eq");
            else if (id.endsWith("_eq")) id.replace("_eq", "_T");
        }
        m.run = run.contains(m.id) || run.contains(id);
    }
    e.classification.clear(); e.regression.clear();
    for (const DsInfo& i : m_all)
        if (m_selected.contains(i.name)) {
            if (i.task == "classification") e.classification << i.name;
            else if (i.task == "regression") e.regression << i.name;
        }
    return e;
}

void GEExperimentDialog::accept() {
    const GEExperiment e = experiment();
    QStringList problems;
    if (e.name.isEmpty()) problems << "Give the experiment a name.";
    if (e.output.isEmpty()) problems << "Set the results prefix (e.g. results/bench64).";
    if (!QDir(resolve(e.dataDir)).exists()) problems << "The data folder does not exist.";
    if (e.datasets().isEmpty()) problems << "Select at least one dataset.";
    if (e.runModels().isEmpty()) problems << "Select at least one model.";
    bool geModel = false;
    for (const auto& m : e.models) if (m.run && !geBaselineIds().contains(m.id)) geModel = true;
    if (geModel && e.configs.isEmpty()) problems << "Models that use the GE search need at least one grammar configuration.";
    QStringList unusable;
    for (const DsInfo& i : m_all)
        if (m_selected.contains(i.name) && i.task != "classification" && i.task != "regression") unusable << i.name;
    if (!problems.isEmpty()) { QMessageBox::warning(this, "Experiment", problems.join("\n")); return; }
    if (!unusable.isEmpty() &&
        QMessageBox::question(this, "Experiment", QString("These selected datasets cannot be used (missing or unreadable) "
                              "and will be left out:\n%1\n\nContinue?").arg(unusable.join(", "))) != QMessageBox::Yes)
        return;
    QDialog::accept();
}

} // namespace NeuralStudio
