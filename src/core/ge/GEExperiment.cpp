// GEExperiment.cpp — experiment files of Mode 6 (see GEExperiment.h).
#include "GEExperiment.h"
#include "GEReport.h"
#include "Baselines.h"
#include "core/dataset/DatasetLoader.h"
#include "core/nn/NeuralNetwork.h"
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QHash>
#include <QSet>
#include <algorithm>
#include <exception>

namespace NeuralStudio {

namespace {
QString quoteIfNeeded(const QString& s) {
    if (s.contains(' ') || s.contains('\t')) return "\"" + s + "\"";
    return s;
}
QStringList toStringList(const QJsonValue& v) {
    QStringList out;
    if (v.isArray()) for (const QJsonValue& x : v.toArray()) { const QString s = x.toString().trimmed(); if (!s.isEmpty()) out << s; }
    else if (v.isString()) for (const QString& s : v.toString().split(',', Qt::SkipEmptyParts)) out << s.trimmed();
    return out;
}
// Option keys accepted in "options" (all ge_batch options except the ones the file sets itself)
const QStringList kOptionKeys = {"gens", "pop", "inner-pop", "inner-iters", "final-pop", "final-iters", "topk",
    "patience", "parsimony", "metric", "optimizer", "val-split", "bound", "baseline-hidden", "baseline-act",
    "total-budget", "search-frac", "polish-frac", "bound-mode", "fan-k", "lamarck", "finalists",
    "inner-polish", "inner-polish-evals", "se-rule", "ens-pool", "ens-max"};
} // namespace

QStringList GEExperiment::runModels() const {
    QStringList out;
    for (const auto& m : models) if (m.run) out << m.id;
    return out;
}

QStringList geClassicGeMethodIds(bool fixedBudget) {
    return fixedBudget
        ? QStringList{"standard", "standard_T", "grid_T", "ge1", "ge1_T", "ge1_T_nofeat", "ge1_T_nobounds"}
        : QStringList{"standard", "standard_eq", "grid_eq", "ge1", "ge1_nobounds", "ge1_eq", "ge1_nobounds_eq"};
}

QStringList geAdaptiveMethodIds() {
    return {"standard_T_ad", "std_ens_T", "ge1_T_ad", "ens_T", "ens_T_nofeat", "ens_T_nobounds", "ens_T_div", "ens_T_cv", "std_ens_T_cv",
            "sage", "sage_noemb", "sage_nofeat", "sage_nobounds", "sage_one", "sage_std"};
}

QStringList geGeMethodIds(bool fixedBudget) {
    return fixedBudget ? geClassicGeMethodIds(true) + geAdaptiveMethodIds() : geClassicGeMethodIds(false);
}

QString geFoldsToText(QList<int> folds) {
    std::sort(folds.begin(), folds.end());
    folds.erase(std::unique(folds.begin(), folds.end()), folds.end());
    QStringList parts;
    for (int i = 0; i < folds.size();) {
        int j = i;
        while (j + 1 < folds.size() && folds[j + 1] == folds[j] + 1) ++j;
        parts << (j > i ? QString("%1-%2").arg(folds[i]).arg(folds[j]) : QString::number(folds[i]));
        i = j + 1;
    }
    return parts.join(',');
}

bool geExperimentFromRaw(const QString& rawPath, GEExperiment& e, QString& error, QStringList& warnings) {
    warnings.clear();
    QString prefix = rawPath;
    if (prefix.endsWith("_raw.csv")) prefix.chop(8);
    const QString json = prefix + "_experiment.json";
    if (QFileInfo::exists(json)) return geLoadExperiment(json, e, error, warnings);

    QVector<GERawRow> rows;
    if (!geLoadRawCsv(rawPath, rows, error)) return false;
    e = GEExperiment();
    e.name = QFileInfo(prefix).fileName();
    e.output = prefix;
    e.dataDir = "sample_data/tenfolding";
    QSet<QString> cls, reg, cfgs, methods; QList<int> folds; int seeds = 1;
    QStringList dsOrder, cfgOrder;
    for (const GERawRow& r : rows) {
        if (!cls.contains(r.ds) && !reg.contains(r.ds)) dsOrder << r.ds;
        (r.cls ? cls : reg).insert(r.ds);
        if (!cfgs.contains(r.cfg)) cfgOrder << r.cfg;
        cfgs.insert(r.cfg); methods.insert(r.method);
        if (!folds.contains(r.fold)) folds << r.fold;
        seeds = std::max(seeds, r.seed);
    }
    for (const QString& d : dsOrder) (cls.contains(d) ? e.classification : e.regression) << d;
    e.configs.clear();
    for (const QString& c : cfgOrder) if (c != "baselines") e.configs << c;
    if (e.configs.isEmpty()) e.configs << "unified-ge";
    e.folds = geFoldsToText(folds);
    e.seeds = seeds;
    bool fixed = false;
    for (const QString& m : methods) fixed = fixed || m.endsWith("_T") || m.contains("_T_");
    if (fixed) e.options["total-budget"] = "600000";
    e.models = geDefaultModels(fixed);
    for (auto& m : e.models) m.run = methods.contains(m.id);
    warnings << QString("No %1 next to these results: the data folder (%2) and the options are defaults. "
                        "Check them in Edit… before adding GE models, so new rows use the same settings.")
                    .arg(QFileInfo(json).fileName(), e.dataDir);
    return true;
}

QVector<GEExperimentModel> geDefaultModels(bool fixedBudget) {
    QVector<GEExperimentModel> out;
    // the self-adaptive rows are listed but off by default (they cost extra training)
    for (const QString& id : geGeMethodIds(fixedBudget))
        out.append({id, geMethodInfo(id).bnf, !geAdaptiveMethodIds().contains(id)});
    for (const QString& id : geBaselineIds()) out.append({id, geMethodInfo(id).bnf, true});
    return out;
}

GEExperiment geBenchmarkExperiment() {
    GEExperiment e;
    e.name = "bench64";
    e.description = "The datasets of the BNF-training paper available in tenfolding (hepatitis and plastic excluded: "
                    "their files do not match the task of the paper). Fixed total budget T per run.";
    e.dataDir = "sample_data/tenfolding";
    e.output = "results/bench64";
    e.folds = "1-10";
    e.seeds = 1;
    e.classification = QStringList{"alcohol", "australian", "bands", "cleveland", "circular", "dermatology", "ecoli",
        "haberman", "hayes-roth", "heart", "heartAttack", "housevotes", "ionosphere", "liverdisorder", "lymography",
        "mammographic", "parkinsons", "pima", "popfailures", "regions2", "saheart", "statheart", "student", "tae",
        "transfusion", "wdbc", "wine", "z_f_s", "zo_nf_s", "z_o_n_f_s", "zonf_s", "zoo",
        "segment", "spiral", "spambase", "phoneme", "page-blocks", "ring", "magic"};
    e.regression = QStringList{"auto", "BK", "BL", "baseball", "concrete", "dee", "eu", "FA", "friedman", "fy", "ho",
        "housing", "laser", "lw", "mortgage", "pl", "realestate", "SN", "stock", "treasury", "tz", "ve", "airfoil",
        "quake", "abalone"};
    e.configs = {"unified-ge"};
    e.options = {{"gens", "40"}, {"pop", "40"}, {"inner-pop", "30"}, {"inner-iters", "60"}, {"final-pop", "80"},
                 {"final-iters", "200"}, {"total-budget", "600000"}, {"patience", "15"}};
    e.models = geDefaultModels(true);
    return e;
}

bool geLoadExperiment(const QString& path, GEExperiment& e, QString& error, QStringList& warnings) {
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) { error = "Cannot open " + path; e = GEExperiment(); warnings.clear(); return false; }
    return geExperimentFromJson(f.readAll(), QFileInfo(path).completeBaseName(), e, error, warnings);
}

bool geExperimentFromJson(const QByteArray& bytes, const QString& nameFallback, GEExperiment& e, QString& error,
                          QStringList& warnings) {
    e = GEExperiment();
    warnings.clear();
    QJsonParseError pe;
    const QJsonDocument doc = QJsonDocument::fromJson(bytes, &pe);
    if (pe.error != QJsonParseError::NoError) {
        error = QString("JSON error at offset %1: %2").arg(pe.offset).arg(pe.errorString()); return false;
    }
    if (!doc.isObject()) { error = "The experiment file must contain a JSON object."; return false; }
    const QJsonObject o = doc.object();
    static const QSet<QString> known = {"name", "description", "data_dir", "output", "folds", "seeds", "datasets",
                                        "configs", "models", "options", "development_datasets"};
    for (const QString& k : o.keys()) if (!known.contains(k)) warnings << QString("Unknown key '%1' ignored.").arg(k);

    e.name = o.value("name").toString(nameFallback);
    e.description = o.value("description").toString();
    e.dataDir = o.value("data_dir").toString();
    e.output = o.value("output").toString();
    if (e.dataDir.isEmpty()) { error = "Missing \"data_dir\" (folder with <dataset>_<fold>.train files)."; return false; }
    if (e.output.isEmpty()) { error = "Missing \"output\" (results prefix, e.g. results/bench64)."; return false; }
    if (o.contains("folds")) {
        const QJsonValue fv = o.value("folds");
        e.folds = fv.isDouble() ? QString("1-%1").arg(fv.toInt()) : fv.toString();
    }
    if (o.contains("seeds")) e.seeds = o.value("seeds").toInt(1);
    if (e.seeds < 1) { error = "\"seeds\" must be a positive integer."; return false; }

    const QJsonValue dsv = o.value("datasets");
    if (dsv.isObject()) {
        const QJsonObject d = dsv.toObject();
        for (const QString& k : d.keys())
            if (k != "classification" && k != "regression") warnings << QString("datasets.%1 ignored (use classification / regression).").arg(k);
        e.classification = toStringList(d.value("classification"));
        e.regression = toStringList(d.value("regression"));
    } else if (dsv.isArray() || dsv.isString()) {
        error = "\"datasets\" must be an object: { \"classification\": [...], \"regression\": [...] }."; return false;
    }
    if (e.datasets().isEmpty()) { error = "No datasets: fill datasets.classification and/or datasets.regression."; return false; }
    {
        QSet<QString> seen;
        for (const QString& d : e.datasets()) {
            if (seen.contains(d)) warnings << QString("Dataset '%1' is listed twice.").arg(d);
            seen.insert(d);
        }
    }
    if (o.contains("development_datasets")) {
        e.development = toStringList(o.value("development_datasets"));
        for (const QString& d : e.development)
            if (!e.datasets().contains(d)) warnings << QString("Development dataset '%1' is not one of the datasets.").arg(d);
    }
    if (o.contains("configs")) e.configs = toStringList(o.value("configs"));
    if (e.configs.isEmpty()) e.configs = {"unified-ge"};

    const QJsonValue ov = o.value("options");
    if (ov.isObject()) {
        const QJsonObject opts = ov.toObject();
        for (const QString& k0 : opts.keys()) {
            const QString k = k0.startsWith("--") ? k0.mid(2) : k0;
            if (!kOptionKeys.contains(k)) { warnings << QString("Unknown option '%1' ignored.").arg(k0); continue; }
            const QJsonValue v = opts.value(k0);
            QString s;
            if (v.isDouble()) {
                const double d = v.toDouble();
                s = (d == static_cast<qint64>(d)) ? QString::number(static_cast<qint64>(d)) : QString::number(d, 'g', 12);
            } else if (v.isBool()) s = v.toBool() ? "1" : "0";
            else s = v.toString();
            e.options[k] = s;
        }
    }

    // models
    const bool fixed = e.fixedBudget();
    const QVector<GEExperimentModel> valid = geDefaultModels(fixed);
    QStringList validIds;
    for (const auto& m : valid) validIds << m.id;
    if (o.contains("models")) {
        const QJsonValue mv = o.value("models");
        if (!mv.isArray()) { error = "\"models\" must be an array of { \"id\", \"bnf\", \"run\" }."; return false; }
        for (const QJsonValue& x : mv.toArray()) {
            GEExperimentModel m;
            if (x.isString()) { m.id = x.toString(); m.bnf = geMethodInfo(m.id).bnf; }
            else if (x.isObject()) {
                const QJsonObject mo = x.toObject();
                m.id = mo.value("id").toString();
                const bool truth = geMethodInfo(m.id).bnf;
                m.bnf = mo.contains("bnf") ? mo.value("bnf").toBool() : truth;
                m.run = mo.value("run").toBool(true);
                if (m.bnf != truth) {
                    warnings << QString("Model '%1' %2 the BNF grammar; the file says \"bnf\": %3 — corrected.")
                                    .arg(m.id, truth ? "uses" : "does not use", m.bnf ? "true" : "false");
                    m.bnf = truth;
                }
            } else continue;
            if (!validIds.contains(m.id)) {
                warnings << QString("Model '%1' is not produced in the %2 mode (valid: %3) — ignored.")
                                .arg(m.id, fixed ? "fixed-total-budget" : "equal-budget", validIds.join(", "));
                continue;
            }
            e.models.append(m);
        }
        // models not mentioned are listed as not run, so the file stays explicit
        for (const auto& v : valid) {
            bool found = false;
            for (const auto& m : e.models) found = found || m.id == v.id;
            if (!found) { GEExperimentModel m = v; m.run = false; e.models.append(m); }
        }
    } else {
        e.models = valid;
    }
    if (e.runModels().isEmpty()) { error = "Every model has \"run\": false — nothing to compute."; return false; }
    bool anyBnf = false;
    for (const auto& m : e.models) anyBnf = anyBnf || (m.run && m.bnf);
    if (!anyBnf) warnings << "No BNF model is run; the GE search will still run (it is needed for the comparison).";
    return true;
}

bool geSaveExperiment(const QString& path, const GEExperiment& e, QString& error) {
    QFile f(path);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate)) { error = "Cannot write " + path; return false; }
    f.write(geExperimentToJson(e));
    return true;
}

QByteArray geExperimentToJson(const GEExperiment& e) {
    QJsonObject o;
    o["name"] = e.name;
    if (!e.description.isEmpty()) o["description"] = e.description;
    o["data_dir"] = e.dataDir;
    o["output"] = e.output;
    o["folds"] = e.folds;
    o["seeds"] = e.seeds;
    QJsonObject ds;
    ds["classification"] = QJsonArray::fromStringList(e.classification);
    ds["regression"] = QJsonArray::fromStringList(e.regression);
    o["datasets"] = ds;
    if (!e.development.isEmpty()) o["development_datasets"] = QJsonArray::fromStringList(e.development);
    o["configs"] = QJsonArray::fromStringList(e.configs);
    QJsonArray models;
    for (const auto& m : e.models) {
        QJsonObject mo;
        mo["id"] = m.id;
        mo["label"] = geMethodInfo(m.id).label;
        mo["bnf"] = m.bnf;
        mo["run"] = m.run;
        models.append(mo);
    }
    o["models"] = models;
    QJsonObject opts;
    for (auto it = e.options.begin(); it != e.options.end(); ++it) {
        bool okI = false, okD = false;
        const qint64 iv = it.value().toLongLong(&okI);
        const double dv = it.value().toDouble(&okD);
        if (okI) opts[it.key()] = static_cast<double>(iv);
        else if (okD) opts[it.key()] = dv;
        else opts[it.key()] = it.value();
    }
    o["options"] = opts;
    return QJsonDocument(o).toJson(QJsonDocument::Indented);
}

QString geExperimentToCommand(const GEExperiment& e) {
    QStringList a;
    a << "--dir" << quoteIfNeeded(e.dataDir)
      << "--datasets" << e.datasets().join(',')
      << "--folds" << e.folds
      << "--seeds" << QString::number(e.seeds)
      << "--configs" << [&] {
             // grammar configurations only if a GE model runs; "baselines" if a reference model runs
             const QStringList base = geBaselineIds();
             bool anyGe = false, anyBase = false;
             for (const auto& m : e.models) if (m.run) (base.contains(m.id) ? anyBase : anyGe) = true;
             QStringList cfgs;
             for (const QString& c : e.configs) if (c != "baselines" && anyGe) cfgs << c;
             if (anyBase) cfgs << "baselines";
             if (cfgs.isEmpty()) cfgs = e.configs;
             return cfgs.join(',');
         }();
    for (const QString& k : kOptionKeys)
        if (e.options.contains(k)) a << "--" + k << quoteIfNeeded(e.options.value(k));
    // --methods is left out only when exactly the classic rows and the reference models run
    // (what a configuration computes by default), so older commands keep their meaning.
    const QStringList run = e.runModels();
    QStringList classic = geClassicGeMethodIds(e.fixedBudget()) + geBaselineIds();
    bool all = true;
    for (const auto& m : e.models) {
        const bool shouldRun = classic.contains(m.id);
        all = all && m.run == shouldRun;
    }
    if (!all) a << "--methods" << run.join(',');
    if (!e.development.isEmpty()) a << "--dev-datasets" << e.development.join(',');
    a << "--out" << quoteIfNeeded(e.output);
    return a.join(' ');
}

QString geDetectDatasetTask(const QString& dir, const QString& ds, QString* detail) {
    const QString path = QString("%1/%2_1.train").arg(dir, ds);
    if (!QFileInfo::exists(path)) { if (detail) *detail = "file not found"; return QString(); }
    try {
        auto d = DatasetLoader::load(path);
        if (!d) { if (detail) *detail = "cannot read"; return QString(); }
        d->computeStatistics();
        const TaskType t = detectTask(d.get());
        const auto uv = d->uniqueOutputValues(0);
        if (t == TaskType::Regression) { if (detail) *detail = QString("%1 distinct target values").arg(uv.size()); return "regression"; }
        if (detail) *detail = QString("%1 classes").arg(uv.size());
        return "classification";
    } catch (const std::exception& ex) {
        if (detail) *detail = QString::fromStdString(ex.what());
        return QString();
    }
}

QString geExperimentSummary(const GEExperiment& e) {
    QString s;
    s += QString("Experiment: %1\n").arg(e.name);
    if (!e.description.isEmpty()) s += e.description + "\n";
    s += QString("Data folder: %1   Output: %2\n").arg(e.dataDir, e.output);
    s += QString("Folds: %1   Seeds: %2   Configurations: %3\n").arg(e.folds).arg(e.seeds).arg(e.configs.join(", "));
    s += QString("Budget mode: %1\n").arg(e.fixedBudget() ? "fixed total budget T = " + e.options.value("total-budget")
                                                         : QString("equal budget (no total-budget)"));
    s += QString("\nClassification datasets (%1): %2\n").arg(e.classification.size()).arg(e.classification.join(", "));
    s += QString("Regression datasets (%1): %2\n").arg(e.regression.size()).arg(e.regression.join(", "));
    s += "\nModels:\n";
    for (const auto& m : e.models)
        s += QString("  %1 %2  BNF: %3   %4\n").arg(m.run ? "[run] " : "[skip]")
                 .arg(geMethodInfo(m.id).label, -22).arg(m.bnf ? "yes" : "no ").arg(geMethodInfo(m.id).description);
    if (!e.options.isEmpty()) {
        s += "\nOptions:";
        for (auto it = e.options.begin(); it != e.options.end(); ++it) s += QString(" --%1 %2").arg(it.key(), it.value());
        s += "\n";
    }
    return s;
}

} // namespace NeuralStudio

namespace NeuralStudio {
int geExperimentDone(const GEExperiment& e, const QVector<GERawRow>& rows, int& total) {
    QList<int> folds;
    for (const QString& p : e.folds.split(',', Qt::SkipEmptyParts)) {
        const QStringList ab = p.split('-');
        if (ab.size() == 2) for (int i = ab[0].toInt(); i <= ab[1].toInt(); ++i) folds << i; else folds << p.toInt();
    }
    const QStringList base = geBaselineIds();
    QStringList geRun, baseRun;
    for (const auto& m : e.models) if (m.run) (base.contains(m.id) ? baseRun : geRun) << m.id;
    QStringList cfgs;
    if (!geRun.isEmpty()) cfgs << e.configs;
    if (!baseRun.isEmpty()) cfgs << "baselines";
    QHash<QString, QSet<QString>> have;
    for (const GERawRow& r : rows) have[QString("%1|%2|%3|%4").arg(r.ds).arg(r.fold).arg(r.cfg).arg(r.seed)].insert(r.method);
    total = 0; int done = 0;
    for (const QString& ds : e.datasets()) for (int f : folds) for (const QString& c : cfgs) for (int s = 1; s <= e.seeds; ++s) {
        ++total;
        const QSet<QString>& h = have.value(QString("%1|%2|%3|%4").arg(ds).arg(f).arg(c).arg(s));
        bool ok = !h.isEmpty();
        for (const QString& m : (c == "baselines" ? baseRun : geRun)) ok = ok && h.contains(m);
        if (ok) ++done;
    }
    return done;
}
} // namespace NeuralStudio
