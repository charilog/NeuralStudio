#include "CliArgs.h"

#include <algorithm>
#include <limits>

namespace NeuralStudio::cli {

namespace {

// Levenshtein distance, for "did you mean --xyz?"
int editDistance(const QString& a, const QString& b) {
    const int n = a.size(), m = b.size();
    QVector<int> prev(m + 1), cur(m + 1);
    for (int j = 0; j <= m; ++j) prev[j] = j;
    for (int i = 1; i <= n; ++i) {
        cur[0] = i;
        for (int j = 1; j <= m; ++j) {
            const int sub = prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1);
            cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1, sub});
        }
        std::swap(prev, cur);
    }
    return prev[m];
}

const OptSpec* findSpec(const QVector<OptSpec>& specs, const QString& name) {
    for (const auto& s : specs)
        if (s.name == name) return &s;
    return nullptr;
}

QString suggestion(const QVector<OptSpec>& specs, const QString& name) {
    QString best;
    int bestD = std::numeric_limits<int>::max();
    for (const auto& s : specs) {
        const int d = editDistance(name, s.name);
        if (d < bestD) { bestD = d; best = s.name; }
    }
    return (bestD <= 3 && !best.isEmpty()) ? QString(" Did you mean --%1?").arg(best) : QString();
}

bool looksLikeNumber(const QString& s) {
    bool ok = false;
    s.toDouble(&ok);
    return ok;
}

} // namespace

bool Args::parse(const QStringList& argv, const QVector<OptSpec>& specs, QString& error) {
    m_values.clear();
    m_positional.clear();
    m_errors.clear();
    for (int i = 0; i < argv.size(); ++i) {
        const QString a = argv[i];
        if (a == "--") {                       // everything after "--" is positional
            for (int j = i + 1; j < argv.size(); ++j) m_positional << argv[j];
            break;
        }
        if (!a.startsWith("--") || a.size() == 2) {
            if (a.startsWith('-') && a.size() > 1 && !looksLikeNumber(a)) {
                error = QString("Unknown option '%1' (options start with --).").arg(a);
                return false;
            }
            m_positional << a;
            continue;
        }
        QString name = a.mid(2), value;
        bool hasInline = false;
        const int eq = name.indexOf('=');
        if (eq >= 0) { value = name.mid(eq + 1); name = name.left(eq); hasInline = true; }
        const OptSpec* spec = findSpec(specs, name);
        if (!spec) {
            error = QString("Unknown option --%1.%2").arg(name, suggestion(specs, name));
            return false;
        }
        if (spec->valueName.isEmpty()) {
            if (hasInline) { error = QString("Option --%1 does not take a value.").arg(name); return false; }
            m_values[name] << "1";
            continue;
        }
        if (!hasInline) {
            if (i + 1 >= argv.size() || (argv[i + 1].startsWith("--") && argv[i + 1].size() > 2)) {
                error = QString("Option --%1 needs a value (%2).").arg(name, spec->valueName);
                return false;
            }
            value = argv[++i];
        }
        m_values[name] << value;
    }
    return true;
}

QString Args::str(const QString& name, const QString& def) const {
    const auto it = m_values.constFind(name);
    if (it == m_values.constEnd() || it->isEmpty()) return def;
    return it->last();   // the last occurrence wins
}

QStringList Args::list(const QString& name) const {
    QStringList out;
    const auto it = m_values.constFind(name);
    if (it == m_values.constEnd()) return out;
    for (const QString& v : *it)
        for (const QString& part : v.split(',', Qt::SkipEmptyParts)) {
            const QString t = part.trimmed();
            if (!t.isEmpty()) out << t;
        }
    return out;
}

int Args::integer(const QString& name, int def, int lo, int hi) const {
    if (!has(name)) return def;
    bool ok = false;
    const int v = str(name).trimmed().toInt(&ok);
    if (!ok) { addError(QString("--%1 expects an integer, got '%2'.").arg(name, str(name))); return def; }
    if (v < lo || v > hi) {
        addError(QString("--%1 must be between %2 and %3 (got %4).").arg(name).arg(lo).arg(hi).arg(v));
        return def;
    }
    return v;
}

qint64 Args::bigInt(const QString& name, qint64 def, qint64 lo, qint64 hi) const {
    if (!has(name)) return def;
    bool ok = false;
    const qint64 v = str(name).trimmed().toLongLong(&ok);
    if (!ok) { addError(QString("--%1 expects an integer, got '%2'.").arg(name, str(name))); return def; }
    if (v < lo || v > hi) {
        addError(QString("--%1 must be between %2 and %3 (got %4).").arg(name).arg(lo).arg(hi).arg(v));
        return def;
    }
    return v;
}

double Args::number(const QString& name, double def, double lo, double hi) const {
    if (!has(name)) return def;
    bool ok = false;
    const double v = str(name).trimmed().toDouble(&ok);
    if (!ok) { addError(QString("--%1 expects a number, got '%2'.").arg(name, str(name))); return def; }
    if (!(v >= lo && v <= hi)) {
        addError(QString("--%1 must be between %2 and %3 (got %4).").arg(name).arg(lo).arg(hi).arg(v));
        return def;
    }
    return v;
}

bool Args::boolean(const QString& name, bool def) const {
    if (!has(name)) return def;
    const QString v = str(name).trimmed().toLower();
    if (v == "1" || v == "on" || v == "yes" || v == "true") return true;
    if (v == "0" || v == "off" || v == "no" || v == "false") return false;
    addError(QString("--%1 expects 1/0 (on/off), got '%2'.").arg(name, str(name)));
    return def;
}

QString Args::choice(const QString& name, const QString& def, const QStringList& allowed) const {
    if (!has(name)) return def;
    const QString v = str(name).trimmed().toLower();
    for (const QString& a : allowed)
        if (a.toLower() == v) return a.toLower();
    addError(QString("--%1: '%2' is not one of: %3.").arg(name, str(name), allowed.join(", ")));
    return def;
}

QString formatOptions(const QVector<OptSpec>& specs) {
    QString out;
    QString group = "\x01";
    int width = 0;
    for (const auto& s : specs) {
        const int w = 2 + s.name.size() + (s.valueName.isEmpty() ? 0 : 1 + s.valueName.size());
        width = std::max(width, w);
    }
    width = std::min(width, 34);
    for (const auto& s : specs) {
        if (s.group != group) {
            group = s.group;
            if (!group.isEmpty()) out += "\n" + group + ":\n";
        }
        QString left = "  --" + s.name + (s.valueName.isEmpty() ? QString() : " " + s.valueName);
        QString line = left;
        if (left.size() < width + 2) line += QString(width + 4 - left.size(), ' ');
        else line += "\n" + QString(width + 4, ' ');
        line += s.help;
        if (!s.def.isEmpty()) line += QString(" [default: %1]").arg(s.def);
        out += line + "\n";
    }
    return out;
}

} // namespace NeuralStudio::cli
