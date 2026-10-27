#pragma once
// CliArgs.h — option table and parser of the NeuralStudio console program.
//
// Every command declares its options in a table (name, value placeholder, default,
// help text). The parser accepts "--name value", "--name=value" and bare flags,
// rejects unknown options (with a "did you mean" hint), and keeps everything that does
// not start with "--" as a positional argument (data files). The same table prints the
// help text, so help and parser can never disagree.
#include <QMap>
#include <QString>
#include <QStringList>
#include <QVector>

namespace NeuralStudio::cli {

struct OptSpec {
    QString name;        // without the leading "--"
    QString valueName;   // empty: a flag (no value)
    QString def;         // default value shown in the help ("" = none)
    QString help;
    QString group;       // heading in the help text
};

class Args {
public:
    // false + error when an option is unknown, a value is missing or a flag got a value
    bool parse(const QStringList& argv, const QVector<OptSpec>& specs, QString& error);

    bool    has(const QString& name) const { return m_values.contains(name); }
    bool    flag(const QString& name) const { return m_values.contains(name); }
    QString str(const QString& name, const QString& def = QString()) const;
    QStringList list(const QString& name) const;   // comma separated, every occurrence, trimmed

    // Typed getters: an unparsable or out-of-range value records an error (see firstError()).
    int     integer(const QString& name, int def, int lo, int hi) const;
    double  number (const QString& name, double def, double lo, double hi) const;
    qint64  bigInt (const QString& name, qint64 def, qint64 lo, qint64 hi) const;
    bool    boolean(const QString& name, bool def) const;   // 1/0, on/off, yes/no, true/false
    // value must be one of `allowed` (case-insensitive); returns it lower-case
    QString choice (const QString& name, const QString& def, const QStringList& allowed) const;

    const QStringList& positional() const { return m_positional; }
    QString firstError() const { return m_errors.isEmpty() ? QString() : m_errors.first(); }
    bool    ok() const { return m_errors.isEmpty(); }
    void    addError(const QString& e) const { m_errors << e; }

private:
    QMap<QString, QStringList> m_values;   // name -> values (in order; flags hold "1")
    QStringList                m_positional;
    mutable QStringList        m_errors;
};

// Help text generated from the option table.
QString formatOptions(const QVector<OptSpec>& specs);

} // namespace NeuralStudio::cli
