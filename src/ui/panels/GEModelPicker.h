#pragma once
// GEModelPicker.h — click-to-select list of the machine-learning models of Mode 6.
//
// The counterpart of the dataset lists: every model the engine can run is shown in two groups,
// "Without BNF" and "With BNF", each with a check box, its description and (optionally) how many
// result rows already exist. A group check box selects / clears the whole group. Used on the
// Single run page, on the Experiment page and in the experiment editor.
#include <QMap>
#include <QString>
#include <QStringList>
#include <QVector>
#include <QWidget>

class QTreeWidget; class QTreeWidgetItem; class QLabel;

namespace NeuralStudio {

class GEModelPicker : public QWidget {
    Q_OBJECT
public:
    struct Entry {
        QString id;
        QString label;
        bool    bnf = false;
        QString description;
    };

    explicit GEModelPicker(QWidget* parent = nullptr);

    // Entries in report order; the ticks of ids that exist in both lists are kept.
    void setEntries(const QVector<Entry>& entries);
    // Entries of the method catalog (GEReport) for these ids.
    static QVector<Entry> catalogEntries(const QStringList& ids);

    void        setChecked(const QStringList& ids);   // exactly these ids ticked
    QStringList checked() const;                      // ticked ids (without-BNF group first)
    QStringList ids() const;                          // all ids (without-BNF group first)

    // Narrow layout (no Description column; the description stays in the tooltip).
    void setCompact(bool on);

    // Optional third column, e.g. "Results (rows)" -> {id: "20"}. An empty title hides the column.
    void setInfoColumn(const QString& title, const QMap<QString, QString>& values);

signals:
    void changed();    // a tick changed (by the user or by the buttons)

private:
    void updateCount();
    void notifyChanged();
    QTreeWidgetItem* groupItem(bool bnf) const;

    QTreeWidget*     m_tree = nullptr;
    QTreeWidgetItem *m_noBnf = nullptr, *m_withBnf = nullptr;
    QLabel*          m_count = nullptr;
    bool             m_filling = false;
    bool             m_pending = false;
    bool             m_compact = false;
};

} // namespace NeuralStudio
