#include "GEModelPicker.h"
#include "core/ge/GEReport.h"
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QPushButton>
#include <QSet>
#include <QTimer>
#include <functional>
#include <QTreeWidget>
#include <QVBoxLayout>

namespace NeuralStudio {

namespace {
constexpr int kIdRole = Qt::UserRole;
constexpr int kColModel = 0, kColInfo = 1, kColDesc = 2;
}

GEModelPicker::GEModelPicker(QWidget* parent) : QWidget(parent) {
    auto* l = new QVBoxLayout(this);
    l->setContentsMargins(0, 0, 0, 0);
    l->setSpacing(4);

    auto* top = new QHBoxLayout;
    top->setSpacing(4);
    auto mk = [&](const QString& t, const QString& tip) {
        auto* b = new QPushButton(t);
        b->setToolTip(tip);
        b->setAutoDefault(false);
        top->addWidget(b);
        return b;
    };
    auto* bAll  = mk("All", "Tick every model");
    auto* bNone = mk("None", "Clear every model");
    auto* bBnf  = mk("With BNF", "Only the models that use the BNF grammar");
    auto* bNo   = mk("Without BNF", "Only the models that do not use the BNF grammar");
    top->addStretch();
    l->addLayout(top);
    m_count = new QLabel;
    m_count->setStyleSheet("color:palette(mid);");

    m_tree = new QTreeWidget;
    m_tree->setColumnCount(3);
    m_tree->setHeaderLabels({"Model", "", "Description"});
    m_tree->setRootIsDecorated(true);
    m_tree->setUniformRowHeights(true);
    m_tree->setSelectionMode(QAbstractItemView::NoSelection);
    m_tree->setColumnHidden(kColInfo, true);
    m_tree->header()->setStretchLastSection(true);
    m_tree->header()->setSectionResizeMode(kColModel, QHeaderView::ResizeToContents);
    m_tree->header()->setSectionResizeMode(kColInfo, QHeaderView::ResizeToContents);
    l->addWidget(m_tree, 1);
    l->addWidget(m_count);

    auto mkGroup = [&](const QString& title, const QString& tip) {
        auto* g = new QTreeWidgetItem(m_tree, {title});
        g->setFlags(Qt::ItemIsEnabled | Qt::ItemIsUserCheckable | Qt::ItemIsAutoTristate);
        g->setCheckState(kColModel, Qt::Unchecked);
        QFont f = g->font(kColModel); f.setBold(true); g->setFont(kColModel, f);
        g->setToolTip(kColModel, tip);
        g->setFirstColumnSpanned(true);
        g->setExpanded(true);
        return g;
    };
    m_noBnf = mkGroup("Without BNF", "Networks of the GE comparison and reference machine-learning models");
    m_withBnf = mkGroup("With BNF", "Models built or trained with the BNF grammar (grammatical evolution)");

    connect(m_tree, &QTreeWidget::itemChanged, this, [this](QTreeWidgetItem*, int col) {
        if (m_filling || col != kColModel) return;
        updateCount();
        notifyChanged();
    });
    // a click anywhere on a row toggles it (not only on the small box)
    connect(m_tree, &QTreeWidget::itemClicked, this, [this](QTreeWidgetItem* it, int col) {
        if (!it || !it->parent() || col == kColModel) return;
        it->setCheckState(kColModel, it->checkState(kColModel) == Qt::Checked ? Qt::Unchecked : Qt::Checked);
    });
    auto tickWhere = [this](const std::function<bool(QTreeWidgetItem*)>& pred) {
        m_filling = true;
        for (QTreeWidgetItem* g : {m_noBnf, m_withBnf})
            for (int i = 0; i < g->childCount(); ++i)
                g->child(i)->setCheckState(kColModel, pred(g->child(i)) ? Qt::Checked : Qt::Unchecked);
        m_filling = false;
        updateCount();
        notifyChanged();
    };
    connect(bAll, &QPushButton::clicked, this, [tickWhere] { tickWhere([](QTreeWidgetItem*) { return true; }); });
    connect(bNone, &QPushButton::clicked, this, [tickWhere] { tickWhere([](QTreeWidgetItem*) { return false; }); });
    connect(bBnf, &QPushButton::clicked, this, [this, tickWhere] { tickWhere([this](QTreeWidgetItem* c) { return c->parent() == m_withBnf; }); });
    connect(bNo, &QPushButton::clicked, this, [this, tickWhere] { tickWhere([this](QTreeWidgetItem* c) { return c->parent() == m_noBnf; }); });
    updateCount();
}

QVector<GEModelPicker::Entry> GEModelPicker::catalogEntries(const QStringList& ids) {
    QVector<Entry> v;
    for (const QString& id : ids) {
        const GEMethodInfo mi = geMethodInfo(id);
        v.push_back({id, mi.label, mi.bnf, mi.description});
    }
    return v;
}

void GEModelPicker::setCompact(bool on) {
    // narrow layout: the description moves to the tooltip of the model
    m_tree->setColumnHidden(kColDesc, on);
    m_compact = on;
}

QTreeWidgetItem* GEModelPicker::groupItem(bool bnf) const { return bnf ? m_withBnf : m_noBnf; }

void GEModelPicker::setEntries(const QVector<Entry>& entries) {
    const QStringList was = checked();
    const QSet<QString> keep(was.begin(), was.end());
    m_filling = true;
    for (QTreeWidgetItem* g : {m_noBnf, m_withBnf})
        while (g->childCount() > 0) delete g->takeChild(0);
    for (const Entry& e : entries) {
        auto* it = new QTreeWidgetItem(groupItem(e.bnf));
        it->setText(kColModel, e.label);
        it->setText(kColDesc, e.description);
        it->setToolTip(kColModel, QString("%1  (id: %2)").arg(e.description, e.id));
        it->setToolTip(kColDesc, e.description);
        it->setData(kColModel, kIdRole, e.id);
        it->setFlags(Qt::ItemIsEnabled | Qt::ItemIsUserCheckable);
        it->setCheckState(kColModel, keep.contains(e.id) ? Qt::Checked : Qt::Unchecked);
    }
    m_noBnf->setHidden(m_noBnf->childCount() == 0);
    m_withBnf->setHidden(m_withBnf->childCount() == 0);
    m_filling = false;
    updateCount();
}

void GEModelPicker::setChecked(const QStringList& ids) {
    m_filling = true;
    for (QTreeWidgetItem* g : {m_noBnf, m_withBnf})
        for (int i = 0; i < g->childCount(); ++i) {
            QTreeWidgetItem* c = g->child(i);
            c->setCheckState(kColModel, ids.contains(c->data(kColModel, kIdRole).toString()) ? Qt::Checked : Qt::Unchecked);
        }
    m_filling = false;
    updateCount();
}

QStringList GEModelPicker::checked() const {
    QStringList out;
    for (QTreeWidgetItem* g : {m_noBnf, m_withBnf})
        for (int i = 0; i < g->childCount(); ++i)
            if (g->child(i)->checkState(kColModel) == Qt::Checked) out << g->child(i)->data(kColModel, kIdRole).toString();
    return out;
}

QStringList GEModelPicker::ids() const {
    QStringList out;
    for (QTreeWidgetItem* g : {m_noBnf, m_withBnf})
        for (int i = 0; i < g->childCount(); ++i) out << g->child(i)->data(kColModel, kIdRole).toString();
    return out;
}

void GEModelPicker::setInfoColumn(const QString& title, const QMap<QString, QString>& values) {
    if (m_compact && !title.isEmpty()) m_tree->setColumnHidden(kColDesc, true);
    m_tree->setColumnHidden(kColInfo, title.isEmpty());
    m_tree->headerItem()->setText(kColInfo, title);
    for (QTreeWidgetItem* g : {m_noBnf, m_withBnf})
        for (int i = 0; i < g->childCount(); ++i) {
            QTreeWidgetItem* c = g->child(i);
            c->setText(kColInfo, values.value(c->data(kColModel, kIdRole).toString()));
            c->setTextAlignment(kColInfo, Qt::AlignRight | Qt::AlignVCenter);
        }
}

// One changed() per user action (a group tick changes every child).
void GEModelPicker::notifyChanged() {
    if (m_pending) return;
    m_pending = true;
    QTimer::singleShot(0, this, [this] { m_pending = false; emit changed(); });
}

void GEModelPicker::updateCount() {
    const bool was = m_filling;
    m_filling = true;
    int with = 0, without = 0;
    for (int i = 0; i < m_withBnf->childCount(); ++i) with += m_withBnf->child(i)->checkState(kColModel) == Qt::Checked;
    for (int i = 0; i < m_noBnf->childCount(); ++i) without += m_noBnf->child(i)->checkState(kColModel) == Qt::Checked;
    m_count->setText(QString("%1 selected: %2 with BNF, %3 without").arg(with + without).arg(with).arg(without));
    m_noBnf->setText(kColModel, QString("Without BNF  (%1 / %2)").arg(without).arg(m_noBnf->childCount()));
    m_withBnf->setText(kColModel, QString("With BNF  (%1 / %2)").arg(with).arg(m_withBnf->childCount()));
    m_filling = was;
}

} // namespace NeuralStudio
