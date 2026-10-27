#pragma once
#include "core/stats/Statistics.h"
#include <QWidget>
#include <QString>
#include <QVector>
#include <QMap>
#include <QColor>

namespace NeuralStudio {

// ─── BatchPlotWidget ──────────────────────────────────────────────────────────
//  Custom painter-based boxplot widget.
//
//  Feed it setData(names, values) and it draws:
//    • One box per group (optimizer)
//    • Whiskers (1.5×IQR), outlier dots
//    • Optional significance brackets (annotated with p-value stars)
//    • Y-axis auto-scaled to data
//
//  For high-DPI PNG export use exportPng().
// ─────────────────────────────────────────────────────────────────────────────
class BatchPlotWidget : public QWidget {
    Q_OBJECT
public:
    explicit BatchPlotWidget(QWidget* parent = nullptr);

    // ── Data ─────────────────────────────────────────────────────────────────
    void setData(const QVector<QString>& names,
                 const QVector<QVector<double>>& values);

    void setYAxisLabel(const QString& label)    { m_yLabel = label; update(); }
    void setTitle(const QString& title)         { m_title  = title; update(); }
    void setLowerIsBetter(bool v)               { m_lowerBetter = v; update(); }

    // Significance brackets: pairs of (i,j) group indices with stars + raw p-value
    struct Bracket { int i; int j; QString stars; double pValue = -1.0; };
    void setBrackets(const QVector<Bracket>& b) { m_brackets = b; update(); }

    // ── Export ───────────────────────────────────────────────────────────────
    // Renders to PNG at 300 DPI; returns true on success.
    bool exportPng(const QString& path) const;

protected:
    void paintEvent(QPaintEvent*) override;

private:
    QVector<QString>              m_names;
    QVector<Statistics::BoxStats> m_boxes;
    QVector<Bracket>              m_brackets;
    QString m_yLabel;
    QString m_title;
    bool    m_lowerBetter = true;

    // sc=1.0 for screen, sc≈3.125 for 300 DPI export
    void renderTo(QPainter& p, const QRect& rect, double sc = 1.0) const;
    QColor rankColor(int rank, int total) const;  // teal→red gradient
};

} // namespace NeuralStudio
