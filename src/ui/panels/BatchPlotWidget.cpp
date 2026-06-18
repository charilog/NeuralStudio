#include "BatchPlotWidget.h"
#include <QPainter>
#include <QImage>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace NeuralStudio {

BatchPlotWidget::BatchPlotWidget(QWidget* parent)
    : QWidget(parent)
{
    setMinimumSize(320, 260);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    setAutoFillBackground(false);
}

void BatchPlotWidget::setData(const QVector<QString>& names,
                               const QVector<QVector<double>>& values)
{
    m_names = names;
    m_boxes.clear();
    for (const auto& v : values) {
        std::vector<double> sv(v.begin(), v.end());
        m_boxes.push_back(Statistics::boxStats(sv));
    }
    update();
}

QColor BatchPlotWidget::rankColor(int rank, int total) const {
    if (total <= 1) return QColor("#1D9E75");
    double t = static_cast<double>(rank) / std::max(1, total - 1);
    if (t < 0.5) {
        double u = t * 2.0;
        return QColor(int(29+u*(232-29)), int(158+u*(128-158)), int(117+u*(0-117)));
    }
    double u = (t-0.5)*2.0;
    return QColor(int(232+u*(192-232)), int(128+u*(57-128)), 0);
}

void BatchPlotWidget::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.setRenderHints(QPainter::Antialiasing|QPainter::TextAntialiasing);
    renderTo(p, rect());
}

// ─────────────────────────────────────────────────────────────────────────────
//  Layout (top → bottom):
//    padT | title | [bracket zone] | [box plot area] | x-labels | padB
//
//  Brackets are drawn ABOVE the boxes (scientific paper style).
//  Each bracket row: text ABOVE the horizontal connecting bar.
//
//  Font sizes: use setPixelSize() so sc (scale factor) controls them exactly.
//              sc=1.0 for screen, sc=SCALE for PNG export.
// ─────────────────────────────────────────────────────────────────────────────
void BatchPlotWidget::renderTo(QPainter& p, const QRect& r, double sc) const {
    p.fillRect(r, QColor(28,30,44));

    if (m_names.isEmpty() || m_boxes.isEmpty()) {
        p.setPen(QColor(160,160,180));
        p.drawText(r, Qt::AlignCenter, "No data");
        return;
    }
    const int n = m_names.size();

    // ── Layout constants (all scaled by sc) ──────────────────────────────────
    auto px = [&](double v){ return int(v * sc + 0.5); };

    const int ylabW  = px(24);
    const int ynumW  = px(58);
    const int padR   = px(16);
    const int padT   = px(16);
    const int titleH = m_title.isEmpty() ? 0 : px(30);
    const int xlabH  = px(46);
    const int padB   = px(12);

    // Bracket zone: one row per bracket, drawn ABOVE the boxes
    const int nBk    = std::min(6, (int)m_brackets.size());
    const int bkRowH = px(28);               // height of one bracket row
    const int bkZone = nBk * bkRowH;        // total reserved above boxes

    const int marginL = ylabW + ynumW;
    const int plotL   = r.left() + marginL;
    const int plotR   = r.right() - padR;
    const int plotW   = plotR - plotL;

    // plotT = start of the COMBINED bracket+box area
    // boxes are drawn in [plotT+bkZone, plotB]
    const int plotT   = r.top()  + padT + titleH;
    const int plotB   = r.bottom() - padB - xlabH;
    const int boxTop  = plotT + bkZone;      // boxes start here
    const int plotH   = plotB - boxTop;

    if (plotW < 20 || plotH < 20) return;

    // ── Y-range ───────────────────────────────────────────────────────────────
    double rawMin = 1e18, rawMax = -1e18;
    for (const auto& bs : m_boxes) {
        if (!bs.valid) continue;
        rawMin = std::min(rawMin, bs.q0);
        rawMax = std::max(rawMax, bs.q4);
        for (double o : bs.outliers) { rawMin=std::min(rawMin,o); rawMax=std::max(rawMax,o); }
    }
    if (rawMin > rawMax) { rawMin=0; rawMax=1; }
    if (std::abs(rawMax-rawMin) < 1e-9) { rawMin-=1; rawMax+=1; }
    const double yPad  = (rawMax-rawMin)*0.10;
    const double yMin  = (rawMin >= 0.0) ? 0.0 : rawMin-yPad;
    const double yMax  = rawMax+yPad;
    const double yRange = yMax-yMin;

    auto Y = [&](double v) -> int {
        return plotB - int((v-yMin)/yRange * plotH);
    };

    // ── Grid + y-numbers ─────────────────────────────────────────────────────
    {
        QFont nf; nf.setPixelSize(px(11));  p.setFont(nf);
        const int NG = 5;
        for (int gi=0; gi<=NG; ++gi) {
            double v = yMin + yRange*gi/NG;
            int ys   = Y(v);
            p.setPen(QPen(QColor(50,55,82), 1, Qt::DashLine));
            p.drawLine(plotL, ys, plotR, ys);
            p.setPen(QColor(148,154,178));
            QString s = (std::abs(v)<1e4) ? QString::number(v,'f',1) : QString::number(v,'g',3);
            p.drawText(QRect(r.left()+ylabW, ys-px(10), ynumW-px(4), px(20)),
                       Qt::AlignRight|Qt::AlignVCenter, s);
        }
    }
    p.setPen(QPen(QColor(70,75,100), 1));
    p.drawLine(plotL, boxTop, plotL, plotB);

    // ── Y-axis label (rotated) ────────────────────────────────────────────────
    if (!m_yLabel.isEmpty()) {
        p.save();
        QFont lf; lf.setPixelSize(px(13)); lf.setBold(true); p.setFont(lf);
        p.setPen(QColor(210,215,230));
        p.translate(r.left()+px(13), (boxTop+plotB)/2);
        p.rotate(-90);
        p.drawText(QRect(-px(90),-px(12),px(180),px(24)), Qt::AlignCenter, m_yLabel);
        p.restore();
    }

    // ── Title ─────────────────────────────────────────────────────────────────
    if (!m_title.isEmpty()) {
        QFont tf; tf.setPixelSize(px(16)); tf.setBold(true); p.setFont(tf);
        p.setPen(QColor(228,232,248));
        p.drawText(QRect(plotL, r.top()+padT, plotW, titleH),
                   Qt::AlignCenter|Qt::AlignVCenter, m_title);
    }

    // ── Rank order ────────────────────────────────────────────────────────────
    QVector<int> rankIdx(n); std::iota(rankIdx.begin(),rankIdx.end(),0);
    std::sort(rankIdx.begin(), rankIdx.end(), [&](int a, int b){
        double ma = m_boxes[a].valid ? m_boxes[a].q2 : 1e18;
        double mb = m_boxes[b].valid ? m_boxes[b].q2 : 1e18;
        return m_lowerBetter ? ma < mb : ma > mb;
    });
    QVector<int> rankOf(n);
    for (int ri=0; ri<n; ++ri) rankOf[rankIdx[ri]] = ri;

    // ── Box positions ─────────────────────────────────────────────────────────
    const double stepW   = static_cast<double>(plotW) / n;
    const int    divN    = std::max(1, n*2+1);
    const double rawBW   = static_cast<double>(plotW) / divN;
    const double boxW    = std::max(double(px(30)), std::min(double(px(130)), rawBW));
    QVector<double> cx(n);
    for (int i=0; i<n; ++i) cx[i] = plotL + stepW*i + stepW*0.5;

    // ── Draw boxes ────────────────────────────────────────────────────────────
    for (int i=0; i<n; ++i) {
        const auto& bs  = m_boxes[i];
        const QColor col  = rankColor(rankOf[i], n);
        const QColor colD = col.darker(145);

        // X-label
        {
            QFont xf; xf.setPixelSize(px(12)); p.setFont(xf);
            p.setPen(QColor(200,205,220));
            QString lbl = m_names[i];
            if (lbl.length() > 8 && !lbl.contains('\n'))
                lbl = lbl.left(8) + "\n" + lbl.mid(8);
            const int lw = std::min(int(boxW*2.2+px(10)), int(stepW)-px(4));
            p.drawText(QRect(int(cx[i])-lw/2, plotB+px(6), lw, px(38)),
                       Qt::AlignHCenter|Qt::AlignTop, lbl);
        }

        if (!bs.valid) continue;
        int yQ1=Y(bs.q1), yQ3=Y(bs.q3), yQ2=Y(bs.q2), yQ0=Y(bs.q0), yQ4=Y(bs.q4);
        int bx=int(cx[i]-boxW/2), bw=int(boxW);

        QLinearGradient grad(bx,yQ3,bx+bw,yQ1);
        grad.setColorAt(0,colD); grad.setColorAt(1,col.lighter(125));
        p.fillRect(bx,yQ3,bw,yQ1-yQ3,grad);
        p.setPen(QPen(col.lighter(155),1.5));  p.drawRect(bx,yQ3,bw,yQ1-yQ3);
        p.setPen(QPen(Qt::white,2.5));          p.drawLine(bx,yQ2,bx+bw,yQ2);

        const int icx=int(cx[i]), cap=int(boxW*0.30);
        p.setPen(QPen(col,1.5));
        p.drawLine(icx,yQ3,icx,yQ4); p.drawLine(icx,yQ1,icx,yQ0);
        p.drawLine(icx-cap,yQ4,icx+cap,yQ4); p.drawLine(icx-cap,yQ0,icx+cap,yQ0);

        p.setBrush(Qt::NoBrush);
        for (double ov : bs.outliers) p.drawEllipse(icx-px(3),Y(ov)-px(3),px(6),px(6));

        if (n>1) {
            QFont bf; bf.setPixelSize(px(10)); bf.setBold(true); p.setFont(bf);
            p.setPen(col.lighter(185));
            p.drawText(bx+bw+px(3), yQ3+px(13), QString("#%1").arg(rankOf[i]+1));
        }
    }

    // ── Significance brackets (ABOVE the boxes, in bkZone) ───────────────────
    //
    //  Each row:
    //    [text "OPT_A vs OPT_B:  p=0.031  *"   → rows top 20px]
    //    [horizontal bar + ticks                → at row bottom (8px from bottom)]
    //
    //  Rows sorted: most significant first (= drawn closest to top, furthest from boxes).
    //
    if (nBk > 0) {
        QVector<Bracket> sorted = m_brackets;
        if (sorted.size() > 6) sorted.resize(6);
        std::stable_sort(sorted.begin(), sorted.end(), [](const Bracket& a, const Bracket& b){
            if (a.pValue < 0) return false;
            if (b.pValue < 0) return true;
            return a.pValue < b.pValue;   // most significant = smallest p → first
        });

        QFont bkf; bkf.setPixelSize(px(11)); p.setFont(bkf);

        for (int bi=0; bi<sorted.size(); ++bi) {
            const auto& br = sorted[bi];
            if (br.i >= n || br.j >= n) continue;
            const int lo = std::min(br.i, br.j);
            const int hi = std::max(br.i, br.j);

            const bool sig = (br.stars=="*"||br.stars=="**"||br.stars=="***");
            const QColor barCol  = sig ? QColor("#4EDDAD") : QColor(110,115,155);
            const QColor textCol = sig ? QColor("#6CF0C0") : QColor(155,160,195);

            // Row: bi=0 is the TOP row (closest to title, most significant)
            const int rowTop = plotT + bi * bkRowH;
            const int barY   = rowTop + bkRowH - px(6);   // bar near bottom of row
            const int textTop= rowTop + px(2);
            const int textH  = bkRowH - px(10);           // text area

            const double x1 = cx[lo], x2 = cx[hi];

            // Horizontal bar + vertical ticks
            p.setPen(QPen(barCol, px(2)));
            p.drawLine(int(x1), barY, int(x2), barY);
            p.drawLine(int(x1), barY-px(4), int(x1), barY+px(4));
            p.drawLine(int(x2), barY-px(4), int(x2), barY+px(4));

            // Label: "OPT_A vs OPT_B:   p=0.031   *"  ABOVE the bar
            const QString optA   = (lo < m_names.size()) ? m_names[lo] : "?";
            const QString optB   = (hi < m_names.size()) ? m_names[hi] : "?";
            const QString pStr   = (br.pValue >= 0)
                ? QString("p=%1").arg(br.pValue, 0, 'f', 3) : "p=n/a";
            const QString stars  = br.stars.isEmpty() ? "ns" : br.stars;
            const QString label  = QString("%1 vs %2:    %3    %4").arg(optA,optB,pStr,stars);

            p.setPen(textCol);
            const int spanL = std::max(int(x1) - px(20), plotL);
            const int spanR = std::min(int(x2) + px(20), plotR);
            p.drawText(QRect(spanL, textTop, spanR-spanL, textH),
                       Qt::AlignCenter|Qt::AlignVCenter, label);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  exportPng — renders a high-DPI PNG (300 DPI output).
//
//  KEY FIX: We do NOT set setDotsPerMeterX BEFORE rendering.
//  If we set it before, Qt renders fonts at 300 DPI AND then p.scale() enlarges
//  them again → double-scaling (8pt becomes ~25pt). Setting DPI AFTER rendering
//  only adds metadata for printers without affecting font sizes.
// ─────────────────────────────────────────────────────────────────────────────
bool BatchPlotWidget::exportPng(const QString& path) const {
    const int    LW    = 1600;       // logical canvas (matches screen proportions)
    const int    LH    = 900;
    const double SCALE = 300.0 / 96.0;   // ≈ 3.125 — physical pixels per logical pixel

    const int imgW = qRound(LW * SCALE);
    const int imgH = qRound(LH * SCALE);

    // Create image WITHOUT DPI setting (defaults to 96 DPI device)
    // This prevents Qt from double-scaling fonts.
    QImage img(imgW, imgH, QImage::Format_ARGB32);
    img.fill(QColor(28, 30, 44));

    {
        QPainter p(&img);
        p.setRenderHints(QPainter::Antialiasing | QPainter::TextAntialiasing
                         | QPainter::SmoothPixmapTransform);
        // Pass sc=SCALE so all pixel constants and fonts are pre-scaled.
        // This renders everything at physical pixel sizes directly — no transform needed.
        renderTo(p, QRect(0, 0, imgW, imgH), SCALE);
    } // painter must be ended before we modify the image

    // NOW set the 300 DPI metadata (for printers — does not affect pixels)
    img.setDotsPerMeterX(qRound(300.0 / 0.0254));
    img.setDotsPerMeterY(qRound(300.0 / 0.0254));

    return img.save(path, "PNG");
}

} // namespace NeuralStudio
