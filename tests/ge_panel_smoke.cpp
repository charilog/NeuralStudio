// Offscreen smoke test of the Mode 6 panel: loads a dataset, runs, saves screenshots.
//   QT_QPA_PLATFORM=offscreen ./ge_panel_smoke dataset out_prefix
#include "ui/panels/GEStudioPanel.h"
#include <QApplication>
#include <QLabel>
#include <QPushButton>
#include <QSpinBox>
#include <QTabWidget>
#include <QTimer>
#include <QPixmap>
#include <QSettings>
#include <QTableWidget>
#include <cstdio>
using namespace NeuralStudio;

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    if (argc < 3) return 2;
    // GE_SMOKE_MODELS=ge1,knn,linear selects the models of the run (Models list of the Single run page)
    if (qEnvironmentVariableIsSet("GE_SMOKE_MODELS"))
        QSettings("NeuralStudio", "NeuralStudio").setValue("GEStudio/singleModels",
                                                           qEnvironmentVariable("GE_SMOKE_MODELS").split(','));
    GEStudioPanel p;
    p.resize(1280, 1500);
    p.setQueuePaths({argv[1]});
    p.show();
    const QString prefix = argv[2];
    for (auto* sp : p.findChildren<QSpinBox*>())     // keep the smoke run short: generations = 10, pop = 24
        if (sp->maximum() == 5000 && sp->minimum() == 1) sp->setValue(10);
    QTabWidget *setup = nullptr, *tabs = nullptr;
    for (auto* t : p.findChildren<QTabWidget*>()) {
        if (t->tabText(0).startsWith("Single")) setup = t;
        if (t->tabText(0).startsWith("Convergence")) tabs = t;
    }
    setup->setCurrentIndex(0);                      // single run page (the panel remembers the last page)
    QPushButton* run = nullptr;
    for (auto* b : p.findChildren<QPushButton*>()) if (b->text().startsWith("▶")) run = b;
    run->click();
    p.grab().save(prefix + "_running.png");
    auto* timer = new QTimer;
    int ticks = 0;
    QObject::connect(timer, &QTimer::timeout, [&] {
        ++ticks;
        QString status;
        for (auto* l : p.findChildren<QLabel*>()) if (l->text().startsWith("Finished") || l->text().startsWith("Error")) status = l->text();
        if (ticks == 10) p.grab().save(prefix + "_mid.png");
        if (!status.isEmpty() || ticks > 1600) {
            std::printf("status: %s (ticks=%d)\n", qPrintable(status), ticks);
            for (auto* t : p.findChildren<QTableWidget*>())
                if (t->columnCount() == 8 && t->horizontalHeaderItem(1) && t->horizontalHeaderItem(1)->text() == "Method")
                    for (int r = 0; r < t->rowCount(); ++r)
                        std::printf("row: %s | test %s\n", qPrintable(t->item(r, 1)->text()), qPrintable(t->item(r, 3)->text()));
            for (int i = 0; i < tabs->count(); ++i) {
                tabs->setCurrentIndex(i);
                app.processEvents();
                p.grab().save(QString("%1_tab%2.png").arg(prefix).arg(i));
            }
            app.quit();
        }
    });
    if (argc > 3) QTimer::singleShot(atoi(argv[3]), [&] { run->setEnabled(true); QPushButton* st = nullptr;
        for (auto* b : p.findChildren<QPushButton*>()) if (b->text().contains("Stop")) st = b;
        std::printf("clicking stop\n"); st->click(); });
    timer->start(500);  // 400 ticks = 200 s
    return app.exec();
}
