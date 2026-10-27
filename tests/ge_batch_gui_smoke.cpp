// Offscreen smoke test of the batch section of the Mode 6 panel.
//   QT_QPA_PLATFORM=offscreen ./ge_batch_gui_smoke <working folder> "<command>" <out.png> [stop-after-ms]
// Opens the Experiment page, pastes the command, presses the single Run button, follows the progress bar and the status line
// until the batch finishes (or is stopped), then saves a screenshot of the Batch tab.
#include "ui/panels/GEStudioPanel.h"
#include <QApplication>
#include <QLabel>
#include <QLineEdit>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QTabWidget>
#include <QTimer>
#include <QToolButton>
#include <cstdio>
using namespace NeuralStudio;

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    if (argc < 4) { std::fprintf(stderr, "usage: ge_batch_gui_smoke <workdir> <command> <out.png> [stop-ms]\n"); return 2; }
    GEStudioPanel p;
    p.resize(1280, 1500);
    p.show();

    for (auto* t : p.findChildren<QTabWidget*>()) if (t->tabText(0).startsWith("Single")) t->setCurrentIndex(1);
    for (auto* t : p.findChildren<QToolButton*>()) if (t->isCheckable() && !t->isChecked()) t->setChecked(true);
    QPlainTextEdit* cmd = nullptr;
    for (auto* e : p.findChildren<QPlainTextEdit*>()) if (!e->isReadOnly() && e->placeholderText().contains("ge_batch")) cmd = e;
    QLineEdit* wd = nullptr;
    for (auto* e : p.findChildren<QLineEdit*>()) if (e->toolTip().startsWith("Relative paths")) wd = e;
    QPushButton *run = nullptr, *stop = nullptr;
    for (auto* b : p.findChildren<QPushButton*>()) {
        if (b->text().startsWith("▶")) run = b;
        if (b->text().contains("Stop")) stop = b;
    }
    if (!cmd || !wd || !run || !stop) { std::fprintf(stderr, "widgets not found\n"); return 3; }
    cmd->setPlainText(QString::fromLocal8Bit(argv[2]));
    wd->setText(QString::fromLocal8Bit(argv[1]));
    run->click();

    auto* bar = p.findChild<QProgressBar*>();
    int lastBar = -1, decreases = 0, ticks = 0;
    QString lastStatus;
    auto* timer = new QTimer;
    QObject::connect(timer, &QTimer::timeout, [&] {
        ++ticks;
        if (bar->value() < lastBar) ++decreases;
        if (bar->value() != lastBar) std::printf("bar %d\n", bar->value());
        lastBar = bar->value();
        QString status;
        for (auto* l : p.findChildren<QLabel*>())
            if ((l->text().startsWith("Experiment") || l->text().startsWith("Stopped") || l->text().startsWith("Tables"))) status = l->text();
        if (status != lastStatus) { std::printf("status: %s\n", qPrintable(status)); lastStatus = status; }
        const bool done = !stop->isEnabled() && run->isEnabled() && ticks > 2;
        if (ticks == 6) p.grab().save(QString(argv[3]).replace(".png", "_mid.png"));
        if (done || ticks > 2400) {
            std::printf("final status: %s\nprogress decreases: %d\n", qPrintable(status), decreases);
            p.grab().save(argv[3]);
            app.quit();
        }
    });
    // stop-ms < 0: close the application while the batch runs (tests the panel destructor)
    if (argc > 4 && atoi(argv[4]) < 0)
        QTimer::singleShot(-atoi(argv[4]), [&] { std::printf("quitting while running\n"); app.quit(); });
    else if (argc > 4) QTimer::singleShot(atoi(argv[4]), [&] { std::printf("clicking stop\n"); stop->click(); });
    timer->start(250);
    return app.exec();
}
