// Offscreen test of the Experiment page of Mode 6: Open an earlier results file (<prefix>_raw.csv),
// Edit… (add a model and a dataset), untick a model on the Models list of the page, Run (only the
// missing runs), screenshots.
//   QT_QPA_PLATFORM=offscreen ./ge_exp_open_smoke <working folder> <prefix>_raw.csv <add-model> <remove-model> <add-dataset> <out.png>
#include "ui/panels/GEStudioPanel.h"
#include "ui/panels/GEExperimentDialog.h"
#include <QApplication>
#include <QDialogButtonBox>
#include <QFileDialog>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QTabWidget>
#include <QTableWidget>
#include <QTextBrowser>
#include <QTimer>
#include <QTreeWidget>
#include <cstdio>
using namespace NeuralStudio;

// Tick / clear a model on a Models list the way a click does.
static bool tickModel(QWidget* root, const QString& id, bool on) {
    for (auto* t : root->findChildren<QTreeWidget*>())
        for (QTreeWidgetItemIterator it(t); *it; ++it)
            if ((*it)->data(0, Qt::UserRole).toString() == id) {
                (*it)->setCheckState(0, on ? Qt::Checked : Qt::Unchecked);
                return true;
            }
    return false;
}

static void closeMessageBoxes() {
    for (QWidget* w : QApplication::topLevelWidgets())
        if (auto* mb = qobject_cast<QMessageBox*>(w); mb && mb->isVisible()) {
            std::printf("message: %s\n", qPrintable(mb->text().left(200)));
            mb->accept();
        }
}

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    setvbuf(stdout, nullptr, _IONBF, 0);
    QApplication::setAttribute(Qt::AA_DontUseNativeDialogs);
    if (argc < 7) return 2;
    const QString wd = argv[1], raw = argv[2], addModel = argv[3], rmModel = argv[4], addDs = argv[5], out = argv[6];
    GEStudioPanel p;
    p.resize(1280, 1500);
    p.show();
    for (auto* t : p.findChildren<QTabWidget*>()) if (t->tabText(0).startsWith("Single")) t->setCurrentIndex(1);
    for (auto* e : p.findChildren<QLineEdit*>()) if (e->toolTip().startsWith("Relative paths")) { e->setText(wd); emit e->editingFinished(); }
    auto btn = [&](const QString& prefix) -> QPushButton* {
        for (auto* b : p.findChildren<QPushButton*>()) if (b->text().startsWith(prefix)) return b;
        return nullptr;
    };
    auto* summary = p.findChild<QTextBrowser*>();
    QTimer guard; QObject::connect(&guard, &QTimer::timeout, closeMessageBoxes); guard.start(300);

    // 1. Open… the results file through the (non-native) file dialog
    QTimer fdTimer;
    QObject::connect(&fdTimer, &QTimer::timeout, [&] {
        if (getenv("DBG")) for (QWidget* w : QApplication::topLevelWidgets()) if (w->isVisible()) std::printf("window %s '%s'\n", w->metaObject()->className(), qPrintable(w->windowTitle()));
        for (QWidget* w : QApplication::topLevelWidgets())
            if (auto* fd = qobject_cast<QFileDialog*>(w); fd && fd->isVisible()) {
                fdTimer.stop();
                if (auto* le = fd->findChild<QLineEdit*>("fileNameEdit")) le->setText(raw); else fd->selectFile(raw);
                std::printf("file dialog: selected %s\n", qPrintable(fd->selectedFiles().join(";")));
                QMetaObject::invokeMethod(fd, "accept");
            }
    });
    fdTimer.start(200);
    btn("Open…")->click();
    std::printf("--- after Open\n%s\n", qPrintable(summary->toPlainText()));
    p.grab().save(QString(out).replace(".png", "_opened.png"));

    // GE_SMOKE_OPEN_ONLY=1: screenshots of the opened results (every result tab), no edit / run
    if (qEnvironmentVariableIsSet("GE_SMOKE_OPEN_ONLY")) {
        for (auto* t : p.findChildren<QTabWidget*>()) {
            if (t->tabText(0) != "Convergence") continue;
            for (int i = 0; i < t->count(); ++i) if (t->isTabVisible(i)) {
                t->setCurrentIndex(i); app.processEvents();
                p.grab().save(QString(out).replace(".png", "_" + t->tabText(i) + ".png"));
            }
        }
        return 0;
    }
    // 2. Edit…: + model, - model, + dataset
    QTimer dlgTimer;
    QObject::connect(&dlgTimer, &QTimer::timeout, [&] {
        for (QWidget* w : QApplication::topLevelWidgets()) {
            auto* d = qobject_cast<GEExperimentDialog*>(w);
            if (!d || !d->isVisible()) continue;
            for (auto* t : d->findChildren<QTableWidget*>()) {
                const QString h0 = t->horizontalHeaderItem(0) ? t->horizontalHeaderItem(0)->text() : QString();
                for (int r = 0; r < t->rowCount(); ++r) {
                    auto* c = t->item(r, 0);
                    if (h0 == "Use" && t->item(r, 1)->text() == addDs) c->setCheckState(Qt::Checked);
                }
            }
            std::printf("dialog: tick %s -> %d\n", qPrintable(addModel), tickModel(d, addModel, true));
            d->grab().save(QString(out).replace(".png", "_dialog.png"));
            if (auto* tabs = d->findChild<QTabWidget*>()) for (int i = 0; i < tabs->count(); ++i) {
                tabs->setCurrentIndex(i); app.processEvents();
                d->grab().save(QString(out).replace(".png", QString("_dialog%1.png").arg(i)));
            }
            dlgTimer.stop();
            QMetaObject::invokeMethod(d, "accept");
        }
    });
    dlgTimer.start(300);
    btn("Edit…")->click();
    std::printf("--- after Edit\n%s\n", qPrintable(summary->toPlainText()));
    // remove a model straight from the Models list of the Experiment page
    QWidget* expPage = nullptr;
    for (auto* t : p.findChildren<QTabWidget*>()) if (t->tabText(0).startsWith("Single")) expPage = t->widget(1);
    std::printf("page: untick %s -> %d\n", qPrintable(rmModel), tickModel(expPage, rmModel, false));
    for (int i = 0; i < 5; ++i) app.processEvents();
    std::printf("--- after untick on the page\n%s\n", qPrintable(summary->toPlainText()));

    // 3. Run: only the missing runs
    QPushButton* run = btn("▶");
    QPushButton* stop = btn("■");
    run->click();
    int ticks = 0;
    QTimer poll;
    QObject::connect(&poll, &QTimer::timeout, [&] {
        ++ticks;
        if (ticks > 3 && run->isEnabled() && !stop->isEnabled()) {
            QString status;
            for (auto* l : p.findChildren<QLabel*>()) if (l->text().startsWith("Tables") || l->text().startsWith("Experiment")) status = l->text();
            std::printf("--- after Run\nstatus: %s\n%s\n", qPrintable(status), qPrintable(summary->toPlainText()));
            p.grab().save(out);
            // result tabs of the Experiment page, then of the Single run page
            for (auto* t : p.findChildren<QTabWidget*>()) {
                if (t->tabText(0) != "Convergence") continue;
                QStringList vis;
                for (int i = 0; i < t->count(); ++i) if (t->isTabVisible(i)) vis << t->tabText(i);
                std::printf("visible result tabs (experiment): %s\n", qPrintable(vis.join(", ")));
                for (int i = 0; i < t->count(); ++i) if (t->tabText(i) == "Hypotheses") t->setCurrentIndex(i);
                app.processEvents();
                p.grab().save(QString(out).replace(".png", "_hyp.png"));
                for (auto* s2 : p.findChildren<QTabWidget*>()) if (s2->tabText(0).startsWith("Single")) s2->setCurrentIndex(0);
                app.processEvents();
                vis.clear();
                for (int i = 0; i < t->count(); ++i) if (t->isTabVisible(i)) vis << t->tabText(i);
                std::printf("visible result tabs (single run): %s\n", qPrintable(vis.join(", ")));
                for (auto* s2 : p.findChildren<QTabWidget*>()) if (s2->tabText(0).startsWith("Single")) s2->setCurrentIndex(1);
            }
            app.quit();
        }
        if (ticks > 1600) { std::printf("TIMEOUT\n"); p.grab().save(out); app.quit(); }
    });
    poll.start(250);
    return app.exec();
}
