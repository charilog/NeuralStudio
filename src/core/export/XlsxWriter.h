#pragma once
#include <QString>
#include <QStringList>
#include <vector>
#include <memory>

namespace NeuralStudio {

// ─── XlsxCell ────────────────────────────────────────────────────────────────
//  A single spreadsheet cell — either a number, a string, or empty.
//  Implicit constructors make calling code natural:
//      sheet->addRow({ "Dataset", 89, 13, 3.14, "" });
// ─────────────────────────────────────────────────────────────────────────────
struct XlsxCell {
    enum Type { Empty, Number, String };
    Type    type = Empty;
    double  num  = 0.0;
    QString str;

    XlsxCell() = default;
    XlsxCell(double v)         : type(Number), num(v) {}
    XlsxCell(int v)            : type(Number), num(v) {}
    XlsxCell(qint64 v)         : type(Number), num(static_cast<double>(v)) {}
    XlsxCell(const QString& s) : type(s.isEmpty() ? Empty : String), str(s) {}
    XlsxCell(const char* s)    : XlsxCell(QString(s)) {}
};

// ─── XlsxSheet ────────────────────────────────────────────────────────────────
//  One worksheet inside an XLSX workbook.  Add rows in order.
//  The first row added with addHeader() is rendered bold.
// ─────────────────────────────────────────────────────────────────────────────
class XlsxSheet {
public:
    explicit XlsxSheet(const QString& name);
    void setHeader(const QStringList& headers);   // becomes row 1, bold
    void addRow   (const std::vector<XlsxCell>& cells);
    int  rowCount() const { return static_cast<int>(m_rows.size()); }
    const QString& name() const { return m_name; }

    // Internal: emit the worksheet's body XML
    QByteArray render() const;

    // Public static utilities
    static QString colLetter(int col1based);
    static QString xmlEscape(const QString& s);

private:
    QString                          m_name;
    QStringList                      m_headers;
    std::vector<std::vector<XlsxCell>> m_rows;
};

// ─── XlsxWriter ───────────────────────────────────────────────────────────────
//  Top-level workbook.  Add sheets, then call save() to write a .xlsx file.
//  Implementation generates the required XML parts and packs them into a
//  stored-only ZIP archive (no compression).  No external libraries needed.
//
//  Throws std::runtime_error on I/O failure.
// ─────────────────────────────────────────────────────────────────────────────
class XlsxWriter {
public:
    XlsxWriter();
    XlsxSheet* addSheet(const QString& name);
    void       save(const QString& path);

    static QString fileFilter() {
        return "Excel Workbook (*.xlsx);;All files (*)";
    }

private:
    std::vector<std::unique_ptr<XlsxSheet>> m_sheets;
};

} // namespace NeuralStudio
