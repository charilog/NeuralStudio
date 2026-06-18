#include "XlsxWriter.h"

#include <QFile>
#include <QByteArray>
#include <QtEndian>
#include <stdexcept>
#include <cstring>

namespace NeuralStudio {

// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                       Internal ZIP writer                                  ║
// ║  Produces a "stored" (uncompressed) ZIP archive that Excel accepts.        ║
// ║  Implements only the subset of PKZip needed for XLSX (no encryption, no    ║
// ║  ZIP64, no data descriptors).                                              ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
namespace {

// CRC-32 (IEEE 802.3 polynomial 0xEDB88320)
class Crc32 {
public:
    static quint32 compute(const QByteArray& data) {
        ensureTable();
        quint32 c = 0xFFFFFFFFu;
        for (char b : data)
            c = s_table[(c ^ static_cast<quint8>(b)) & 0xFFu] ^ (c >> 8);
        return c ^ 0xFFFFFFFFu;
    }
private:
    static quint32 s_table[256];
    static bool    s_init;
    static void ensureTable() {
        if (s_init) return;
        for (quint32 i = 0; i < 256; ++i) {
            quint32 c = i;
            for (int j = 0; j < 8; ++j)
                c = (c >> 1) ^ ((c & 1u) ? 0xEDB88320u : 0u);
            s_table[i] = c;
        }
        s_init = true;
    }
};
quint32 Crc32::s_table[256] = {};
bool    Crc32::s_init       = false;

class ZipBuilder {
public:
    void addFile(const QString& name, const QByteArray& data) {
        Entry e;
        e.name              = name.toUtf8();
        e.data              = data;
        e.crc               = Crc32::compute(data);
        e.localHeaderOffset = static_cast<quint32>(m_buf.size());
        writeLocalHeader(e);
        m_buf.append(data);
        m_entries.push_back(std::move(e));
    }

    QByteArray finalize() {
        const quint32 cdirStart = static_cast<quint32>(m_buf.size());
        for (const auto& e : m_entries)
            writeCentralHeader(e);
        const quint32 cdirSize = static_cast<quint32>(m_buf.size()) - cdirStart;
        writeEOCD(static_cast<quint16>(m_entries.size()), cdirSize, cdirStart);
        return m_buf;
    }

private:
    struct Entry {
        QByteArray name;
        QByteArray data;
        quint32    crc               = 0;
        quint32    localHeaderOffset = 0;
    };

    QByteArray         m_buf;
    std::vector<Entry> m_entries;

    void le16(quint16 v) {
        char b[2];
        qToLittleEndian<quint16>(v, b);
        m_buf.append(b, 2);
    }
    void le32(quint32 v) {
        char b[4];
        qToLittleEndian<quint32>(v, b);
        m_buf.append(b, 4);
    }

    void writeLocalHeader(const Entry& e) {
        const quint32 sz = static_cast<quint32>(e.data.size());
        le32(0x04034b50);              // signature
        le16(20);                      // version needed
        le16(0);                       // flags
        le16(0);                       // method = stored
        le16(0);                       // mod time
        le16(0);                       // mod date
        le32(e.crc);                   // CRC-32
        le32(sz);                      // compressed size
        le32(sz);                      // uncompressed size
        le16(static_cast<quint16>(e.name.size())); // filename length
        le16(0);                       // extra length
        m_buf.append(e.name);
    }

    void writeCentralHeader(const Entry& e) {
        const quint32 sz = static_cast<quint32>(e.data.size());
        le32(0x02014b50);              // signature
        le16(20);                      // version made by
        le16(20);                      // version needed
        le16(0);                       // flags
        le16(0);                       // method
        le16(0);                       // mod time
        le16(0);                       // mod date
        le32(e.crc);
        le32(sz);
        le32(sz);
        le16(static_cast<quint16>(e.name.size()));
        le16(0);                       // extra length
        le16(0);                       // comment length
        le16(0);                       // disk number
        le16(0);                       // internal attrs
        le32(0);                       // external attrs
        le32(e.localHeaderOffset);
        m_buf.append(e.name);
    }

    void writeEOCD(quint16 n, quint32 cdirSize, quint32 cdirOffset) {
        le32(0x06054b50);              // signature
        le16(0);                       // this disk
        le16(0);                       // disk where CD starts
        le16(n);                       // entries on this disk
        le16(n);                       // total entries
        le32(cdirSize);
        le32(cdirOffset);
        le16(0);                       // comment length
    }
};

} // anonymous namespace


// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                              XlsxSheet                                     ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
XlsxSheet::XlsxSheet(const QString& name) : m_name(name) {}

void XlsxSheet::setHeader(const QStringList& headers) {
    m_headers = headers;
}

void XlsxSheet::addRow(const std::vector<XlsxCell>& cells) {
    m_rows.push_back(cells);
}

// Convert 1-indexed column number → Excel letters (1→A, 27→AA, ...)
QString XlsxSheet::colLetter(int col1based) {
    QString s;
    int c = col1based;
    while (c > 0) {
        --c;
        s.prepend(QChar('A' + (c % 26)));
        c /= 26;
    }
    return s;
}

QString XlsxSheet::xmlEscape(const QString& s) {
    QString out; out.reserve(s.size());
    for (QChar ch : s) {
        switch (ch.unicode()) {
            case '&':  out += "&amp;";  break;
            case '<':  out += "&lt;";   break;
            case '>':  out += "&gt;";   break;
            case '"':  out += "&quot;"; break;
            case '\'': out += "&apos;"; break;
            default:
                if (ch.unicode() < 0x20 && ch != '\t' && ch != '\n' && ch != '\r')
                    out += ' ';  // strip control chars
                else
                    out += ch;
        }
    }
    return out;
}

QByteArray XlsxSheet::render() const {
    QString xml;
    xml.reserve(8192);
    xml += "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
           "<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">"
           "<sheetData>";

    int rowNum = 1;

    // Header row (style index 1 = bold)
    if (!m_headers.isEmpty()) {
        xml += QString("<row r=\"%1\">").arg(rowNum);
        for (int c = 0; c < m_headers.size(); ++c) {
            const QString ref = colLetter(c + 1) + QString::number(rowNum);
            xml += QString("<c r=\"%1\" s=\"1\" t=\"inlineStr\"><is><t>%2</t></is></c>")
                       .arg(ref, xmlEscape(m_headers[c]));
        }
        xml += "</row>";
        ++rowNum;
    }

    // Data rows
    for (const auto& row : m_rows) {
        xml += QString("<row r=\"%1\">").arg(rowNum);
        for (int c = 0; c < static_cast<int>(row.size()); ++c) {
            const auto& cell = row[c];
            if (cell.type == XlsxCell::Empty) continue;
            const QString ref = colLetter(c + 1) + QString::number(rowNum);
            if (cell.type == XlsxCell::Number) {
                xml += QString("<c r=\"%1\"><v>%2</v></c>")
                           .arg(ref, QString::number(cell.num, 'g', 12));
            } else {
                xml += QString("<c r=\"%1\" t=\"inlineStr\"><is><t>%2</t></is></c>")
                           .arg(ref, xmlEscape(cell.str));
            }
        }
        xml += "</row>";
        ++rowNum;
    }

    xml += "</sheetData></worksheet>";
    return xml.toUtf8();
}


// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                              XlsxWriter                                    ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
XlsxWriter::XlsxWriter() = default;

XlsxSheet* XlsxWriter::addSheet(const QString& name) {
    m_sheets.push_back(std::make_unique<XlsxSheet>(name));
    return m_sheets.back().get();
}

// ── Static XML templates for the package skeleton ─────────────────────────────
static QByteArray kContentTypes(int sheetCount) {
    QString s =
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
        "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>"
        "<Default Extension=\"xml\" ContentType=\"application/xml\"/>"
        "<Override PartName=\"/xl/workbook.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml\"/>"
        "<Override PartName=\"/xl/styles.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml\"/>";
    for (int i = 1; i <= sheetCount; ++i)
        s += QString("<Override PartName=\"/xl/worksheets/sheet%1.xml\" "
                     "ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml\"/>")
                 .arg(i);
    s += "</Types>";
    return s.toUtf8();
}

static QByteArray kRootRels() {
    return QByteArray(
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
        "<Relationship Id=\"rId1\" "
        "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" "
        "Target=\"xl/workbook.xml\"/>"
        "</Relationships>");
}

static QByteArray kWorkbook(const std::vector<std::unique_ptr<XlsxSheet>>& sheets) {
    QString s =
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<workbook xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" "
        "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">"
        "<sheets>";
    for (int i = 0; i < static_cast<int>(sheets.size()); ++i) {
        // Sheet names are simple ASCII in our use case; Excel allows letters/digits/spaces/hyphens/underscores.
        s += QString("<sheet name=\"%1\" sheetId=\"%2\" r:id=\"rId%2\"/>")
                 .arg(sheets[i]->name())
                 .arg(i + 1);
    }
    s += "</sheets></workbook>";
    return s.toUtf8();
}

static QByteArray kWorkbookRels(int sheetCount) {
    QString s =
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
        "<Relationship Id=\"rIdStyles\" "
        "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles\" "
        "Target=\"styles.xml\"/>";
    for (int i = 1; i <= sheetCount; ++i)
        s += QString("<Relationship Id=\"rId%1\" "
                     "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet\" "
                     "Target=\"worksheets/sheet%1.xml\"/>").arg(i);
    s += "</Relationships>";
    return s.toUtf8();
}

// Two cell formats: index 0 = default, index 1 = bold (used for headers)
static QByteArray kStyles() {
    return QByteArray(
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<styleSheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">"
        "<fonts count=\"2\">"
        "<font><sz val=\"11\"/><name val=\"Calibri\"/></font>"
        "<font><b/><sz val=\"11\"/><name val=\"Calibri\"/></font>"
        "</fonts>"
        "<fills count=\"1\"><fill><patternFill patternType=\"none\"/></fill></fills>"
        "<borders count=\"1\"><border/></borders>"
        "<cellStyleXfs count=\"1\"><xf numFmtId=\"0\" fontId=\"0\" fillId=\"0\" borderId=\"0\"/></cellStyleXfs>"
        "<cellXfs count=\"2\">"
        "<xf numFmtId=\"0\" fontId=\"0\" fillId=\"0\" borderId=\"0\" xfId=\"0\"/>"
        "<xf numFmtId=\"0\" fontId=\"1\" fillId=\"0\" borderId=\"0\" xfId=\"0\" applyFont=\"1\"/>"
        "</cellXfs>"
        "</styleSheet>");
}

void XlsxWriter::save(const QString& path) {
    if (m_sheets.empty())
        throw std::runtime_error("XlsxWriter: cannot save workbook with no sheets.");

    ZipBuilder zip;

    zip.addFile("[Content_Types].xml", kContentTypes(static_cast<int>(m_sheets.size())));
    zip.addFile("_rels/.rels",          kRootRels());
    zip.addFile("xl/workbook.xml",      kWorkbook(m_sheets));
    zip.addFile("xl/_rels/workbook.xml.rels", kWorkbookRels(static_cast<int>(m_sheets.size())));
    zip.addFile("xl/styles.xml",        kStyles());

    for (int i = 0; i < static_cast<int>(m_sheets.size()); ++i) {
        const QString sheetPath = QString("xl/worksheets/sheet%1.xml").arg(i + 1);
        zip.addFile(sheetPath, m_sheets[i]->render());
    }

    QByteArray archive = zip.finalize();

    QFile f(path);
    if (!f.open(QIODevice::WriteOnly))
        throw std::runtime_error("Cannot open output file: " + path.toStdString());
    if (f.write(archive) != archive.size())
        throw std::runtime_error("Write failed: " + path.toStdString());
    f.close();
}

} // namespace NeuralStudio
