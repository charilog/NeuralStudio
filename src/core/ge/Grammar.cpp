#include "Grammar.h"
#include <algorithm>
#include <sstream>
#include <limits>
#include <cctype>

namespace NeuralStudio::ge {

namespace {
constexpr int kInf = std::numeric_limits<int>::max() / 4;

std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

std::string collapseSpaces(const std::string& s) {
    std::string o;
    bool sp = false;
    for (char c : s) {
        if (std::isspace(static_cast<unsigned char>(c))) { sp = true; continue; }
        if (sp && !o.empty()) o.push_back(' ');
        sp = false;
        o.push_back(c);
    }
    if (sp && !o.empty()) o.push_back(' ');
    return o;
}

std::string stripComment(const std::string& line) {
    const std::string t = trim(line);
    if (!t.empty() && t[0] == '#') return "";
    const auto p = line.find("//");
    return p == std::string::npos ? line : line.substr(0, p);
}

// Is `line` a rule header "<name> ::= ..."?  On success fills name/body.
bool isHeader(const std::string& line, std::string& name, std::string& body) {
    const auto pos = line.find("::=");
    if (pos == std::string::npos) return false;
    const std::string lhs = trim(line.substr(0, pos));
    if (lhs.size() < 3 || lhs.front() != '<' || lhs.back() != '>') return false;
    name = lhs.substr(1, lhs.size() - 2);
    if (name.empty() || name.find_first_of(" \t<>") != std::string::npos) return false;
    body = line.substr(pos + 3);
    return true;
}
} // namespace

// ── parse ─────────────────────────────────────────────────────────────────────
Grammar Grammar::parse(const std::string& bnf) {
    Grammar g;
    struct RawRule { std::string name, body; int line; };
    std::vector<RawRule> raws;

    std::istringstream in(bnf);
    std::string line;
    int lineNo = 0;
    while (std::getline(in, line)) {
        ++lineNo;
        line = stripComment(line);
        if (trim(line).empty()) continue;
        std::string name, body;
        if (isHeader(line, name, body)) {
            raws.push_back({name, body, lineNo});
        } else {
            if (raws.empty())
                throw std::runtime_error("line " + std::to_string(lineNo) +
                    ": expected a rule header like  <name> ::= ...");
            raws.back().body += " " + line;
        }
    }
    if (raws.empty()) throw std::runtime_error("grammar is empty");

    for (const auto& r : raws) {
        if (g.m_index.count(r.name))
            throw std::runtime_error("line " + std::to_string(r.line) +
                ": rule <" + r.name + "> is defined twice");
        g.m_index[r.name] = static_cast<int>(g.m_rules.size());
        Rule rule; rule.name = r.name;
        g.m_rules.push_back(std::move(rule));
    }

    for (size_t ri = 0; ri < raws.size(); ++ri) {
        const std::string& body = raws[ri].body;
        // split on '|'
        std::vector<std::string> alts;
        {
            std::string cur;
            for (char c : body) {
                if (c == '|') { alts.push_back(cur); cur.clear(); }
                else cur.push_back(c);
            }
            alts.push_back(cur);
        }
        if (trim(body).empty())
            throw std::runtime_error("line " + std::to_string(raws[ri].line) +
                ": rule <" + raws[ri].name + "> has no alternatives");

        for (const std::string& rawAlt : alts) {
            Production p;
            const std::string alt = trim(rawAlt);
            size_t i = 0;
            std::string lit;
            auto flush = [&] {
                const std::string c = collapseSpaces(lit);
                if (!c.empty()) { Symbol s; s.terminal = true; s.text = c; p.syms.push_back(s); }
                lit.clear();
            };
            while (i < alt.size()) {
                if (alt[i] == '<') {
                    const auto e = alt.find('>', i);
                    if (e == std::string::npos)
                        throw std::runtime_error("line " + std::to_string(raws[ri].line) +
                            ": unterminated '<' in rule <" + raws[ri].name + ">");
                    const std::string nm = alt.substr(i + 1, e - i - 1);
                    const auto it = g.m_index.find(nm);
                    if (it == g.m_index.end())
                        throw std::runtime_error("line " + std::to_string(raws[ri].line) +
                            ": rule <" + raws[ri].name + "> uses undefined <" + nm + ">");
                    flush();
                    Symbol s; s.terminal = false; s.nt = it->second; s.text = nm;
                    p.syms.push_back(s);
                    i = e + 1;
                } else {
                    lit.push_back(alt[i]);
                    ++i;
                }
            }
            flush();
            g.m_rules[ri].prods.push_back(std::move(p));
        }
    }
    g.computeMinDepths();
    return g;
}

void Grammar::computeMinDepths() {
    for (auto& r : m_rules) { r.minDepth = kInf; for (auto& p : r.prods) p.minDepth = kInf; }
    bool changed = true;
    while (changed) {
        changed = false;
        for (auto& r : m_rules) {
            for (auto& p : r.prods) {
                int worst = 0;
                bool ok = true;
                for (const auto& s : p.syms) {
                    if (s.terminal) continue;
                    const int d = m_rules[s.nt].minDepth;
                    if (d >= kInf) { ok = false; break; }
                    worst = std::max(worst, d);
                }
                if (!ok) continue;
                const int d = 1 + worst;
                if (d < p.minDepth) { p.minDepth = d; changed = true; }
            }
            int m = kInf;
            for (const auto& p : r.prods) m = std::min(m, p.minDepth);
            if (m < r.minDepth) { r.minDepth = m; changed = true; }
        }
    }
    for (const auto& r : m_rules)
        if (r.minDepth >= kInf)
            throw std::runtime_error("rule <" + r.name +
                "> can never terminate (every alternative recurses forever)");
}

int Grammar::ntIndex(const std::string& name) const {
    const auto it = m_index.find(name);
    return it == m_index.end() ? -1 : it->second;
}

int Grammar::totalProductions() const {
    int n = 0;
    for (const auto& r : m_rules) n += static_cast<int>(r.prods.size());
    return n;
}

std::string Grammar::toString() const {
    std::string o;
    for (const auto& r : m_rules) {
        o += "<" + r.name + "> ::= ";
        for (size_t k = 0; k < r.prods.size(); ++k) {
            if (k) o += " | ";
            for (const auto& s : r.prods[k].syms)
                o += s.terminal ? s.text : "<" + s.text + ">";
        }
        o += "\n";
    }
    return o;
}

// ── derive ────────────────────────────────────────────────────────────────────
namespace {
struct Deriver {
    const std::vector<Rule>& rules;
    ChoiceSource& src;
    const Grammar::DeriveOptions& opt;
    std::vector<Choice>* trace;
    int nodes = 0;

    bool run(int nt, int depth, Node& out) {
        if (++nodes > opt.maxNodes) return false;
        const Rule& r = rules[nt];
        out.nt = nt;
        const int n = static_cast<int>(r.prods.size());
        int k = 0;
        if (n > 1) {
            k = src.choose(nt, n);
            if (k < 0) return false;
            k %= n;
            // Depth budget: force the shallowest production when needed.
            if (depth + r.prods[k].minDepth > opt.maxDepth) {
                int best = 0;
                for (int j = 1; j < n; ++j)
                    if (r.prods[j].minDepth < r.prods[best].minDepth) best = j;
                k = best;
            }
            if (trace) trace->push_back({nt, k});
        }
        out.prod = k;
        const Production& p = r.prods[k];
        out.kids.reserve(p.syms.size());
        for (const Symbol& s : p.syms) {
            Node child;
            if (s.terminal) {
                child.nt = -1;
                child.text = s.text;
                if (++nodes > opt.maxNodes) return false;
            } else if (!run(s.nt, depth + 1, child)) {
                return false;
            }
            out.kids.push_back(std::move(child));
        }
        return true;
    }
};
} // namespace

bool Grammar::derive(ChoiceSource& src, Node& root, const DeriveOptions& opt,
                     std::vector<Choice>* trace) const {
    if (trace) trace->clear();
    Deriver d{m_rules, src, opt, trace};
    root = Node{};
    return d.run(start(), 0, root);
}

std::string Grammar::text(const Node& n) {
    if (n.isTerminal()) return n.text;
    std::string o;
    for (const auto& k : n.kids) o += text(k);
    return o;
}

// ── CodonSource ───────────────────────────────────────────────────────────────
int CodonSource::choose(int /*nt*/, int nProds) {
    if (m_g.empty()) return -1;
    if (m_pos >= m_g.size()) {
        if (m_wraps >= m_maxWraps) return -1;
        ++m_wraps;
        m_pos = 0;
    }
    ++m_used;
    const int c = m_g[m_pos++];
    return (c < 0 ? -c : c) % nProds;
}

} // namespace NeuralStudio::ge
