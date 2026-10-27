#include "NetSpec.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <set>

namespace NeuralStudio::ge {

// ── Bound commands ────────────────────────────────────────────────────────────
double commandFactor(const std::string& tokRaw) {
    std::string t;
    for (char c : tokRaw) if (!std::isspace(static_cast<unsigned char>(c)))
        t.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
    if (t == "NOTHING")       return 1.0;
    if (t == "SHRINK")        return 0.5;
    if (t == "EXPAND")        return 1.5;
    if (t == "SHRINK_EXPAND") return 0.75;   // 0.5 * 1.5
    if (t == "EXPAND_SHRINK") return 0.75;   // 1.5 * 0.5
    if (t == "SHRINK_SHRINK") return 0.25;   // 0.5 * 0.5
    if (t == "EXPAND_EXPAND") return 2.25;   // 1.5 * 1.5
    char* end = nullptr;
    const double v = std::strtod(t.c_str(), &end);
    if (end != t.c_str() && *end == '\0' && std::isfinite(v))
        return std::min(100.0, std::max(0.01, v));
    return 1.0;
}

// ── Expression parser ─────────────────────────────────────────────────────────
enum OpKind { kConst = 0, kVar, kAdd, kSub, kMul, kDiv, kNeg,
              kSin, kCos, kExp, kLog, kSqrt, kAbs, kTanh };

struct ExprParser {
    const std::string& s;
    size_t p = 0;
    int nIn;
    std::vector<Expr::Op>& out;
    bool ok = true;

    void ws() { while (p < s.size() && std::isspace(static_cast<unsigned char>(s[p]))) ++p; }
    bool eat(char c) { ws(); if (p < s.size() && s[p] == c) { ++p; return true; } return false; }
    void emit(int k, double v = 0, int i = 0) { out.push_back({k, v, i}); }

    void expr() {
        term();
        for (;;) {
            if (eat('+')) { term(); emit(kAdd); }
            else if (eat('-')) { term(); emit(kSub); }
            else break;
            if (!ok) return;
        }
    }
    void term() {
        factor();
        for (;;) {
            if (eat('*')) { factor(); emit(kMul); }
            else if (eat('/')) { factor(); emit(kDiv); }
            else break;
            if (!ok) return;
        }
    }
    void factor() {
        ws();
        if (p >= s.size()) { ok = false; return; }
        if (eat('-')) { factor(); emit(kNeg); return; }
        if (eat('(')) { expr(); if (!eat(')')) ok = false; return; }
        const char c = s[p];
        if (std::isdigit(static_cast<unsigned char>(c)) || c == '.') {
            char* end = nullptr;
            const double v = std::strtod(s.c_str() + p, &end);
            if (end == s.c_str() + p) { ok = false; return; }
            p = static_cast<size_t>(end - s.c_str());
            emit(kConst, v);
            return;
        }
        if (std::isalpha(static_cast<unsigned char>(c))) {
            size_t q = p;
            while (q < s.size() && std::isalnum(static_cast<unsigned char>(s[q]))) ++q;
            const std::string id = s.substr(p, q - p);
            p = q;
            if (id.size() >= 2 && id[0] == 'x' &&
                std::all_of(id.begin() + 1, id.end(),
                            [](char ch) { return std::isdigit(static_cast<unsigned char>(ch)); })) {
                const int idx = std::atoi(id.c_str() + 1);
                if (idx < 1 || idx > nIn) { ok = false; return; }
                emit(kVar, 0, idx - 1);
                return;
            }
            int fk = -1;
            if (id == "sin") fk = kSin; else if (id == "cos") fk = kCos;
            else if (id == "exp") fk = kExp; else if (id == "log") fk = kLog;
            else if (id == "sqrt") fk = kSqrt; else if (id == "abs") fk = kAbs;
            else if (id == "tanh") fk = kTanh;
            if (fk < 0 || !eat('(')) { ok = false; return; }
            expr();
            if (!eat(')')) { ok = false; return; }
            emit(fk);
            return;
        }
        ok = false;
    }
};

static bool isUnaryKind(int k) { return k == kNeg || k >= kSin; }

static double finiteClamp(double r) {
    if (!std::isfinite(r)) r = 0.0;
    return std::max(-1e9, std::min(1e9, r));
}

static double unaryOp(int k, double a) {
    switch (k) {
    case kNeg:  return -a;
    case kSin:  return std::sin(a);
    case kCos:  return std::cos(a);
    case kTanh: return std::tanh(a);
    case kExp:  return std::exp(std::max(-30.0, std::min(30.0, a)));
    case kLog:  return std::fabs(a) < 1e-9 ? 0.0 : std::log(std::fabs(a));
    case kSqrt: return std::sqrt(std::fabs(a));
    case kAbs:  return std::fabs(a);
    }
    return a;
}

static double binaryOp(int k, double a, double b) {
    switch (k) {
    case kAdd: return a + b;
    case kSub: return a - b;
    case kMul: return a * b;
    default:   return std::fabs(b) < 1e-9 ? 1.0 : a / b;     // protected division
    }
}

std::shared_ptr<Expr> Expr::parse(const std::string& src, int nInputs) {
    auto e = std::make_shared<Expr>();
    e->m_src = src;
    std::vector<Op> raw;
    ExprParser P{src, 0, nInputs, raw};
    P.expr();
    P.ws();
    if (!P.ok || P.p != src.size() || raw.empty()) return nullptr;

    // Constant folding: sub-expressions without variables collapse into one constant,
    // so log(exp(log(sin(3)))) costs one operand and cannot inflate the size.
    std::vector<Op>& f = e->m_rpn;
    for (const Op& o : raw) {
        if (o.kind == kConst || o.kind == kVar) { f.push_back(o); continue; }
        if (isUnaryKind(o.kind)) {
            if (!f.empty() && f.back().kind == kConst)
                f.back().val = finiteClamp(unaryOp(o.kind, f.back().val));
            else f.push_back(o);
        } else {
            const size_t n = f.size();
            if (n >= 2 && f[n - 1].kind == kConst && f[n - 2].kind == kConst) {
                const double v = finiteClamp(binaryOp(o.kind, f[n - 2].val, f[n - 1].val));
                f.pop_back();
                f.back().val = v;
            } else f.push_back(o);
        }
    }
    return e;
}

bool Expr::usesVariable() const {
    for (const Op& o : m_rpn) if (o.kind == kVar) return true;
    return false;
}

double Expr::eval(const double* x) const {
    thread_local std::vector<double> st;
    st.clear();
    for (const Op& o : m_rpn) {
        if (o.kind == kConst)      st.push_back(o.val);
        else if (o.kind == kVar)   st.push_back(x[o.idx]);
        else if (isUnaryKind(o.kind)) st.back() = unaryOp(o.kind, st.back());
        else {
            const double b = st.back(); st.pop_back();
            st.back() = binaryOp(o.kind, st.back(), b);
        }
    }
    return finiteClamp(st.empty() ? 0.0 : st.back());
}

// ── NetSpec ───────────────────────────────────────────────────────────────────
static std::string f3(double v) {
    char b[32]; std::snprintf(b, sizeof b, "%.3f", v); return b;
}

std::string NetSpec::key() const {
    std::string k;
    for (const auto& l : hidden)
        k += std::to_string(l.size) + ":" + l.act + ":" + f3(l.lbFactor) + ":" + f3(l.ubFactor) + ";";
    k += "O:" + f3(outLbFactor) + ":" + f3(outUbFactor);
    for (const auto& f : features) k += "|" + f;
    if (embSigma > 0) k += "|E:" + f3(embSigma);
    return k;
}

std::string NetSpec::describe() const {
    std::string d = std::to_string(hidden.size()) + "L: ";
    for (size_t i = 0; i < hidden.size(); ++i) {
        const auto& l = hidden[i];
        if (i) d += " \xE2\x86\x92 ";
        d += std::to_string(l.size) + "\xC2\xB7" + l.act +
             "[" + l.lbCmd + "," + l.ubCmd + "]";
    }
    d += " \xE2\x86\x92 out[" + outLbCmd + "," + outUbCmd + "]";
    if (embSigma > 0) d += " | periodic embedding s=" + f3(embSigma);
    if (!features.empty()) {
        d += " | feats: ";
        for (size_t i = 0; i < features.size(); ++i) {
            if (i) d += ", ";
            d += features[i];
        }
    }
    return d;
}

NetSpec NetSpec::withoutFeatures() const {
    NetSpec s = *this;
    s.features.clear();
    return s;
}

NetSpec NetSpec::withoutEmbedding() const {
    NetSpec s = *this;
    s.embSigma = 0.0;
    return s;
}

NetSpec NetSpec::withoutBounds() const {
    NetSpec s = *this;
    for (auto& l : s.hidden) { l.lbFactor = l.ubFactor = 1.0; l.lbCmd = l.ubCmd = "NOTHING"; }
    s.outLbFactor = s.outUbFactor = 1.0;
    s.outLbCmd = s.outUbCmd = "NOTHING";
    return s;
}

// ── Tree interpretation ───────────────────────────────────────────────────────
namespace {
void collect(const Grammar& g, const Node& n, const std::string& name,
             std::vector<const Node*>& out, size_t limit) {
    if (out.size() >= limit) return;
    if (!n.isTerminal() && g.ntName(n.nt) == name) { out.push_back(&n); return; }
    for (const auto& k : n.kids) collect(g, k, name, out, limit);
}

std::string trimmed(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

void walk(const Grammar& g, const Node& n, const InterpretOptions& opt,
          NetSpec& spec, std::set<std::string>& seenFeat) {
    if (n.isTerminal()) return;
    const std::string& nm = g.ntName(n.nt);

    if (nm == "layer") {
        std::vector<const Node*> sz, ac, cm;
        collect(g, n, "size", sz, 1);
        collect(g, n, "act", ac, 1);
        collect(g, n, "cmd", cm, 2);
        LayerSpec L;
        if (!sz.empty()) {
            const int v = std::atoi(trimmed(Grammar::text(*sz[0])).c_str());
            L.size = std::max(1, std::min(opt.maxSize, v));
        }
        if (!ac.empty()) L.act = trimmed(Grammar::text(*ac[0]));
        L.lbCmd = L.ubCmd = "NOTHING";
        if (cm.size() > 0) { L.lbCmd = trimmed(Grammar::text(*cm[0])); L.lbFactor = commandFactor(L.lbCmd); }
        if (cm.size() > 1) { L.ubCmd = trimmed(Grammar::text(*cm[1])); L.ubFactor = commandFactor(L.ubCmd); }
        if (static_cast<int>(spec.hidden.size()) < opt.maxLayers) spec.hidden.push_back(L);
        return;
    }
    if (nm == "out") {
        std::vector<const Node*> cm;
        collect(g, n, "cmd", cm, 2);
        spec.outLbCmd = spec.outUbCmd = "NOTHING";
        if (cm.size() > 0) { spec.outLbCmd = trimmed(Grammar::text(*cm[0])); spec.outLbFactor = commandFactor(spec.outLbCmd); }
        if (cm.size() > 1) { spec.outUbCmd = trimmed(Grammar::text(*cm[1])); spec.outUbFactor = commandFactor(spec.outUbCmd); }
        return;
    }
    if (nm == "emb") {   // E(none) | E(<number>): scale of the periodic input embedding
        const std::string t = trimmed(Grammar::text(n));
        const auto a = t.find('('), b = t.rfind(')');
        const std::string v = (a != std::string::npos && b != std::string::npos && b > a) ? trimmed(t.substr(a + 1, b - a - 1)) : "";
        spec.embSigma = (v.empty() || v == "none") ? 0.0 : std::max(0.0, std::atof(v.c_str()));
        return;
    }
    if (nm == "feat") {
        std::vector<const Node*> ex;
        collect(g, n, "expr", ex, 1);
        const std::string src = trimmed(ex.empty() ? Grammar::text(n) : Grammar::text(*ex[0]));
        if (static_cast<int>(spec.features.size()) < opt.maxFeatures && !seenFeat.count(src)) {
            const auto e = Expr::parse(src, opt.nInputs);
            if (e && e->usesVariable() && e->size() <= opt.maxFeatureSize) {   // constants carry no information; bloat is capped
                seenFeat.insert(src);
                spec.features.push_back(src);
            }
        }
        return;
    }
    for (const auto& k : n.kids) walk(g, k, opt, spec, seenFeat);
}
} // namespace

NetSpec interpret(const Grammar& g, const Node& tree, const InterpretOptions& opt) {
    NetSpec spec;
    spec.outLbCmd = spec.outUbCmd = "NOTHING";
    std::set<std::string> seen;
    walk(g, tree, opt, spec, seen);
    return spec;
}

// ── Grammar presets ───────────────────────────────────────────────────────────
static const char* kCmds =
"<cmd>    ::= NOTHING | SHRINK | EXPAND | SHRINK_EXPAND | EXPAND_SHRINK | SHRINK_SHRINK | EXPAND_EXPAND\n";

std::string presetGrammar(GrammarPreset p) {
    const std::string common =
"<layers> ::= <layer> | <layer> <layers>\n"
"<size>   ::= 2 | 4 | 8 | 16 | 32 | 64 | 128\n"
"<act>    ::= relu | tanh | sigmoid\n";
    switch (p) {
    case GrammarPreset::ArchOnly:
        return
"# GE Studio - architecture only (no bound commands, no features)\n"
"<S>      ::= <layers>\n"
"<layer>  ::= L(<size>,<act>)\n" + common;
    case GrammarPreset::ArchAndBounds:
        return
"# GE Studio - architecture + per-layer bound commands\n"
"# <cmd> acts on the lower / upper bound of the layer's parameters\n"
"<S>      ::= <layers> <out>\n"
"<layer>  ::= L(<size>,<act>,<cmd>,<cmd>)\n"
"<out>    ::= O(<cmd>,<cmd>)\n" + common + kCmds;
    case GrammarPreset::UnifiedEmbedding:
        return
"# GE Studio - unified grammar with a periodic input embedding\n"
"# <emb> chooses the representation of the inputs: none, or sin/cos features of every\n"
"# scaled input at random frequencies of scale <freq> (periodic embedding)\n"
"<S>      ::= <layers> <out> <emb> <feats>\n"
"<layer>  ::= L(<size>,<act>,<cmd>,<cmd>)\n"
"<out>    ::= O(<cmd>,<cmd>)\n"
"<emb>    ::= E(none) | E(<freq>)\n"
"<freq>   ::= 0.5 | 1 | 2 | 4 | 8 | 16\n" + common + kCmds +
"<feats>  ::= | <feat> | <feat> <feats>\n"
"<feat>   ::= F(<expr>)\n"
"<expr>   ::= (<expr><op><expr>) | <func>(<expr>) | <term> | <term>\n"
"<op>     ::= + | - | * | /\n"
"<func>   ::= log | sqrt | abs | tanh | exp | sin\n"
"<term>   ::= <var> | <var> | <const>\n"
"<var>    ::= @VARS@\n"
"<const>  ::= 0.5 | 1 | 2 | 3\n";
    case GrammarPreset::Unified:
    default:
        return
"# GE Studio - unified grammar: architecture + bound commands + features\n"
"# Meaning is carried by the NAMES <layer> <size> <act> <cmd> <out> <feat> <expr>.\n"
"<S>      ::= <layers> <out> <feats>\n"
"<layer>  ::= L(<size>,<act>,<cmd>,<cmd>)\n"
"<out>    ::= O(<cmd>,<cmd>)\n" + common + kCmds +
"# optional constructed features (appended to the original inputs)\n"
"<feats>  ::= | <feat> | <feat> <feats>\n"
"<feat>   ::= F(<expr>)\n"
"<expr>   ::= (<expr><op><expr>) | <func>(<expr>) | <term> | <term>\n"
"<op>     ::= + | - | * | /\n"
"<func>   ::= log | sqrt | abs | tanh | exp | sin\n"
"<term>   ::= <var> | <var> | <const>\n"
"<var>    ::= @VARS@\n"
"<const>  ::= 0.5 | 1 | 2 | 3\n";
    }
}

std::string substituteVars(const std::string& text, int nInputs) {
    std::string vars;
    for (int i = 1; i <= std::max(1, nInputs); ++i) {
        if (i > 1) vars += " | ";
        vars += "x" + std::to_string(i);
    }
    std::string out = text;
    const std::string tag = "@VARS@";
    for (size_t pos = out.find(tag); pos != std::string::npos; pos = out.find(tag, pos + vars.size()))
        out.replace(pos, tag.size(), vars);
    return out;
}

std::string defaultUnifiedGrammar(int nInputs) {
    return substituteVars(presetGrammar(GrammarPreset::Unified), nInputs);
}

} // namespace NeuralStudio::ge
