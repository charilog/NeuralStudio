#pragma once
// NetSpec.h — Mode 6 (GE Studio): turns a derivation tree into a network
// specification (architecture + per-layer bound commands + constructed
// features). Pure STL — no Qt.
//
// Conventions a grammar must follow so the interpreter understands it
// (non-terminal NAMES carry the meaning):
//   <layer>  one hidden layer; inside it the first <size>, the first <act> and
//            the first two <cmd> (lower bound, upper bound) are read
//   <out>    the output layer; its first two <cmd> are read
//   <feat>   one constructed feature; the text of its first <expr> is the
//            formula over x1..xN
//   <cmd>    NOTHING | SHRINK | EXPAND | SHRINK_EXPAND | EXPAND_SHRINK |
//            SHRINK_SHRINK | EXPAND_EXPAND   (or a plain number = factor)
#include "Grammar.h"
#include <memory>
#include <string>
#include <vector>

namespace NeuralStudio::ge {

// ── Bound commands (Tsoulos & Charilogis, Big Data Cogn. Comput. 2025) ───────
double commandFactor(const std::string& token);

// ── Feature expressions ───────────────────────────────────────────────────────
class Expr {
public:
    // Returns nullptr when `src` is not a valid expression over x1..x{nInputs}.
    static std::shared_ptr<Expr> parse(const std::string& src, int nInputs);
    double eval(const double* x) const;      // protected: never NaN/Inf
    const std::string& source() const { return m_src; }
    bool usesVariable() const;     // false for constant formulas such as cos(3)
    int  size() const { return static_cast<int>(m_rpn.size()); }   // after constant folding
private:
    struct Op { int kind; double val; int idx; };
    std::vector<Op> m_rpn;
    std::string     m_src;
    friend struct ExprParser;
};

struct LayerSpec {
    int         size = 8;
    std::string act  = "relu";
    double      lbFactor = 1.0;     // multiplies the base bound (lower side)
    double      ubFactor = 1.0;     // multiplies the base bound (upper side)
    std::string lbCmd, ubCmd;       // for display
};

struct NetSpec {
    std::vector<LayerSpec>   hidden;
    double                   outLbFactor = 1.0, outUbFactor = 1.0;
    std::string              outLbCmd, outUbCmd;
    std::vector<std::string> features;          // validated expressions
    // Periodic input embedding chosen by the grammar (<emb> ::= E(none) | E(<freq>)):
    // every min-max scaled input u gets sin/cos(2*pi*c*u) features, c = embSigma * fixed N(0,1)
    // draws (Gorishniy et al. 2022 style periodic embedding, scale chosen by evolution). 0 = none.
    double                   embSigma = 0.0;

    bool valid() const { return !hidden.empty(); }
    // Canonical key — identical phenotypes share it (used for caching).
    std::string key() const;
    // Human-readable description, e.g. "2L: 16·relu[SHRINK,NOTHING] → 8·tanh[...] | +2 features"
    std::string describe() const;
    // Same architecture + features but all bound factors = 1 (ablation).
    NetSpec withoutBounds() const;
    NetSpec withoutFeatures() const;
    NetSpec withoutEmbedding() const;
    int numLayersWithOutput() const { return static_cast<int>(hidden.size()) + 1; }
};

struct InterpretOptions {
    int maxLayers   = 4;
    int maxFeatures = 4;
    int nInputs     = 1;       // original input count (features may use x1..xN)
    int maxSize     = 512;
    int maxFeatureSize = 15;   // max operations/operands of one feature after constant folding
};

NetSpec interpret(const Grammar& g, const Node& tree, const InterpretOptions& opt);

// ── Grammar presets ───────────────────────────────────────────────────────────
// Preset texts contain the placeholder @VARS@ where the list of input variables
// (x1 | x2 | ... | xN) belongs; it is filled in per dataset by substituteVars().
enum class GrammarPreset { Unified = 0, ArchAndBounds = 1, ArchOnly = 2, UnifiedEmbedding = 3 };
std::string presetGrammar(GrammarPreset p);
std::string substituteVars(const std::string& text, int nInputs);

// Convenience: the unified preset with variables substituted.
std::string defaultUnifiedGrammar(int nInputs);

} // namespace NeuralStudio::ge
