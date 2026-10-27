#pragma once
// Grammar.h — Mode 6 (GE Studio): BNF grammar, derivation tree and mapping.
// Pure C++20 / STL (no Qt) so it can be unit-tested headlessly.
//
// BNF text format
//   <rule> ::= alt1 | alt2 | alt3
//   * a rule starts with  <name> ::=  ; following lines continue it until the
//     next rule header
//   * anything that is not <name> is a literal terminal fragment
//   * '#' or '//' start a comment (to end of line)
//   * the first rule is the start symbol
//   * '|' separates alternatives (it cannot appear inside a terminal)
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <stdexcept>

namespace NeuralStudio::ge {

struct Symbol {
    bool        terminal = true;
    std::string text;       // terminal text
    int         nt = -1;    // non-terminal index when !terminal
};

struct Production {
    std::vector<Symbol> syms;
    int minDepth = 0;       // shortest derivation depth of this production
};

struct Rule {
    std::string             name;
    std::vector<Production> prods;
    int                     minDepth = 0;   // min over productions
};

// One step of a derivation: which production was chosen for which non-terminal.
struct Choice { int nt = 0; int prod = 0; };

// Derivation tree node. nt < 0 means terminal leaf (text holds the literal).
struct Node {
    int               nt   = -1;
    int               prod = 0;
    std::string       text;
    std::vector<Node> kids;
    bool isTerminal() const { return nt < 0; }
};

// Source of production choices during derivation.
struct ChoiceSource {
    virtual ~ChoiceSource() = default;
    // Return an index in [0, nProds) for non-terminal `nt`.
    // Return -1 if the source is exhausted (derivation fails).
    virtual int choose(int nt, int nProds) = 0;
};

class Grammar {
public:
    // Throws std::runtime_error with a readable message on malformed input.
    static Grammar parse(const std::string& bnf);

    const std::vector<Rule>& rules() const { return m_rules; }
    int  start() const { return 0; }
    int  ntIndex(const std::string& name) const;   // -1 if unknown
    const std::string& ntName(int i) const { return m_rules[i].name; }
    int  totalProductions() const;
    std::string toString() const;

    struct DeriveOptions {
        int maxDepth = 24;       // hard limit on recursion depth
        int maxNodes = 4000;     // hard limit on tree size
    };

    // Derive a tree from the start symbol. Choices come from `src`.
    // When the depth budget gets tight the production with the smallest
    // minDepth is forced, so the derivation always terminates.
    // `trace` (optional) receives the choices actually applied, in order.
    // Returns false only if `src` ran out of choices or maxNodes was hit.
    bool derive(ChoiceSource& src, Node& root, const DeriveOptions& opt,
                std::vector<Choice>* trace = nullptr) const;

    // Concatenate the terminals of a tree (the "program text").
    static std::string text(const Node& n);

private:
    std::vector<Rule>          m_rules;
    std::map<std::string,int>  m_index;
    void computeMinDepths();
};

// ── Integer-codon source: classic GE mapping (codon mod n), with wrapping ────
class CodonSource : public ChoiceSource {
public:
    CodonSource(const std::vector<int>& genes, int maxWraps)
        : m_g(genes), m_maxWraps(maxWraps) {}
    int choose(int nt, int nProds) override;
    int codonsUsed() const { return m_used; }
    int wraps() const { return m_wraps; }
private:
    const std::vector<int>& m_g;
    int m_maxWraps;
    size_t m_pos = 0;
    int m_wraps = 0;
    int m_used  = 0;
};

} // namespace NeuralStudio::ge
