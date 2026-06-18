#pragma once
// TaskType is defined in NeuralNetwork.h — use a forward-only include
// to avoid circular dependencies (RBFNetwork does not depend on NeuralNetwork class)
#include "core/nn/NeuralNetwork.h"
#include <vector>
#include <string>
#include <memory>

namespace NeuralStudio {

// ─── Enumerations ─────────────────────────────────────────────────────────────
enum class RBFKernel {
    Gaussian,           // φ(r) = exp(−r²/2σ²)   ← default, most common
    Multiquadric,       // φ(r) = sqrt(r² + σ²)
    InvMultiquadric     // φ(r) = 1 / sqrt(r² + σ²)
};

enum class RBFWidthStrategy {
    MaxDist,            // σ = d_max / sqrt(2k)   — global, fast
    NearestNeighbor,    // σᵢ = dist to nearest centre (per-centre)
    MeanNN              // σᵢ = mean dist to 3 nearest centres
};

// ─── RBFConfig ────────────────────────────────────────────────────────────────
struct RBFConfig {
    int              nCenters      = 20;
    RBFKernel        kernel        = RBFKernel::Gaussian;
    RBFWidthStrategy widthStrategy = RBFWidthStrategy::MaxDist;
    double           widthScale    = 1.0;    // multiplier on computed σ
    double           ridgeLambda   = 1e-4;   // L2 regularisation for output layer
    int              kmeansMaxIter = 150;
    double           valSplit      = 0.20;
};

// ─── RBFNetwork ───────────────────────────────────────────────────────────────
//  Architecture:
//    input (nIn) → RBF hidden layer (k centres) → linear output (nOut)
//
//  Training is ANALYTIC (not gradient-based):
//    1. K-means++ to find centres  μ₁…μₖ
//    2. Heuristic to compute widths σ₁…σₖ
//    3. Build design matrix  Φ  (N × k+1,  last column = 1 for bias)
//    4. Solve normal equations  (ΦᵀΦ + λI) W = ΦᵀY  for output weights W
// ─────────────────────────────────────────────────────────────────────────────
class RBFNetwork {
public:
    using Vec = std::vector<double>;
    using Mat = std::vector<Vec>;

    // Initialise structure (called before training)
    void init(int nInputs, int nOutputs, TaskType task, const RBFConfig& cfg);

    // Called by RBFTrainer after fitting
    void setCenters(Mat centres, Vec widths);
    void setOutputWeights(Mat W);      // W is (k+1) × nOut,  row k = bias row

    // ── Inference ─────────────────────────────────────────────────────────────
    Vec predict(const Vec& x) const;   // returns raw output (regression) or probs

    // Compute the k+1 activation vector (last element = 1 for bias)
    Vec activations(const Vec& x) const;

    // Build design matrix for a batch of inputs:  Φ  is  N × (k+1)
    Mat designMatrix(const Mat& X) const;

    // ── State queries ─────────────────────────────────────────────────────────
    bool          isBuilt()  const { return m_built; }
    TaskType      taskType() const { return m_task; }
    int           nInputs()  const { return m_nIn; }
    int           nOutputs() const { return m_nOut; }
    int           nCenters() const { return int(m_centres.size()); }
    const RBFConfig& config() const { return m_cfg; }
    RBFConfig&       config()       { return m_cfg; }  // mutable access for training params

    const Mat& centres() const { return m_centres; }
    const Vec& widths()  const { return m_widths;  }
    const Mat& weights() const { return m_W; }     // (k+1) × nOut

private:
    double kernelValue(double r2, double sigma) const;

    bool      m_built = false;
    TaskType  m_task  = TaskType::Regression;
    int       m_nIn = 0, m_nOut = 0;
    RBFConfig m_cfg;

    Mat m_centres;   // [k][nIn]
    Vec m_widths;    // [k]
    Mat m_W;         // [k+1][nOut]  — row k = biases
};

} // namespace NeuralStudio
