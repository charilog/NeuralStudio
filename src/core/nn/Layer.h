#pragma once
#include "ActivationFunction.h"
#include <vector>

namespace NeuralStudio {

// ─── Layer ────────────────────────────────────────────────────────────────────
//  Dense (fully-connected) layer with optional dropout regularisation.
//
//  Dropout: during training, each neuron's output is independently set to 0
//  with probability `dropoutRate`, and the surviving outputs are scaled by
//  1/(1-rate) so the expected sum is unchanged. At inference time dropout is
//  disabled.  Apply only between hidden layers (never on the output layer).
// ─────────────────────────────────────────────────────────────────────────────
class Layer {
public:
    Layer(int inputSize, int outputSize, Activation activation);

    std::vector<double> forward (const std::vector<double>& input);
    std::vector<double> backward(const std::vector<double>& gradOutput);

    void zeroGradients();
    void updateSGD    (double lr, int bs);
    void updateAdam   (double lr, double b1, double b2, double eps, int t, int bs);
    void updateAdamW  (double lr, double b1, double b2, double eps, int t, int bs, double wd);
    void updateRMSProp(double lr, double rho, double eps, int bs);
    void updateNadam  (double lr, double b1, double b2, double eps, int t, int bs);
    void updateAdaGrad(double lr, double eps, int bs);
    void initAdam();
    void initAdaGrad();

    // Dropout
    void   setDropoutRate(double r)    { m_dropoutRate = r; }
    void   setTrainingMode(bool train) { m_training = train; }
    double dropoutRate() const         { return m_dropoutRate; }

    int        inputSize()  const { return m_in; }
    int        outputSize() const { return m_out; }
    Activation activation() const { return m_act; }
    int        paramCount() const { return (m_in + 1) * m_out; }

    // Read accumulated gradients (used by NSProblem::evalAndGrad for L-BFGS)
    const std::vector<std::vector<double>>& dW() const { return m_dW; }
    const std::vector<double>&              dB() const { return m_dB; }

    std::vector<std::vector<double>> weights;
    std::vector<double>              biases;

private:
    int        m_in;
    int        m_out;
    Activation m_act;

    std::vector<double> m_lastInput;
    std::vector<double> m_lastZ;
    std::vector<double> m_lastOutput;
    std::vector<double> m_dropoutMask;

    std::vector<std::vector<double>> m_dW;
    std::vector<double>              m_dB;

    std::vector<std::vector<double>> m_mW, m_vW;
    std::vector<double>              m_mB, m_vB;
    bool m_adamInit    = false;

    // AdaGrad: accumulated sum of squared gradients (never decays)
    std::vector<std::vector<double>> m_gW;
    std::vector<double>              m_gB;
    bool m_adaGradInit = false;

    double m_dropoutRate = 0.0;
    bool   m_training    = false;
};

} // namespace NeuralStudio
