#pragma once

#include "../core/Tensor.h"
#include <memory>
#include <vector>

class Adam {
private:
    struct State {
        std::vector<double> m;
        std::vector<double> v;
    };

    std::vector<std::shared_ptr<Tensor>> parameters;
    std::vector<State> states;

    double learningRate;
    double beta1;
    double beta2;
    double epsilon;

    size_t timestep;

public:
    Adam(
        std::vector<std::shared_ptr<Tensor>> parameters,
        double learningRate = 0.001,
        double beta1 = 0.9,
        double beta2 = 0.999,
        double epsilon = 1e-8
    );

    void step();
};