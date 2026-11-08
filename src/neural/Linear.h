#pragma once

#include "../core/Tensor.h"
#include <memory>
#include <cstddef>

class Linear {
private:
    std::shared_ptr<Tensor> weights;
    std::shared_ptr<Tensor> bias;

    size_t inputSize;
    size_t outputSize;

public:
    Linear(size_t inputSize, size_t outputSize);

    std::shared_ptr<Tensor> forward(
        const std::shared_ptr<Tensor>& input
    );

    std::shared_ptr<Tensor> getWeights() const;
    std::shared_ptr<Tensor> getBias() const;
};