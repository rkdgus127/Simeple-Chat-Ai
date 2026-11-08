#pragma once

#include <memory>
#include <vector>

#include "../../core/Tensor.h"

class LayerNorm {
private:
    size_t embeddingDim;

    std::shared_ptr<Tensor> gamma;
    std::shared_ptr<Tensor> beta;

    double epsilon;

public:
    explicit LayerNorm(size_t embeddingDim, double epsilon = 1e-5);

    std::shared_ptr<Tensor> forward(
        const std::shared_ptr<Tensor>& input
    );

    std::vector<std::shared_ptr<Tensor>> getParameters();
};