#pragma once

#include "../layers/Linear.h"
#include "../../core/Tensor.h"

#include <memory>
#include <vector>

class SelfAttention {
private:
    size_t embeddingDim;

    Linear queryLayer;
    Linear keyLayer;
    Linear valueLayer;

public:
    explicit SelfAttention(size_t embeddingDim);

    std::shared_ptr<Tensor> forward(
        const std::shared_ptr<Tensor>& input
    );

    std::vector<std::shared_ptr<Tensor>> getParameters() const;
};