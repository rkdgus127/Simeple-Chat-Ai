#pragma once

#include "../../core/Tensor.h"

#include <memory>
#include <cstddef>

class PositionalEncoding {
private:
    size_t embeddingDim;

public:
    explicit PositionalEncoding(size_t embeddingDim);

    std::shared_ptr<Tensor> forward(
        const std::shared_ptr<Tensor>& input
    ) const;

    size_t getEmbeddingDim() const;
};