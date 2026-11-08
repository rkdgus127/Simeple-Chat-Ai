#pragma once

#include "../../core/Tensor.h"

#include <memory>
#include <vector>

class Embedding {
private:
    size_t vocabSize;
    size_t embeddingDim;

    std::shared_ptr<Tensor> weights;

public:
    Embedding(size_t vocabSize, size_t embeddingDim);

    std::shared_ptr<Tensor> forward(const std::vector<int>& tokens);

    std::shared_ptr<Tensor> getWeights() const;

    size_t getVocabSize() const;

    size_t getEmbeddingDim() const;
};