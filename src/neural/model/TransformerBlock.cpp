#include "TransformerBlock.h"

#include <stdexcept>

TransformerBlock::TransformerBlock(
    size_t embeddingDim,
    size_t numHeads,
    size_t hiddenDim
) {
    if (embeddingDim == 0 || numHeads == 0 ||
        embeddingDim % numHeads != 0) {
        throw std::invalid_argument("Invalid TransformerBlock configuration");
    }

    this->embeddingDim = embeddingDim;
    this->numHeads = numHeads;
    this->hiddenDim = hiddenDim == 0 ? embeddingDim * 4 : hiddenDim;

    attention = std::make_unique<MultiHeadAttention>(
        embeddingDim,
        numHeads
    );

    norm1 = std::make_unique<LayerNorm>(embeddingDim);
    norm2 = std::make_unique<LayerNorm>(embeddingDim);

    feedForward = std::make_unique<FeedForward>(
        embeddingDim,
        this->hiddenDim
    );
}

std::shared_ptr<Tensor> TransformerBlock::forward(
    const std::shared_ptr<Tensor>& input,
    bool causal
){
    if (input->getShape().size() != 2 ||
        input->getShape()[1] != embeddingDim) {
        throw std::invalid_argument("Invalid TransformerBlock input shape");
    }

    auto attentionOutput = attention->forward(input, causal);

    auto residual1 = input->add(attentionOutput);

    auto normalized1 = norm1->forward(residual1);

    auto feedForwardOutput = feedForward->forward(normalized1);

    auto residual2 = normalized1->add(feedForwardOutput);

    return norm2->forward(residual2);
}

std::vector<std::shared_ptr<Tensor>> TransformerBlock::getParameters() {
    std::vector<std::shared_ptr<Tensor>> parameters;

    auto attentionParameters = attention->getParameters();
    auto norm1Parameters = norm1->getParameters();
    auto norm2Parameters = norm2->getParameters();
    auto feedForwardParameters = feedForward->getParameters();

    parameters.insert(
        parameters.end(),
        attentionParameters.begin(),
        attentionParameters.end()
    );

    parameters.insert(
        parameters.end(),
        norm1Parameters.begin(),
        norm1Parameters.end()
    );

    parameters.insert(
        parameters.end(),
        norm2Parameters.begin(),
        norm2Parameters.end()
    );

    parameters.insert(
        parameters.end(),
        feedForwardParameters.begin(),
        feedForwardParameters.end()
    );

    return parameters;
}