#include "FeedForward.h"

#include "Activation.h"

#include <stdexcept>

FeedForward::FeedForward(size_t embeddingDim, size_t hiddenDim) {
    if (embeddingDim == 0) {
        throw std::invalid_argument("Embedding dimension cannot be zero");
    }

    this->embeddingDim = embeddingDim;
    this->hiddenDim = hiddenDim == 0 ? embeddingDim * 4 : hiddenDim;

    linear1 = std::make_unique<Linear>(
        embeddingDim,
        this->hiddenDim
    );

    linear2 = std::make_unique<Linear>(
        this->hiddenDim,
        embeddingDim
    );
}

std::shared_ptr<Tensor> FeedForward::forward(
    const std::shared_ptr<Tensor>& input
) {
    if (input->getShape().size() != 2 ||
        input->getShape()[1] != embeddingDim) {
        throw std::invalid_argument("Invalid FeedForward input shape");
        }

    auto hidden = linear1->forward(input);

    hidden = Activation::relu(hidden);

    return linear2->forward(hidden);
}

std::vector<std::shared_ptr<Tensor>> FeedForward::getParameters() {
    return {
        linear1->getWeights(),
        linear1->getBias(),
        linear2->getWeights(),
        linear2->getBias()
    };
}