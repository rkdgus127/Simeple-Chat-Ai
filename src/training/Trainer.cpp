#include "Trainer.h"

#include <stdexcept>

Trainer::Trainer(
    LanguageModel& model,
    double learningRate
)
    : model(model),
      optimizer(model.getParameters(), learningRate) {
}

double Trainer::trainStep(const std::vector<int>& tokens) {
    if (tokens.size() < 2) {
        throw std::invalid_argument("Training sequence must contain at least 2 tokens");
    }

    std::vector<int> inputTokens(
        tokens.begin(),
        tokens.end() - 1
    );

    std::vector<int> targetTokens(
        tokens.begin() + 1,
        tokens.end()
    );

    auto prediction = model.forward(inputTokens, true);

    auto loss = prediction->crossEntropy(targetTokens);

    double lossValue = loss->getData()[0];

    loss->backward();

    optimizer.step();

    return lossValue;
}

double Trainer::trainEpoch(
    const std::vector<std::vector<int>>& dataset
) {
    if (dataset.empty()) {
        throw std::invalid_argument("Training dataset cannot be empty");
    }

    double totalLoss = 0.0;

    for (const auto& tokens : dataset) {
        totalLoss += trainStep(tokens);
    }

    return totalLoss / static_cast<double>(dataset.size());
}