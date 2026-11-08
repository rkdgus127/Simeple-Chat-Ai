#include "TextGenerator.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

class BPETokenizer;
class LanguageModel;

int TextGenerator::selectToken(
    const std::vector<double>& logits,
    double temperature
) {
    if (logits.empty()) {
        throw std::invalid_argument("Logits cannot be empty");
    }

    if (temperature < 0.0) {
        throw std::invalid_argument("Temperature cannot be negative");
    }

    if (temperature == 0.0) {
        return static_cast<int>(
            std::distance(
                logits.begin(),
                std::max_element(logits.begin(), logits.end())
            )
        );
    }

    double maxLogit = *std::max_element(
        logits.begin(),
        logits.end()
    );

    std::vector<double> probabilities;
    probabilities.reserve(logits.size());

    double total = 0.0;

    for (double logit : logits) {
        double probability = std::exp(
            (logit - maxLogit) / temperature
        );

        probabilities.push_back(probability);
        total += probability;
    }

    for (double& probability : probabilities) {
        probability /= total;
    }

    static std::mt19937 generator(
        std::random_device{}()
    );

    std::discrete_distribution<int> distribution(
        probabilities.begin(),
        probabilities.end()
    );

    return distribution(generator);
}

std::string TextGenerator::generate(
    LanguageModel& model,
    BPETokenizer& tokenizer,
    const std::string& prompt,
    size_t maxNewTokens,
    double temperature
) {
    if (prompt.empty()) {
        throw std::invalid_argument("Prompt cannot be empty");
    }

    if (temperature < 0.0) {
        throw std::invalid_argument("Temperature cannot be negative");
    }

    std::vector<int> tokens = tokenizer.encode(prompt);

    if (tokens.empty()) {
        throw std::invalid_argument("Tokenized prompt cannot be empty");
    }

    if (model.getParameters().empty()) {
        throw std::runtime_error("Model has no parameters");
    }

    for (size_t i = 0; i < maxNewTokens; i++) {
        auto output = model.forward(tokens, true);

        size_t vocabSize = output->getShape()[1];
        size_t lastPosition = output->getShape()[0] - 1;

        std::vector<double> logits(
            output->getData().begin() + lastPosition * vocabSize,
            output->getData().begin() + (lastPosition + 1) * vocabSize
        );

        int nextToken = selectToken(logits, temperature);

        tokens.push_back(nextToken);
    }

    return tokenizer.decode(tokens);
}