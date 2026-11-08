#include "Tokenizer.h"

#include <stdexcept>

std::vector<int> Tokenizer::encode(const std::string& text) {
    std::vector<int> tokens;

    tokens.reserve(text.size());

    for (unsigned char character : text) {
        tokens.push_back(static_cast<int>(character));
    }

    return tokens;
}

std::string Tokenizer::decode(const std::vector<int>& tokens) {
    std::string text;

    text.reserve(tokens.size());

    for (int token : tokens) {
        if (token < 0 || token > 255) {
            throw std::invalid_argument("Invalid token ID");
        }

        text.push_back(static_cast<char>(token));
    }

    return text;
}

size_t Tokenizer::vocabSize() {
    return 256;
}