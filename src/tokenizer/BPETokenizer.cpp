#include "BPETokenizer.h"

#include <algorithm>
#include <stdexcept>

std::vector<int> BPETokenizer::toBytes(const std::string& text) {
    std::vector<int> tokens;

    for (unsigned char character : text) {
        tokens.push_back(static_cast<int>(character));
    }

    return tokens;
}

std::vector<int> BPETokenizer::applyMerge(
    const std::vector<int>& tokens,
    int first,
    int second,
    int result
) {
    std::vector<int> merged;

    for (size_t i = 0; i < tokens.size(); i++) {
        if (i + 1 < tokens.size() &&
            tokens[i] == first &&
            tokens[i + 1] == second) {

            merged.push_back(result);
            i++;
        } else {
            merged.push_back(tokens[i]);
        }
    }

    return merged;
}

void BPETokenizer::train(const std::string& text, int mergeCount) {
    merges.clear();
    vocabulary.clear();

    for (int i = 0; i < 256; i++) {
        vocabulary[i] = std::string(1, static_cast<char>(i));
    }

    std::vector<int> tokens = toBytes(text);

    int nextToken = 256;

    for (int iteration = 0; iteration < mergeCount; iteration++) {
        std::map<std::pair<int, int>, int> frequencies;

        for (size_t i = 0; i + 1 < tokens.size(); i++) {
            frequencies[{tokens[i], tokens[i + 1]}]++;
        }

        if (frequencies.empty()) {
            break;
        }

        auto best = frequencies.begin();

        for (auto it = frequencies.begin(); it != frequencies.end(); ++it) {
            if (it->second > best->second) {
                best = it;
            }
        }

        if (best->second < 2) {
            break;
        }

        int first = best->first.first;
        int second = best->first.second;

        std::string mergedText =
            vocabulary.at(first) + vocabulary.at(second);

        vocabulary[nextToken] = mergedText;

        merges.push_back({
            first,
            second,
            nextToken
        });

        tokens = applyMerge(tokens, first, second, nextToken);

        nextToken++;
    }
}

std::vector<int> BPETokenizer::encode(const std::string& text) const {
    std::vector<int> tokens = toBytes(text);

    for (const auto& merge : merges) {
        tokens = applyMerge(
            tokens,
            merge.first,
            merge.second,
            merge.result
        );
    }

    return tokens;
}

std::string BPETokenizer::decode(const std::vector<int>& tokens) const {
    std::string result;

    for (int token : tokens) {
        auto it = vocabulary.find(token);

        if (it == vocabulary.end()) {
            throw std::invalid_argument("Invalid token ID");
        }

        result += it->second;
    }

    return result;
}

size_t BPETokenizer::vocabSize() const {
    return vocabulary.size();
}