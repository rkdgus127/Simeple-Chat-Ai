#include "BPETokenizer.h"

#include <map>
#include <utility>

void BPETokenizer::train(const std::string& text, int mergeCount) {
    merges.clear();
    vocabulary.clear();

    for (int i = 0; i < 256; i++) {
        vocabulary[i] = std::string(1, static_cast<char>(i));
    }

    vocabulary[PAD] = "<PAD>";
    vocabulary[UNK] = "<UNK>";
    vocabulary[BOS] = "<BOS>";
    vocabulary[EOS] = "<EOS>";

    std::vector<int> tokens;

    for (unsigned char byte : text) {
        tokens.push_back(static_cast<int>(byte));
    }

    int nextTokenId = FIRST_MERGE_TOKEN;

    for (int iteration = 0; iteration < mergeCount; iteration++) {
        std::map<std::pair<int, int>, int> frequencies;

        for (size_t i = 0; i + 1 < tokens.size(); i++) {
            frequencies[{tokens[i], tokens[i + 1]}]++;
        }

        if (frequencies.empty()) {
            break;
        }

        auto bestPair = frequencies.begin();
        int maxFrequency = 0;

        for (auto it = frequencies.begin(); it != frequencies.end(); ++it) {
            if (it->second > maxFrequency) {
                maxFrequency = it->second;
                bestPair = it;
            }
        }

        if (maxFrequency < 2) {
            break;
        }

        int first = bestPair->first.first;
        int second = bestPair->first.second;

        vocabulary[nextTokenId] =
            vocabulary[first] + vocabulary[second];

        std::vector<int> mergedTokens;

        for (size_t i = 0; i < tokens.size();) {
            if (i + 1 < tokens.size() &&
                tokens[i] == first &&
                tokens[i + 1] == second) {

                mergedTokens.push_back(nextTokenId);
                i += 2;
            } else {
                mergedTokens.push_back(tokens[i]);
                i++;
            }
        }

        merges.push_back({first, second, nextTokenId});

        tokens = std::move(mergedTokens);

        nextTokenId++;
    }
}

std::vector<int> BPETokenizer::encode(const std::string& text) const {
    std::vector<int> tokens;

    for (unsigned char byte : text) {
        tokens.push_back(static_cast<int>(byte));
    }

    for (const auto& merge : merges) {
        std::vector<int> mergedTokens;

        for (size_t i = 0; i < tokens.size();) {
            if (i + 1 < tokens.size() &&
                tokens[i] == merge.first &&
                tokens[i + 1] == merge.second) {

                mergedTokens.push_back(merge.result);
                i += 2;
            } else {
                mergedTokens.push_back(tokens[i]);
                i++;
            }
        }

        tokens = std::move(mergedTokens);
    }

    return tokens;
}

std::vector<int> BPETokenizer::encodeSequence(const std::string& text) const {
    std::vector<int> tokens = {BOS};

    std::vector<int> encoded = encode(text);

    tokens.insert(tokens.end(), encoded.begin(), encoded.end());
    tokens.push_back(EOS);

    return tokens;
}

std::string BPETokenizer::decode(const std::vector<int>& tokens) const {
    std::string result;

    for (int token : tokens) {
        if (token == BOS || token == PAD) {
            continue;
        }

        if (token == EOS) {
            break;
        }

        auto it = vocabulary.find(token);

        if (it == vocabulary.end() || token == UNK) {
            result += "<UNK>";
            continue;
        }

        result += it->second;
    }

    return result;
}

size_t BPETokenizer::vocabSize() const {
    return vocabulary.size();
}

const std::vector<BPETokenizer::Merge>& BPETokenizer::getMerges() const {
    return merges;
}

const std::map<int, std::string>& BPETokenizer::getVocabulary() const {
    return vocabulary;
}