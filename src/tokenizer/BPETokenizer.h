#pragma once

#include <string>
#include <vector>
#include <map>

class BPETokenizer {
private:
    struct Merge {
        int first;
        int second;
        int result;
    };

    std::vector<Merge> merges;
    std::map<int, std::string> vocabulary;

    static std::vector<int> toBytes(const std::string& text);

    static std::vector<int> applyMerge(
        const std::vector<int>& tokens,
        int first,
        int second,
        int result
    );

public:
    void train(const std::string& text, int mergeCount);

    std::vector<int> encode(const std::string& text) const;

    std::string decode(const std::vector<int>& tokens) const;

    size_t vocabSize() const;
};