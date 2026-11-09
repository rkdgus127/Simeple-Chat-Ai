#pragma once

#include <string>
#include <vector>
#include <map>

class BPETokenizer {
public:
    static constexpr int PAD = 256;
    static constexpr int UNK = 257;
    static constexpr int BOS = 258;
    static constexpr int EOS = 259;
    static constexpr int FIRST_MERGE_TOKEN = 260;

    struct Merge {
        int first;
        int second;
        int result;
    };

private:
    std::vector<Merge> merges;
    std::map<int, std::string> vocabulary;

public:
    void train(const std::string& text, int mergeCount);

    std::vector<int> encode(const std::string& text) const;
    std::vector<int> encodeSequence(const std::string& text) const;

    std::string decode(const std::vector<int>& tokens) const;

    size_t vocabSize() const;

    const std::vector<Merge>& getMerges() const;
    const std::map<int, std::string>& getVocabulary() const;
};