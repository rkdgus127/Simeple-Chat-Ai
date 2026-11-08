#pragma once

#include <string>
#include <vector>

class Tokenizer {
public:
    static std::vector<int> encode(const std::string& text);

    static std::string decode(const std::vector<int>& tokens);

    static size_t vocabSize();
};