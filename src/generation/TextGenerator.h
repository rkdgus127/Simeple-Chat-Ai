#pragma once

#include <string>
#include <vector>

#include "../tokenizer/BPETokenizer.h"
#include "../neural/model/LanguageModel.h"

class TextGenerator {
public:
    static std::string generate(
        LanguageModel& model,
        BPETokenizer& tokenizer,
        const std::string& prompt,
        size_t maxNewTokens,
        double temperature = 1.0
    );

private:
    static int selectToken(
        const std::vector<double>& logits,
        double temperature
    );
};