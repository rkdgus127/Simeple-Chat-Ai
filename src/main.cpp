#include <iostream>
#include <windows.h>

#include "neural/attention/MultiHeadAttention.h"
#include "tokenizer/BPETokenizer.h"
#include "neural/layers/Embedding.h"
#include "neural/encoding/PositionalEncoding.h"
#include "neural/attention/SelfAttention.h"
#include "neural/normalization/LayerNorm.h"

int main() {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    BPETokenizer tokenizer;

    std::string trainingData =
        "hello hello hello help help "
        "hello hello help "
        "안녕하세요 안녕하세요 "
        "AI를 만들고 있습니다 AI를 만들고 있습니다 ";

    tokenizer.train(trainingData, 30);

    std::string text = "hello 안녕하세요 AI를 만들고 있습니다";

    auto tokens = tokenizer.encode(text);

    Embedding embedding(tokenizer.vocabSize(), 8);

    PositionalEncoding positionalEncoding(8);

    SelfAttention attention(8);

    auto embedded = embedding.forward(tokens);

    auto positioned = positionalEncoding.forward(embedded);

    auto output = attention.forward(positioned);

    std::cout << "Token Count: "
              << tokens.size() << std::endl;

    std::cout << "\nAttention Output Shape:" << std::endl;

    for (size_t dimension : output->getShape()) {
        std::cout << dimension << " ";
    }

    std::cout << "\n\nFirst Token Attention Output:" << std::endl;

    for (size_t i = 0; i < 8; i++) {
        std::cout << output->getData()[i] << " ";
    }

    output->sum()->backward();

    std::cout << "\n\nQuery Weight Gradient:" << std::endl;

    for (double gradient : attention.getParameters()[0]->getGrad()) {
        std::cout << gradient << " ";
    }

    std::cout << "\n\nAttention Parameters: ";

    size_t parameterCount = 0;
    MultiHeadAttention multiAttention(8, 4);

    auto multiOutput = multiAttention.forward(embedded);

    std::cout << "Multi-Head Attention Output Shape:\n";

    for (auto dim : multiOutput->getShape()) {
        std::cout << dim << " ";
    }

    std::cout << "\n\n";

    multiOutput->sum()->backward();

    auto parameters = multiAttention.getParameters();

    size_t totalParameters = 0;

    for (const auto& parameter : parameters) {
        totalParameters += parameter->size();
    }

    std::cout << "Attention Parameters: "
              << totalParameters << "\n";

    std::cout << "Query Weight Gradient:\n";

    auto queryWeights = parameters[0];

    for (size_t i = 0; i < 8; i++) {
        std::cout << queryWeights->getGrad()[i] << " ";
    }

    std::cout << "\n";

    LayerNorm layerNorm(8);

    auto normalized = layerNorm.forward(embedded);

    std::cout << "LayerNorm Output Shape:\n";

    for (auto dim : normalized->getShape()) {
        std::cout << dim << " ";
    }

    std::cout << "\n\n";

    std::cout << "First Token Normalized:\n";

    for (size_t i = 0; i < 8; i++) {
        std::cout << normalized->getData()[i] << " ";
    }

    std::cout << "\n\n";

    normalized->sum()->backward();

    auto layerNormParameters = layerNorm.getParameters();

    std::cout << "Gamma Gradient:\n";

    for (double value : layerNormParameters[0]->getGrad()) {
        std::cout << value << " ";
    }

    std::cout << "\n\nBeta Gradient:\n";

    for (double value : layerNormParameters[1]->getGrad()) {
        std::cout << value << " ";
    }

    std::cout << "\n";

    std::cout << parameterCount << std::endl;

    return 0;
}
