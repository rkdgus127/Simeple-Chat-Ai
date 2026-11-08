#include <iostream>
#include <windows.h>

#include "neural/attention/MultiHeadAttention.h"
#include "tokenizer/BPETokenizer.h"
#include "neural/layers/Embedding.h"
#include "neural/encoding/PositionalEncoding.h"
#include "neural/attention/SelfAttention.h"
#include "neural/layers/FeedForward.h"
#include "neural/loss/Loss.h"
#include "neural/model/LanguageModel.h"
#include "neural/model/TransformerBlock.h"
#include "neural/normalization/LayerNorm.h"
#include "training/Trainer.h"
#include "neural/model/ModelSaver.h"

int main()
{
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

    FeedForward feedForward(8);

    auto feedForwardOutput = feedForward.forward(embedded);

    std::cout << "Feed Forward Output Shape:\n";

    for (auto dim : feedForwardOutput->getShape()) {
        std::cout << dim << " ";
    }

    std::cout << "\n\n";

    std::cout << "First Token Feed Forward Output:\n";

    for (size_t i = 0; i < 8; i++) {
        std::cout << feedForwardOutput->getData()[i] << " ";
    }

    std::cout << "\n\n";

    feedForwardOutput->sum()->backward();

    auto feedForwardParameters = feedForward.getParameters();

    size_t feedForwardTotalParameters = 0;

    for (const auto& parameter : feedForwardParameters) {
        feedForwardTotalParameters += parameter->size();
    }

    std::cout << "Feed Forward Parameters: "
              << feedForwardTotalParameters << "\n";

    std::cout << "First Layer Weight Gradient:\n";

    for (size_t i = 0; i < 16; i++) {
        std::cout << feedForwardParameters[0]->getGrad()[i] << " ";
    }

    std::cout << "\n";

    TransformerBlock transformerBlock(8, 4);

    auto transformerOutput = transformerBlock.forward(embedded);

    std::cout << "Transformer Block Output Shape:\n";

    for (auto dim : transformerOutput->getShape()) {
        std::cout << dim << " ";
    }

    std::cout << "\n\n";

    std::cout << "First Token Transformer Output:\n";

    for (size_t i = 0; i < 8; i++) {
        std::cout << transformerOutput->getData()[i] << " ";
    }

    std::cout << "\n\n";

    transformerOutput->sum()->backward();

    auto transformerParameters = transformerBlock.getParameters();

    size_t transformerTotalParameters = 0;

    for (const auto& parameter : transformerParameters) {
        transformerTotalParameters += parameter->size();
    }

    std::cout << "Transformer Parameters: "
              << transformerTotalParameters << "\n";

    std::cout << "Parameter Tensor Count: "
              << transformerParameters.size() << "\n";

    TransformerBlock causalBlock(8, 4);

    auto causalOutput = causalBlock.forward(embedded, true);

    std::cout << "Causal Transformer Output Shape:\n";

    for (auto dim : causalOutput->getShape()) {
        std::cout << dim << " ";
    }

    std::cout << "\n\n";

    std::cout << "First Token Causal Output:\n";

    for (size_t i = 0; i < 8; i++) {
        std::cout << causalOutput->getData()[i] << " ";
    }

    std::cout << "\n\n";

    causalOutput->sum()->backward();

    auto causalParameters = causalBlock.getParameters();

    size_t causalTotalParameters = 0;

    for (const auto& parameter : causalParameters) {
        causalTotalParameters += parameter->size();
    }

    std::cout << "Causal Transformer Parameters: "
              << causalTotalParameters << "\n";

    auto logits = Tensor::create({
    2.0, 1.0, 0.1,
    0.5, 2.5, 0.3,
    1.0, 0.2, 3.0
}, {3, 3}, true);

    std::vector<int> targets = {
        0,
        1,
        2
    };

    auto loss = Loss::crossEntropy(logits, targets);

    std::cout << "Cross Entropy Loss:\n";
    std::cout << loss->getData()[0] << "\n\n";

    loss->backward();

    std::cout << "Logits Gradient:\n";

    for (double gradient : logits->getGrad()) {
        std::cout << gradient << " ";
    }

    std::cout << "\n";

    LanguageModel model(
        tokenizer.vocabSize(),
        8,
        4,
        2
    );

    auto modelOutput = model.forward(tokens);

    std::cout << "Language Model Output Shape:\n";

    for (auto dim : modelOutput->getShape()) {
        std::cout << dim << " ";
    }

    std::cout << "\n\n";

    std::cout << "First Token Logits:\n";

    for (size_t i = 0; i < 10; i++) {
        std::cout << modelOutput->getData()[i] << " ";
    }

    std::cout << "\n\n";

    auto modelParameters = model.getParameters();

    size_t modelTotalParameters = 0;

    for (const auto& parameter : modelParameters) {
        modelTotalParameters += parameter->size();
    }

    std::cout << "Total Parameters: "
              << modelTotalParameters << "\n";

    std::cout << "Parameter Tensor Count: "
              << modelParameters.size() << "\n";

    auto modelLoss = modelOutput->crossEntropy(
        std::vector<int>(tokens.size(), 0)
    );

    std::cout << "Test Loss: "
              << modelLoss->getData()[0] << "\n";

    Trainer trainer(model, 0.001);

    std::vector<int> trainingTokens = tokens;

    std::cout << "\nTraining Start\n\n";

    for (int epoch = 0; epoch <= 500; epoch++) {
        double loss = trainer.trainStep(trainingTokens);

        if (epoch % 50 == 0) {
            std::cout << "Epoch: "
                      << epoch
                      << " | Loss: "
                      << loss
                      << "\n";
        }
    }

    std::cout << "\nTraining Complete\n";

    ModelSaver::save(model, "model.bin");

    std::cout << "Model Saved!\n";

    LanguageModel loadedModel(
        tokenizer.vocabSize(),
        8,
        4,
        2
    );

    ModelSaver::load(loadedModel, "model.bin");

    std::cout << "Model Loaded!\n";

    auto loadedOutput = loadedModel.forward(tokens);

    std::cout << "Loaded Model Output:\n";

    for (size_t i = 0; i < 10; i++) {
        std::cout << loadedOutput->getData()[i] << " ";
    }

    std::cout << "\n";

    return 0;
}