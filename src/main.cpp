#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <windows.h>

#include "dataset/Dataset.h"
#include "tokenizer/BPETokenizer.h"
#include "neural/model/LanguageModel.h"
#include "neural/model/ModelSaver.h"
#include "training/Trainer.h"
#include "generation/TextGenerator.h"

int main() {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    std::string datasetPath = "data/training.txt";

    std::string trainingData = Dataset::readText(datasetPath);

    BPETokenizer tokenizer;
    tokenizer.train(trainingData, 100);

    Dataset dataset;
    dataset.load(datasetPath, tokenizer);

    std::cout << "Vocabulary Size: "
              << tokenizer.vocabSize() << "\n";

    std::cout << "Dataset Size: "
              << dataset.size() << "\n\n";

    LanguageModel model(
        tokenizer.vocabSize(),
        8,
        4,
        2
    );

    auto parameters = model.getParameters();

    size_t totalParameters = 0;

    for (const auto& parameter : parameters) {
        totalParameters += parameter->size();
    }

    std::cout << "Total Parameters: "
              << totalParameters << "\n";

    Trainer trainer(model, 0.001);

    std::cout << "\nTraining Start\n\n";

    for (int epoch = 0; epoch < 500; epoch++) {
        double loss = trainer.trainEpoch(dataset.getSequences());

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

    std::string prompt = "안녕하세요";

    std::string generatedText = TextGenerator::generate(
        loadedModel,
        tokenizer,
        prompt,
        20,
        0.8
    );

    std::cout << "\nText Generation Start\n\n";
    std::cout << "Prompt: " << prompt << "\n";
    std::cout << "Generated Text:\n";
    std::cout << generatedText << "\n";

    return 0;
}