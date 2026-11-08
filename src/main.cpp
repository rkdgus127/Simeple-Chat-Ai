#include <iostream>
#include "tokenizer/BPETokenizer.h"
#include <windows.h>

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

    std::cout << "Original Text:\n";
    std::cout << text << "\n";

    std::cout << "\nToken IDs:\n";

    for (int token : tokens) {
        std::cout << token << " ";
    }

    std::cout << "\n\nToken Count: " << tokens.size() << "\n";
    std::cout << "Vocabulary Size: " << tokenizer.vocabSize() << "\n";

    std::string decoded = tokenizer.decode(tokens);

    std::cout << "\nDecoded Text:\n";
    std::cout << decoded << "\n";

    std::cout << "\nMatch: "
              << (text == decoded ? "TRUE" : "FALSE")
              << "\n";

    return 0;
}