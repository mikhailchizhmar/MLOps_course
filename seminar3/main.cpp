#include <torch/script.h>

int main() {
    torch::jit::script::Module module = torch::jit::load("model_scripted.pt");
    at::Tensor input = torch::ones({1, 3, 224, 224});
    at::Tensor output = module.forward({input}).toTensor();
}
