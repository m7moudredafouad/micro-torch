#include "tensor.hpp"

int main() {
    using namespace micro;
    Tensor t6({3}), t7({3}), t8({3});
    t6.requires_grad(true);

    t6 = 2;
    t6 = t6 * t6;
    // t7 = t6 * t6;
    // t7 = t6 * t6 + t6;
    t7 = t6 + 1;
    t8 = t7 + t6;
    t8.backward();

    LOG(INFO) << *(t6.grad);
    LOG(INFO) << *(t7.grad);
    LOG(INFO) << *(t8.grad);
    return 0;
}