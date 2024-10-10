#include "tensor.hpp"

int main() {
    Tensor t1({3, 3}), t2({1, 3});
    t1 = 1;
    t1 *= 16;
    t1 /= 4;
    t1 -= 1;
    LOG(INFO) << t1;

    t2 = 2;
    t2[{0, 1}] = 0;
    LOG(INFO) << t2;

    auto t3 = t1 * t2;
    LOG(INFO) << t3;

    auto t4 = t1 + t3;
    LOG(INFO) << t4;

    return 0;
}