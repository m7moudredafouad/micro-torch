#include "tensor.hpp"

int main(int arc, char* argv[]) {
    Tensor t1({3, 3}), t2({1, 3});
    t1 = t1 * 0 + 5;
    LOG(INFO) << t1;
    t2 = t2 * 0 + 1;
    LOG(INFO) << t2;
    auto t3 = t2 * 50;
    LOG(INFO) << t3;

    auto t4 = t1 + t3;
    LOG(INFO) << t4;

    return 0;
}