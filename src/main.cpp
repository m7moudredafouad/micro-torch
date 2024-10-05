#include "tensor.hpp"

int main(int arc, char* argv[]) {
    Tensor t1({3, 3});
    LOG(INFO) << t1[{1, 0}];
    auto t2 = t1 * 0 + 1;
    auto t3 = t2 * 50;
    LOG(INFO) << t1;
    LOG(INFO) << t2;
    LOG(INFO) << t3;

    return 0;
}