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

    Tensor tx({3, 6});
    tx = 1;
    t1 = 2;
    auto t5 = t1.mm(tx);
    LOG(INFO) << t5;

    Tensor ty({3}), tz({3});
    ty = 1;
    tz = 2;
    auto t6 = ty.mm(tz);
    LOG(INFO) << t6;

    return 0;
}