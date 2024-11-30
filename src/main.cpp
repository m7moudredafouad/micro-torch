#include "tensor.hpp"

int main() {
    // Tensor t1({3, 3}), t2({1, 3});
    // t1 = 1;
    // t1 *= 16.1;
    // t1 /= 4;
    // t1 -= 1;
    // LOG(INFO) << t1;

    // t2 = 2;
    // t2[{0, 1}] = 0;
    // LOG(INFO) << t2;

    // auto t3 = t1 * t2;
    // LOG(INFO) << t3;

    // auto t4 = t1 + t3;
    // LOG(INFO) << t4;

    // Tensor tx({3, 6});
    // tx = 1;
    // t1 = 2;
    // auto t5 = t1.mm(tx);
    // LOG(INFO) << t5;

    // Tensor ty({3}), tz({3});
    // ty = 1;
    // tz = 2.1;
    // auto t6 = ty.mm(tz);
    // LOG(INFO) << t6;

    Tensor t6({3}), t7({3}), t8({3});
    t6.requires_grad(true);

    t6 = 2;
    t6 = t6 * t6;
    // t7 = t6 * t6;
    t7 = t6 + 1;
    t8 = t7 + t6;
    t8.backward();

    LOG(INFO) << *(t6.grad);
    LOG(INFO) << *(t7.grad);
    LOG(INFO) << *(t8.grad);
    return 0;
}