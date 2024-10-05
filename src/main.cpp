#include "tensor.hpp"

int main(int arc, char* argv[]) {
    Tensor tmp({2, 2});
    auto& old_v = tmp[1];
    tmp[0] = 1;
    old_v = 2;
    LOG(INFO) << "old_v: " << tmp[1] << ", new_v: " << tmp[0] << std::endl;
    return 0;
}