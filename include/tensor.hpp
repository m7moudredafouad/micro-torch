#pragma once
#include <functional>
#include <unordered_set>

#include "includes.hpp"
#include "storage.hpp"

namespace micro {

void with_no_grad();
void with_grad();

class AutogradContext;

enum class Type : uint8_t { UINT32 = 0, INT32, FLOAT32, UNKONWN };

std::ostream& operator<<(std::ostream& os, const Type& type);

struct Element {
    union Data {
        uint32_t u32 = 0;
        int32_t i32;
        float f32;
    } data;

    Element() = default;

    template <typename T>
    Element(T value) {
        if constexpr (std::is_same_v<T, int32_t>) {
            data.i32 = value;
        } else if constexpr (std::is_same_v<T, uint32_t>) {
            data.u32 = value;
        } else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            data.f32 = float(value);
        } else {
            LOG(FATAL) << "Element Type is not supported";
        }
    }

    operator float() const { return this->data.f32; }

    operator float&() { return this->data.f32; }

    operator int32_t() const { return this->data.i32; }

    operator int32_t&() { return this->data.i32; }

    operator uint32_t() const { return this->data.u32; }

    operator uint32_t&() { return this->data.u32; }
};

class Tensor {
   public:
    Tensor() = default;

    Tensor(std::vector<uint32_t> shape, Type dtype = Type::FLOAT32) : m_dtype(dtype), m_shape(std::move(shape)) {
        set_default_strides();
        m_storage = Storage(number_bytes());
    }

    void set_default_strides();

    size_t size() const;

    uint32_t number_bytes() const { return size() * sizeof(Element); }

    Tensor grad();

    void reset_grad();

    void requires_grad(bool requires_grad) {
        LOG_IF(FATAL, m_grad_fn != nullptr) << "you can only change requires_grad flags of leaf variables";
        m_requires_grad = requires_grad;
    }

    Element operator[](const std::initializer_list<uint32_t>& indices) const {
        return const_cast<Tensor*>(this)->operator[](std::vector<uint32_t>{indices});
    }

    Element& operator[](const std::initializer_list<uint32_t>& indices) {
        return this->operator[](std::vector<uint32_t>{indices});
    }

    Element operator[](const std::vector<uint32_t>& indices) const {
        return const_cast<Tensor*>(this)->operator[](indices);
    }

    Element& at(const std::vector<uint32_t>& indices) { return this->operator[](indices); }

    Tensor transpose(uint32_t dim0 = 0, uint32_t dim1 = 1) {
        Tensor t = *this;
        std::swap(t.m_shape[dim0], t.m_shape[dim1]);
        std::swap(t.m_stride[dim0], t.m_stride[dim1]);

        t.m_saved_context = nullptr;
        return t;
    }

    Element& operator[](const std::vector<uint32_t>& indices);
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    Tensor mm(const Tensor& other) const;
    Tensor sum(uint32_t dim, bool keep_dims = false) const;

#define WRITE_ELEMENT(out, value)                                           \
    {                                                                       \
        switch (m_dtype) {                                                  \
            case Type::UINT32:                                              \
                out.data.i32 = uint32_t(value);                             \
                break;                                                      \
            case Type::INT32:                                               \
                out.data.i32 = int32_t(value);                              \
                break;                                                      \
            case Type::FLOAT32:                                             \
                out.data.f32 = float(value);                                \
                break;                                                      \
            default:                                                        \
                LOG(FATAL) << "Can't write value with unknown tensor type"; \
        }                                                                   \
    }

    Tensor& operator=(const std::vector<Element>& values) {
        LOG_IF(FATAL, values.size() != size())
            << "Can't assign an array of size " << values.size() << " to a tensor of size " << size();

        for (size_t i = 0; i < size(); i++) {
            WRITE_ELEMENT((this->operator[](i)), values[i]);
        }

        return *this;
    }

    template <typename T>
    Tensor operator+(T value) const {
        Tensor tensor({1});
        WRITE_ELEMENT(tensor[0], value);
        return this->operator+(tensor);
    }

    template <typename T>
    Tensor operator-(T value) const {
        Tensor tensor({1});
        WRITE_ELEMENT(tensor[0], value);
        return this->operator-(tensor);
    }

    template <typename T>
    Tensor operator*(T value) const {
        Tensor tensor({1});
        WRITE_ELEMENT(tensor[0], value);
        return this->operator*(tensor);
    }

    template <typename T>
    Tensor operator/(T value) const {
        Tensor tensor({1});
        WRITE_ELEMENT(tensor[0], value);
        return this->operator/(tensor);
    }

    template <typename T>
    typename std::enable_if_t<!std::is_same_v<T, Tensor>, void> operator+=(T value) {
        Tensor tensor({1});
        WRITE_ELEMENT(tensor[0], value);
        add_forward_impl(*this, tensor, *this);
    }

    template <typename T>
    typename std::enable_if_t<!std::is_same_v<T, Tensor>, void> operator-=(T value) {
        Tensor tensor({1});
        WRITE_ELEMENT(tensor[0], value);
        sub_forward_impl(*this, tensor, *this);
    }

    template <typename T>
    typename std::enable_if_t<!std::is_same_v<T, Tensor>, void> operator*=(T value) {
        Tensor tensor({1});
        WRITE_ELEMENT(tensor[0], value);
        mul_forward_impl(*this, tensor, *this);
    }

    template <typename T>
    typename std::enable_if_t<!std::is_same_v<T, Tensor>, void> operator/=(T value) {
        Tensor tensor({1});
        WRITE_ELEMENT(tensor[0], value);
        div_forward_impl(*this, tensor, *this);
    }

    template <typename T>
    void operator=(T value) {
        for (uint32_t i = 0; i < size(); i++) {
            WRITE_ELEMENT(this->operator[](i), value);
        }
    }

    void backward();

#undef WRITE_ELEMENT

   private:
    Element operator[](uint32_t offset) const { return const_cast<Tensor*>(this)->operator[](offset); }

    Element& operator[](uint32_t offset) {
        LOG_IF(FATAL, offset >= size()) << "index out of range";
        return *reinterpret_cast<Element*>(m_storage.at((m_offset + offset) * sizeof(Element)));
    }

    Element broadcasted_read(const std::vector<uint32_t>& indices) const;

   private:
    Type m_dtype = Type::FLOAT32;
    bool m_requires_grad = false;
    int64_t m_offset = 0;

   private:
    std::vector<uint32_t> m_shape, m_stride;
    Storage m_storage;

   private:
    std::shared_ptr<AutogradContext> m_saved_context = std::make_shared<AutogradContext>();
    std::function<void(Tensor&)> m_grad_fn;

   public:
    friend std::ostream& operator<<(std::ostream& os, const Tensor& t);
    friend Tensor get_element_wise_empty_output(const Tensor& in1, const Tensor& in2);
    friend Tensor get_matmul_empty_output(const Tensor& in1, const Tensor& in2);
    friend void align_gradient_with_tensor(const Tensor& tensor, Tensor& gradient);

    // Forward Functions
    static void add_forward_impl(const Tensor& in1, const Tensor& in2, Tensor& out);
    static void sub_forward_impl(const Tensor& in1, const Tensor& in2, Tensor& out);
    static void mul_forward_impl(const Tensor& in1, const Tensor& in2, Tensor& out);
    static void div_forward_impl(const Tensor& in1, const Tensor& in2, Tensor& out);
    static void matmul_forward_impl(const Tensor& in1, const Tensor& in2, Tensor& out);
    static void sum_forward_impl(const Tensor& in, const uint32_t dim, Tensor& out);

    // Backward Functions
    static void add_backward_impl(Tensor& out);
    static void sub_backward_impl(Tensor& out);
    static void mul_backward_impl(Tensor& out);
    static void div_backward_impl(Tensor& out);
    static void matmul_backward_impl(Tensor& out);
    static void sum_backward_impl(Tensor& out);

    void topological_sort(Tensor& curr, std::vector<Tensor>& list,
                          std::unordered_set<std::shared_ptr<AutogradContext>>& visited);
};

class AutogradContext {
   public:
    void save_for_backward(const std::vector<Tensor>& tensors_to_save) {
        m_saved_tensors.insert(m_saved_tensors.end(), tensors_to_save.begin(), tensors_to_save.end());
    }

    std::vector<Tensor> get_saved_variables() { return m_saved_tensors; }

    std::shared_ptr<Tensor>& grad() { return m_grad; }

   private:
    std::vector<Tensor> m_saved_tensors;
    std::shared_ptr<Tensor> m_grad = nullptr;
};

};  // namespace micro