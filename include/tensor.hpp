#pragma once
#include <functional>
#include <unordered_set>

#include "includes.hpp"
#include "storage.hpp"

namespace micro {

void with_no_grad();
void with_grad();

enum class Type : uint8_t { UINT32 = 0, INT32, FLOAT32, UNKONWN };

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

class TensorImpl {
   public:
    TensorImpl() = default;

    TensorImpl(std::vector<uint32_t> shape, Type dtype = Type::FLOAT32) : m_dtype(dtype), m_shape(std::move(shape)) {
        set_default_strides();
        m_storage = Storage(num_bytes());
    }

    uint32_t size() const {
        LOG_IF(FATAL, m_shape.size() == 0);
        uint32_t size = 1;
        for (size_t i = 0; i < m_shape.size(); i++) {
            size *= m_shape[i];
        }

        return size;
    }

    uint32_t num_bytes() const { return size() * sizeof(Element); }

    void set_default_strides() {
        LOG_IF(FATAL, m_shape.size() == 0);

        m_stride.resize(m_shape.size());
        m_stride[m_shape.size() - 1] = 1;

        for (int8_t i = (int8_t)m_shape.size() - 2; i >= 0; i--) {
            m_stride[i] = m_stride[i + 1] * m_shape[i + 1];
        }
    }

    Element operator[](const std::vector<uint32_t>& indices) const {
        return const_cast<TensorImpl*>(this)->operator[](indices);
    }

    Element& at(const std::vector<uint32_t>& indices) { return this->operator[](indices); }

    Element& operator[](const std::vector<uint32_t>& indices) {
        LOG_IF(FATAL, indices.size() != m_shape.size())
            << "Indices size=" << indices.size() << " don't match the full_shape=" << m_shape.size();
        uint32_t offset = 0, i = 0;

        for (auto idx : indices) {
            LOG_IF(FATAL, idx >= m_shape[i])
                << "index is out of range, full_shape= " << m_shape[i] << " and index=" << idx;
            offset += idx * m_stride[i];
            i++;
        }

        return this->operator[](offset);
    }

   private:
    Element operator[](uint32_t offset) const { return const_cast<TensorImpl*>(this)->operator[](offset); }

    Element& operator[](uint32_t offset) {
        LOG_IF(FATAL, offset >= Size()) << "index out of range";
        return *reinterpret_cast<Element*>(m_storage.at((m_offset + offset) * sizeof(Element)));
    }

    Element broadcasted_read(const std::vector<uint32_t>& indices) const;

   public:
    friend std::ostream& operator<<(std::ostream& os, TensorImpl& t);
    friend TensorImpl get_element_wise_empty_output(const TensorImpl& in1, const TensorImpl& in2);
    friend TensorImpl get_matmul_empty_output(const TensorImpl& in1, const TensorImpl& in2);

   private:
    Type m_dtype = Type::FLOAT32;
    int64_t m_offset = 0;

   private:
    std::vector<uint32_t> m_shape, m_stride;
    Storage m_storage;
};

class Tensor {
   public:
    Tensor() = default;

    Tensor(std::vector<uint32_t> shape, Type dtype = Type::FLOAT32)
        : m_tensor_impl(std::make_shared<TensorImpl>(dtype, shape)) {}

    Tensor(TensorImpl tensor_impl) : m_tensor_impl(std::make_shared<TensorImpl>(tensor_impl)) {}

    uint32_t defined() const { return m_tensor_impl != nullptr; }

    uint32_t size() const {
        LOG_IF(FATAL, !defined()) << "calling size() on undefined tensor";
        return m_tensor_impl->size();
    };

    void set_default_strides() {
        LOG_IF(FATAL, !defined()) << "calling set_default_strides() on undefined tensor";
        m_tensor_impl->set_default_strides();
    }

    void requires_grad(bool requires_grad) {
        LOG_IF(FATAL, m_grad_fn != nullptr) << "you can only change requires_grad flags of leaf variables";
        m_requires_grad = requires_grad;
    }

    Element& operator[](const std::vector<uint32_t>& indices) {
        LOG_IF(FATAL, !defined()) << "trying to read elements from undefined tensor";
        return m_tensor_impl->operator[](indices);
    }

    Element operator[](const std::vector<uint32_t>& indices) const {
        return const_cast<Tensor*>(this)->operator[](indices);
    }

    Element& at(const std::vector<uint32_t>& indices) { return this->operator[](indices); }

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    Tensor mm(const Tensor& other) const;
    void backward();

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
        add_impl(*this, tensor, *this);
    }

    template <typename T>
    typename std::enable_if_t<!std::is_same_v<T, Tensor>, void> operator-=(T value) {
        Tensor tensor({1});
        WRITE_ELEMENT(tensor[0], value);
        sub_impl(*this, tensor, *this);
    }

    template <typename T>
    typename std::enable_if_t<!std::is_same_v<T, Tensor>, void> operator*=(T value) {
        Tensor tensor({1});
        WRITE_ELEMENT(tensor[0], value);
        mul_impl(*this, tensor, *this);
    }

    template <typename T>
    typename std::enable_if_t<!std::is_same_v<T, Tensor>, void> operator/=(T value) {
        Tensor tensor({1});
        WRITE_ELEMENT(tensor[0], value);
        div_impl(*this, tensor, *this);
    }

    template <typename T>
    void operator=(T value) {
        for (uint32_t i = 0; i < Size(); i++) {
            WRITE_ELEMENT(this->operator[](i), value);
        }
    }

#undef WRITE_ELEMENT

    // Forward Functions
    static void add_impl(const Tensor& in1, const Tensor& in2, Tensor& out);
    static void sub_impl(const Tensor& in1, const Tensor& in2, Tensor& out);
    static void mul_impl(const Tensor& in1, const Tensor& in2, Tensor& out);
    static void div_impl(const Tensor& in1, const Tensor& in2, Tensor& out);
    static void matmul_impl(const Tensor& in1, const Tensor& in2, Tensor& out);

    // Backward Functions
    static void add_backward_impl(Tensor& out);
    static void sub_backward_impl(Tensor& out);
    static void mul_backward_impl(Tensor& out);
    static void div_backward_impl(Tensor& out);
    static void matmul_backward_impl(Tensor& out);

    void topological_sort(Tensor& curr, std::vector<Tensor*>& list, std::unordered_set<Tensor*>& visited);

   public:
    std::shared_ptr<TensorImpl> grad;

   private:
    std::shared_ptr<TensorImpl> m_tensor_impl;
    std::vector<Tensor> m_parents;
    std::function<void(Tensor&)> m_grad_fn;
    bool m_requires_grad = false;
};

};  // namespace micro