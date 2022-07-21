Tensor gemm(const Tensor &A, const Tensor &B, const Tensor &C, double alpha, double beta, int64_t transA,
            int64_t transB);
std::tuple<Tensor &, Tensor &, Tensor &> batchnorm_inplace(
    Tensor &X, const Tensor &scale, const Tensor &B, Tensor &input_mean, Tensor &input_var, const double epsilon,
    const double momentum);  // {"schema": "batchnorm_inplace(Tensor(a!) X, Tensor scale, Tensor b, Tensor(b!)
                             // input_mean, Tensor(c!) input_var, float epsilon, float momentum) -> (Tensor(a!),
                             // Tensor(b!), Tensor(c!))", "dispatch": "False", "default": "True"}
Tensor my_cat(TensorList tensors,
              int64_t dim);  // {"schema": "my_cat(Tensor[] tensors, int dim=0) -> Tensor", "dispatch": "True",
                             // "default": "False"}
