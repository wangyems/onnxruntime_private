# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# pylint: disable=missing-docstring

import unittest

import numpy as np
import onnxruntime_pybind11_state as torch_ort
import torch


class OrtOpTests(unittest.TestCase):
    """test cases for supported eager ops"""

    def get_device(self):
        return torch_ort.device()

    def test_add(self):
        device = self.get_device()
        cpu_ones = torch.Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        ort_ones = cpu_ones.to(device)
        cpu_twos = cpu_ones + cpu_ones
        ort_twos = ort_ones + ort_ones
        assert torch.allclose(cpu_twos, ort_twos.cpu())

    def test_type_promotion_add(self):
        device = self.get_device()
        x = torch.ones(2, 5, dtype=torch.int64)
        y = torch.ones(2, 5, dtype=torch.float32)
        ort_x = x.to(device)
        ort_y = y.to(device)
        ort_z = ort_x + ort_y
        assert ort_z.dtype == torch.float32
        assert torch.allclose(ort_z.cpu(), (x + y))

    def test_add_alpha(self):
        device = self.get_device()
        cpu_ones = torch.Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        ort_ones = cpu_ones.to(device)
        assert torch.allclose(torch.add(cpu_ones, cpu_ones, alpha=2.5), torch.add(ort_ones, ort_ones, alpha=2.5).cpu())

    def test_mul_bool(self):
        device = self.get_device()
        cpu_ones = torch.ones(3, 3, dtype=bool)
        ort_ones = cpu_ones.to(device)
        assert torch.allclose(torch.mul(cpu_ones, cpu_ones), torch.mul(ort_ones, ort_ones).cpu())

    # TODO: Add BFloat16 test coverage
    def test_add_(self):
        device = self.get_device()
        cpu_ones = torch.Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        ort_ones = cpu_ones.to(device)
        cpu_twos = cpu_ones
        cpu_twos += cpu_ones
        ort_twos = ort_ones
        ort_twos += ort_ones
        assert torch.allclose(cpu_twos, ort_twos.cpu())

    def test_sin_(self):
        device = self.get_device()
        cpu_sin_pi_ = torch.Tensor([np.pi])
        torch.sin_(cpu_sin_pi_)
        ort_sin_pi_ = torch.Tensor([np.pi]).to(device)
        torch.sin_(ort_sin_pi_)
        cpu_sin_pi = torch.sin(torch.Tensor([np.pi]))
        ort_sin_pi = torch.sin(torch.Tensor([np.pi]).to(device))
        assert torch.allclose(cpu_sin_pi, ort_sin_pi.cpu())
        assert torch.allclose(cpu_sin_pi_, ort_sin_pi_.cpu())
        assert torch.allclose(ort_sin_pi.cpu(), ort_sin_pi_.cpu())

    def test_sin(self):
        device = self.get_device()
        cpu_sin_pi = torch.sin(torch.Tensor([np.pi]))
        ort_sin_pi = torch.sin(torch.Tensor([np.pi]).to(device))
        assert torch.allclose(cpu_sin_pi, ort_sin_pi.cpu())

    def test_zero_like(self):
        device = self.get_device()
        ones = torch.ones((10, 10), dtype=torch.float32)
        cpu_zeros = torch.zeros_like(ones)
        ort_zeros = torch.zeros_like(ones.to(device))
        assert torch.allclose(cpu_zeros, ort_zeros.cpu())

    def test_gemm(self):
        device = self.get_device()
        cpu_ones = torch.Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        ort_ones = cpu_ones.to(device)
        cpu_ans = cpu_ones * 4
        ort_ans = torch_ort.custom_ops.gemm(ort_ones, ort_ones, ort_ones, 1.0, 1.0, 0, 0)
        assert torch.allclose(cpu_ans, ort_ans.cpu())

    def test_batchnormalization_inplace(self):
        device = self.get_device()
        x = torch.Tensor([[[[-1, 0, 1]], [[2.0, 3.0, 4.0]]]]).to(device)
        s = torch.Tensor([1.0, 1.5]).to(device)
        bias = torch.Tensor([0.0, 1.0]).to(device)
        mean = torch.Tensor([0.0, 3.0]).to(device)
        var = torch.Tensor([1.0, 1.5]).to(device)
        y, mean_out, var_out = torch_ort.custom_ops.batchnorm_inplace(x, s, bias, mean, var, 1e-5, 0.9)
        assert torch.allclose(x.cpu(), y.cpu()), "x != y"
        assert torch.allclose(mean.cpu(), mean_out.cpu()), "mean != mean_out"
        assert torch.allclose(var.cpu(), var_out.cpu()), "var != var_out"

    def test_max(self):
        cpu_tensor = torch.rand(10, 10)
        ort_tensor = cpu_tensor.to("ort")
        ort_min = ort_tensor.max()
        cpu_min = cpu_tensor.max()
        assert torch.allclose(cpu_min, ort_min.cpu())
        assert cpu_min.dim() == ort_min.dim()

    def test_min(self):
        cpu_tensor = torch.rand(10, 10)
        ort_tensor = cpu_tensor.to("ort")
        ort_min = ort_tensor.min()
        cpu_min = cpu_tensor.min()
        assert torch.allclose(cpu_min, ort_min.cpu())
        assert cpu_min.dim() == ort_min.dim()

    def test_equal(self):
        device = self.get_device()
        cpu_a = torch.Tensor([1.0, 1.5])
        ort_a = cpu_a.to(device)
        cpu_b = torch.Tensor([1.0, 1.5])
        ort_b = cpu_b.to(device)
        cpu_c = torch.Tensor([1.0, 1.8])
        ort_c = cpu_c.to(device)
        cpu_d = torch.Tensor([1.0, 1.5, 2.1])
        ort_d = cpu_d.to(device)
        cpu_e = torch.Tensor([[1.0, 1.5]])
        ort_e = cpu_e.to(device)
        assert torch.equal(cpu_a, cpu_b)
        assert torch.equal(ort_a, ort_b)
        assert not torch.equal(cpu_a, cpu_c)
        assert not torch.equal(ort_a, ort_c)
        assert not torch.equal(cpu_a, cpu_d)
        assert not torch.equal(ort_a, ort_d)
        assert not torch.equal(cpu_a, cpu_e)
        assert not torch.equal(ort_a, ort_e)

    def test_torch_ones(self):
        device = self.get_device()
        cpu_ones = torch.ones((10, 10))
        ort_ones = cpu_ones.to(device)
        ort_ones_device = torch.ones((10, 10), device=device)
        assert torch.allclose(cpu_ones, ort_ones.cpu())
        assert torch.allclose(cpu_ones, ort_ones_device.cpu())

    def test_narrow(self):
        cpu_tensor = torch.rand(10, 10)
        cpu_narrow = cpu_tensor.narrow(0, 5, 5)
        ort_narrow = cpu_narrow.to("ort")
        assert torch.allclose(cpu_narrow, ort_narrow.cpu())

    def test_zero_stride(self):
        device = self.get_device()
        cpu_tensor = torch.empty_strided(size=(6, 1024, 512), stride=(0, 0, 0))
        assert cpu_tensor.storage().size() == 1
        ort_tensor_copied = cpu_tensor.to(device)
        assert torch.allclose(cpu_tensor, ort_tensor_copied.cpu())
        ort_tensor = torch.empty_strided(size=(6, 1024, 512), stride=(0, 0, 0), device=device)
        assert ort_tensor.is_ort
        assert ort_tensor.stride() == (0, 0, 0)
        cpu_tensor_copied = ort_tensor.cpu()
        assert cpu_tensor_copied.stride() == (0, 0, 0)

    def test_softmax(self):
        device = self.get_device()
        cpu_tensor = torch.rand(3, 5)
        ort_tensor = cpu_tensor.to(device)
        cpu_result = torch.softmax(cpu_tensor, dim=1)
        ort_result = torch.softmax(ort_tensor, dim=1)
        assert torch.allclose(cpu_result, ort_result.cpu())

    def test_addmm(self):
        device = self.get_device()
        size = 4
        ort_tensor = torch.ones([size, size]).to(device)
        input_bias = torch.ones([size]).to(device)
        output = torch.addmm(input_bias, ort_tensor, ort_tensor)
        expected = torch.ones([size, size]) * 5
        assert torch.equal(output.to("cpu"), expected)

    def test_argmax(self):
        device = self.get_device()
        cpu_tensor = torch.rand(3, 5)
        ort_tensor = cpu_tensor.to(device)
        cpu_out_tensor = torch.tensor([], dtype=torch.long)
        ort_out_tensor = torch.tensor([], dtype=torch.long, device=device)
        cpu_result = torch.argmax(cpu_tensor, dim=1, out=cpu_out_tensor)
        ort_result = torch.argmax(ort_tensor, dim=1, out=ort_out_tensor)
        # assert torch.equal(cpu_out_tensor.to(device), ort_out_tensor)
        assert torch.equal(cpu_result, ort_result.cpu())
        # assert torch.allclose(cpu_out_tensor, ort_out_tensor.cpu)
        assert torch.allclose(cpu_result, ort_result.cpu())
        assert cpu_result.dim() == ort_result.dim()

    def test_eq_tensor(self):
        device = self.get_device()
        cpu_a = torch.Tensor([1.0, 1.5])
        ort_a = cpu_a.to(device)
        cpu_b = torch.Tensor([1.0, 1.5])
        ort_b = cpu_b.to(device)
        cpu_out_tensor = torch.tensor([], dtype=torch.bool)
        ort_out_tensor = cpu_out_tensor.to(device)
        cpu_a_b_eq_result = torch.eq(cpu_a, cpu_b, out=cpu_out_tensor)
        ort_a_b_eq_result = torch.eq(ort_a, ort_b, out=ort_out_tensor)
        assert torch.equal(cpu_a_b_eq_result.to(device), ort_a_b_eq_result)
        # print(cpu_out_tensor)
        # print(ort_out_tensor)
        # print(cpu_a_b_eq_result)
        # print(ort_a_b_eq_result)
        # assert torch.equal(cpu_out_tensor.to(device), ort_out_tensor)
        assert torch.allclose(cpu_a_b_eq_result, ort_a_b_eq_result.to("cpu"))
        # assert torch.allclose(cpu_out_tensor, ort_out_tensor.to("cpu"))

    def test_eq_scalar(self):
        device = self.get_device()
        cpu_int = torch.tensor([1, 1], dtype=torch.int32)
        cpu_scalar_int = torch.scalar_tensor(1, dtype=torch.int)

        cpu_float = torch.tensor([1.1, 1.0], dtype=torch.float32)
        cpu_scalar_float = torch.scalar_tensor(1.0, dtype=torch.float32)
        # print(f"cpu tensor: {cpu_c}, cpu tensor type {cpu_c.dtype}")
        ort_float = cpu_float.to(device)
        ort_int = cpu_int.to(device)
        ort_scalar_float = cpu_scalar_float.to(device)
        ort_scalar_int = cpu_scalar_int.to(device)
        # print(f"cpu tensor: {ort_c}, cpu tensor type {ort_c.dtype}")
        # compare int to int, int to float, float to float, float to int
        cpu_int_int_result = torch.eq(cpu_int, cpu_scalar_int)
        cpu_int_float_result = torch.eq(cpu_int, cpu_scalar_float)
        cpu_float_float_result = torch.eq(cpu_float, cpu_scalar_float)
        cpu_float_int_result = torch.eq(cpu_float, cpu_scalar_int)

        ort_int_int_result = torch.eq(ort_int, ort_scalar_int)
        # ort_int_float_result = torch.eq(ort_int, ort_scalar_float)
        ort_float_float_result = torch.eq(ort_float, ort_scalar_float)
        # ort_float_int_result = torch.eq(ort_float, ort_scalar_int)

        assert torch.equal(cpu_int_int_result.to(device), ort_int_int_result)
        # assert torch.equal(cpu_int_float_result.to(device), ort_int_float_result)
        assert torch.equal(cpu_float_float_result.to(device), ort_float_float_result)
        # assert torch.equal(cpu_float_int_result.to(device), ort_float_int_result)

    def test_fill(self):
        device = self.get_device()
        cpu_a = torch.Tensor([1.0, 1.5])
        ort_a = cpu_a.to(device)
        cpu_a.fill_(3.2)
        print(cpu_a)
        print(cpu_a.size())
        # ort_a.fill_(3.2)
        print(ort_a.size())
        print(ort_a.shape)
        cpu_copy = ort_a.to("cpu")
        print(cpu_copy)


if __name__ == "__main__":
    unittest.main()
