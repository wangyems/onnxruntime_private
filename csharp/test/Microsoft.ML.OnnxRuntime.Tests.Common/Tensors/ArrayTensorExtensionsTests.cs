﻿// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using Xunit;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Microsoft.ML.OnnxRuntime.Tests.ArrayTensorExtensions
{
    public class ArrayTensorExtensionsTests
    {

        [Fact]
        public void ConstructFrom1D()
        {
            var array = new int[] { 1, 2, 3, 4 };
            var tensor = array.ToTensor();

            var expectedDims = new int[] { 4 };
            Assert.Equal(tensor.Length, array.Length);
            Assert.Equal(expectedDims, tensor.Dimensions.ToArray());
        }

        [Fact]
        public void ConstructFrom2D()
        {
            var array = new int[,] { { 1, 2 } , { 3, 4 } };
            var tensor = array.ToTensor();

            var expectedDims = new int[] { 2, 2 };
            Assert.Equal(tensor.Length, array.Length);
            Assert.Equal(expectedDims, tensor.Dimensions.ToArray());
        }

        [Fact]
        public void ConstructFrom3D()
        {
            var array = new int[,,] { { { 1, 2 }, { 3, 4 } }, 
                                      { { 5, 6 }, { 7, 8 } } };
            var tensor = array.ToTensor();

            var expectedDims = new int[] { 2, 2, 2 };
            Assert.Equal(tensor.Length, array.Length);
            Assert.Equal(expectedDims, tensor.Dimensions.ToArray());
        }

        [Fact]
        public void ConstructFrom3DWithDim1()
        {
            var array = new int[,,] { { { 1, 2 } }, 
                                      { { 3, 4 } } };
            var tensor = array.ToTensor();

            var expectedDims = new int[] { 2, 1, 2 };
            Assert.Equal(tensor.Length, array.Length);
            Assert.Equal(expectedDims, tensor.Dimensions.ToArray());
        }

        [Fact]
        public void ConstructFrom4D()
        {
            var array = new int[,,,] {
                { { { 1, 2 }, { 3, 4 } },
                  { { 5, 6 }, { 7, 8 } } }
            };
            var tensor = array.ToTensor();

            var expectedDims = new int[] { 1, 2, 2, 2 };
            Assert.Equal(tensor.Length, array.Length);
            Assert.Equal(expectedDims, tensor.Dimensions.ToArray());
        }

        [Fact]
        public void ConstructFrom5D()
        {
            var array = new int[,,,,] {
                { { { { 1, 2 }, { 3, 4 } },
                    { { 5, 6 }, { 7, 8 } } } }
            };

            // 5D requires cast to Array
            Array a = (Array)array;
            var tensor = a.ToTensor<int>();

            var expectedDims = new int[] { 1, 1, 2, 2, 2 };
            Assert.Equal(tensor.Length, array.Length);
            Assert.Equal(expectedDims, tensor.Dimensions.ToArray());
        }
    }
}
