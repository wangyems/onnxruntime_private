/*
 * Copyright (c) 2022 Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.LongBuffer;

/**
 * A Java object wrapping an OnnxSparseTensor.
 *
 * <p>Sparse tensors support a variety of formats, and the {@link #getValue} method returns a
 * different static inner class representing each type.
 */
public final class OnnxSparseTensor extends OnnxTensorLike {
  private final SparseTensorType sparseTensorType;

  // Held to prevent deallocation while used in native code.
  private final LongBuffer indices;
  private final LongBuffer innerIndices;
  private final Buffer data;

  OnnxSparseTensor(
      long nativeHandle,
      long allocatorHandle,
      SparseTensorType sparseType,
      TensorInfo info,
      LongBuffer indices,
      Buffer data) {
    this(nativeHandle, allocatorHandle, sparseType, info, indices, null, data);
  }

  OnnxSparseTensor(
      long nativeHandle,
      long allocatorHandle,
      SparseTensorType sparseType,
      TensorInfo info,
      LongBuffer indices,
      LongBuffer innerIndices,
      Buffer data) {
    super(nativeHandle, allocatorHandle, info);
    this.sparseTensorType = sparseType;
    this.indices = indices;
    this.innerIndices = innerIndices;
    this.data = data;
  }

  /**
   * Creates a Sparse Tensor in ORT from the Java side representation.
   *
   * @param env The OrtEnvironment.
   * @param tensor The Java side representation.
   * @return The sparse tensor in ORT.
   * @throws OrtException If the tensor could not be created or was invalid.
   */
  public static OnnxSparseTensor createSparseTensor(OrtEnvironment env, SparseTensor tensor)
      throws OrtException {
    return createSparseTensor(env, env.defaultAllocator, tensor);
  }

  static OnnxSparseTensor createSparseTensor(
      OrtEnvironment env, OrtAllocator allocator, SparseTensor tensor) throws OrtException {
    if ((!env.isClosed()) && (!allocator.isClosed())) {
      TensorInfo info = TensorInfo.constructFromSparseTensor(tensor);
      OrtUtil.BufferTuple indicesTuple =
          OrtUtil.prepareBuffer(tensor.getIndices(), OnnxJavaType.INT64);
      OrtUtil.BufferTuple dataTuple = OrtUtil.prepareBuffer(tensor.getData(), info.type);
      LongBuffer indicesBuf;
      if (indicesTuple.data instanceof LongBuffer) {
        indicesBuf = (LongBuffer) indicesTuple.data;
      } else if (indicesTuple.data instanceof ByteBuffer) {
        indicesBuf = ((ByteBuffer) indicesTuple.data).asLongBuffer();
      } else {
        throw new IllegalStateException(
            "Unexpected type of indices buffer, found "
                + indicesTuple.data.getClass()
                + ", expected LongBuffer");
      }
      return new OnnxSparseTensor(
          createSparseTensorFromBuffer(
              OnnxRuntime.ortApiHandle,
              allocator.handle,
              indicesTuple.data,
              indicesTuple.pos,
              indicesTuple.size,
              dataTuple.data,
              dataTuple.pos,
              info.shape,
              tensor.getValuesShape(),
              info.onnxType.value,
              tensor.getSparsityType().value),
          allocator.handle,
          tensor.getSparsityType(),
          info,
          indicesBuf,
          dataTuple.data);
    } else {
      throw new IllegalStateException(
          "Trying to create an OnnxSparseTensor on a closed OrtAllocator.");
    }
  }

  @Override
  public OnnxValueType getType() {
    return OnnxValueType.ONNX_TYPE_SPARSETENSOR;
  }

  /**
   * Returns the native pointer.
   *
   * @return The native pointer.
   */
  long getNativeHandle() {
    return nativeHandle;
  }

  @Override
  public SparseTensor getValue() throws OrtException {
    Buffer buffer = getDataBuffer();
    switch (sparseTensorType) {
      case COO:
        return new COOTensor(getIndexBuffer(), buffer, info.shape, info.type, buffer.remaining());
      case CSRC:
        return new CSRCTensor(getIndexBuffer(), getInnerIndexBuffer(), buffer, info.shape, info.type, buffer.remaining());
      case BLOCK_SPARSE:
        return new BlockSparseTensor(
            getIndexBuffer(), buffer, info.shape, info.type, buffer.remaining());
      case UNDEFINED:
      default:
        throw new IllegalStateException("Undefined sparsity type in this sparse tensor.");
    }
  }

  @Override
  public ValueInfo getInfo() {
    return null;
  }

  @Override
  public void close() {
    close(OnnxRuntime.ortApiHandle, nativeHandle);
  }

  /**
   * Gets a copy of the indices.
   *
   * <p>These are the outer indices if it's a CSRC sparse tensor.
   *
   * @return The indices.
   */
  public LongBuffer getIndexBuffer() {
    return getIndexBuffer(OnnxRuntime.ortApiHandle, nativeHandle).asLongBuffer();
  }

  /**
   * Gets a copy of the inner indices in a CSRC sparse tensor.
   *
   * @return The inner indices.
   */
  public LongBuffer getInnerIndexBuffer() {
    return getInnerIndexBuffer(OnnxRuntime.ortApiHandle, nativeHandle).asLongBuffer();
  }

  /**
   * Gets a copy of the data buffer.
   *
   * @return The data buffer.
   */
  public Buffer getDataBuffer() {
    ByteBuffer buffer = getDataBuffer(OnnxRuntime.ortApiHandle, nativeHandle);
    switch (info.type) {
      case FLOAT:
        return buffer.asFloatBuffer();
      case DOUBLE:
        return buffer.asDoubleBuffer();
      case INT16:
        return buffer.asShortBuffer();
      case INT32:
        return buffer.asIntBuffer();
      case INT64:
        return buffer.asLongBuffer();
      case BOOL:
      case INT8:
      case UINT8:
        return buffer;
      case STRING:
        throw new IllegalStateException("Unsupported data type String");
      case UNKNOWN:
      default:
        throw new IllegalStateException("Unsupported data type");
    }
  }

  /**
   * Wraps the indices in a direct byte buffer.
   *
   * @param apiHandle The OrtApi pointer.
   * @param nativeHandle The OrtSparseTensor pointer.
   * @return A ByteBuffer wrapping the indices.
   */
  private native ByteBuffer getIndexBuffer(long apiHandle, long nativeHandle);

  /**
   * Wraps the inner indices in a direct byte buffer.
   *
   * @param apiHandle The OrtApi pointer.
   * @param nativeHandle The OrtSparseTensor pointer.
   * @return A ByteBuffer wrapping the inner indices.
   */
  private native ByteBuffer getInnerIndexBuffer(long apiHandle, long nativeHandle);

  /**
   * Wraps the data in a direct byte buffer.
   *
   * @param apiHandle The OrtApi pointer.
   * @param nativeHandle The OrtSparseTensor pointer.
   * @return A ByteBuffer wrapping the indices.
   */
  private native ByteBuffer getDataBuffer(long apiHandle, long nativeHandle);

  /**
   * Closes the sparse tensor.
   *
   * @param apiHandle The OrtApi pointer.
   * @param nativeHandle The OrtSparseTensor pointer.
   */
  private native void close(long apiHandle, long nativeHandle);

  /**
   * Creates a sparse COO or block sparse tensor.
   * @param apiHandle The ORT API pointer.
   * @param allocatorHandle The allocator pointer.
   * @param indexData The indices.
   * @param indexBufferPos The indices position in bytes.
   * @param indexBufferSize The indices buffer size in longs.
   * @param data The data.
   * @param bufferPos The data position in bytes.
   * @param denseShape The dense shape of the tensor.
   * @param valuesShape The shape of the values (a vector for COO, and a block shape for block sparse).
   * @param onnxType The type of the values.
   * @param sparsityType The sparsity type.
   * @return A pointer to an ORT sparse tensor value.
   * @throws OrtException If the tensor could not be created.
   */
  private static native long createSparseTensorFromBuffer(
      long apiHandle,
      long allocatorHandle,
      Buffer indexData,
      int indexBufferPos,
      long indexBufferSize,
      Buffer data,
      int bufferPos,
      long[] denseShape,
      long[] valuesShape,
      int onnxType,
      int sparsityType)
      throws OrtException;

  /**
   * The type of the sparse tensor.
   *
   * <p>Should be synchronized with OrtSparseFormat in the C API.
   */
  public enum SparseTensorType {
    /** Undefined sparse tensor. */
    UNDEFINED(0),
    /** COO sparse tensor. */
    COO(1),
    /** CSR or CSC sparse tensor. */
    CSRC(2),
    /** Block sparse tensor. */
    BLOCK_SPARSE(4);

    /** The int value mirroring OrtSparseFormat. */
    public final int value;

    SparseTensorType(int value) {
      this.value = value;
    }
  }

  /**
   * Top level interfaces for the Java side representation of a sparse tensor.
   *
   * <p>Will be sealed to {@link COOTensor}, {@link CSRCTensor} and {@link BlockSparseTensor} when
   * possible.
   */
  public static interface SparseTensor {
    /**
     * Gets the dense shape of the sparse tensor.
     *
     * @return The sparse tensor shape.
     */
    public long[] getShape();

    /**
     * Gets the shape of the values of the sparse tensor.
     *
     * @return The sparse tensor value shape.
     */
    public long[] getValuesShape();

    /**
     * Gets the shape of the indices of the sparse tensor.
     *
     * @return The sparse tensor indices shape.
     */
    public long[] getIndicesShape();

    /**
     * The data type of the sparse tensor.
     *
     * @return The sparse tensor data type.
     */
    public OnnxJavaType getType();

    /**
     * The sparsity type of the sparse tensor.
     *
     * @return The sparse tensor sparsity type.
     */
    public SparseTensorType getSparsityType();

    /**
     * The number of non-zero elements.
     *
     * @return The number of non-zero elements.
     */
    public long getNumNonZeroElements();

    /**
     * Get the indices buffer.
     *
     * @return The indices buffer.
     */
    public LongBuffer getIndices();

    /**
     * Get the data buffer.
     *
     * @return The data buffer.
     */
    public Buffer getData();
  }

  private abstract static class BaseSparseTensor implements SparseTensor {
    private final long[] shape;
    private final OnnxJavaType type;
    private final long numNonZero;

    final LongBuffer indices;
    final Buffer data;

    BaseSparseTensor(
        LongBuffer indices, Buffer data, long[] shape, OnnxJavaType type, long numNonZero) {
      this.indices = indices;
      this.data = data;
      this.shape = shape;
      this.type = type;
      this.numNonZero = numNonZero;
    }

    @Override
    public long[] getShape() {
      return shape;
    }

    @Override
    public OnnxJavaType getType() {
      return type;
    }

    @Override
    public long getNumNonZeroElements() {
      return numNonZero;
    }

    @Override
    public LongBuffer getIndices() {
      return indices;
    }

    @Override
    public Buffer getData() {
      return data;
    }
  }

  /**
   * The Java side representation of a COO sparse tensor.
   */
  public static final class COOTensor extends BaseSparseTensor {

    /**
     * Creates a COO sparse tensor suitable for constructing an ORT Sparse Tensor.
     * @param indices The indices. Should be a 1d vector, or a 2d vector.
     * @param data The data.
     * @param shape The dense shape.
     * @param type The data type.
     * @param numNonZero The number of non-zero elements.
     */
    public COOTensor(
        LongBuffer indices, Buffer data, long[] shape, OnnxJavaType type, long numNonZero) {
      super(indices, data, shape, type, numNonZero);
      if (indices.remaining() != shape.length * data.remaining()) {
        throw new IllegalArgumentException(
            "Invalid number of indices, expected "
                + shape.length * data.remaining()
                + " found "
                + indices.remaining());
      }
      if (data.remaining() != numNonZero) {
        throw new IllegalArgumentException(
            "Expected data.remaining() - "
                + data.remaining()
                + " to equal numNonZero - "
                + numNonZero);
      }
    }

    @Override
    public long[] getValuesShape() {
      return new long[] {getNumNonZeroElements()};
    }

    @Override
    public long[] getIndicesShape() {
      return new long[] {getNumNonZeroElements(), getShape().length};
    }

    @Override
    public SparseTensorType getSparsityType() {
      return SparseTensorType.COO;
    }
  }

  /**
   * The Java side representation of a CSRC sparse tensor.
   */
  public static final class CSRCTensor extends BaseSparseTensor {

    private final LongBuffer innerIndices;

    /**
     * Creates a CSRC sparse tensor suitable for constructing an ORT Sparse Tensor.
     * @param outerIndices The outer indices.
     * @param innerIndices The inner indices.
     * @param data The data.
     * @param shape The dense shape.
     * @param type The data type.
     * @param numNonZero The number of non-zero elements.
     */
    public CSRCTensor(
        LongBuffer outerIndices, LongBuffer innerIndices, Buffer data, long[] shape, OnnxJavaType type, long numNonZero) {
      super(outerIndices, data, shape, type, numNonZero);
      this.innerIndices = innerIndices;
    }

    @Override
    public long[] getValuesShape() {
      return new long[] {getNumNonZeroElements()};
    }

    @Override
    public long[] getIndicesShape() {
      return new long[0];
    }

    /**
     * Gets the shape of the inner indices.
     * @return The inner indices shape.
     */
    public long[] getInnerIndicesShape() {
      return new long[0];
    }

    /**
     * Gets the inner indices buffer.
     * @return The inner indices buffer.
     */
    public LongBuffer getInnerIndices() {
      return innerIndices;
    }

    @Override
    public SparseTensorType getSparsityType() {
      return SparseTensorType.CSRC;
    }
  }

  /**
   * The Java side representation of a block sparse tensor.
   */
  public static final class BlockSparseTensor extends BaseSparseTensor {

    /**
     * Construct a block sparse tensor.
     * @param indices The indices.
     * @param data The data.
     * @param shape The dense shape.
     * @param type The data type.
     * @param numNonZero The number of non-zero elements.
     */
    public BlockSparseTensor(
        LongBuffer indices, Buffer data, long[] shape, OnnxJavaType type, long numNonZero) {
      super(indices, data, shape, type, numNonZero);
    }

    @Override
    public long[] getValuesShape() {
      return new long[0];
    }

    @Override
    public long[] getIndicesShape() {
      return new long[0];
    }

    @Override
    public SparseTensorType getSparsityType() {
      return SparseTensorType.BLOCK_SPARSE;
    }
  }
}
