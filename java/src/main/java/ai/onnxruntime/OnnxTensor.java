/*
 * Copyright (c) 2019, 2023, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import ai.onnxruntime.platform.Fp16Conversions;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.ShortBuffer;
import java.util.Optional;
import java.util.logging.Logger;

/**
 * A Java object wrapping an OnnxTensor. Tensors are the main input to the library, and can also be
 * returned as outputs.
 */
public class OnnxTensor extends OnnxTensorLike {
  private static final Logger logger = Logger.getLogger(OnnxTensor.class.getName());

  /**
   * This reference is held for OnnxTensors backed by a java.nio.Buffer to ensure the buffer does
   * not go out of scope while the OnnxTensor exists.
   */
  private final Buffer buffer;

  /**
   * Denotes if the OnnxTensor made a copy of the buffer on construction (i.e. it may have the only
   * reference).
   */
  private final boolean ownsBuffer;

  OnnxTensor(long nativeHandle, long allocatorHandle, TensorInfo info) {
    this(nativeHandle, allocatorHandle, info, null, false);
  }

  OnnxTensor(
      long nativeHandle, long allocatorHandle, TensorInfo info, Buffer buffer, boolean ownsBuffer) {
    super(nativeHandle, allocatorHandle, info);
    this.buffer = buffer;
    this.ownsBuffer = ownsBuffer;
  }

  /**
   * Returns true if the buffer in this OnnxTensor was created on construction of this tensor, i.e.,
   * it is a copy of a user supplied buffer or array and may hold the only reference to that buffer.
   *
   * <p>When this is true the backing buffer was copied from the user input, so users cannot mutate
   * the state of this buffer without first getting the reference via {@link #getBufferRef()}.
   *
   * @return True if the buffer in this OnnxTensor was allocated by it on construction (i.e., it is
   *     a copy of a user buffer or array.)
   */
  public boolean ownsBuffer() {
    return this.ownsBuffer;
  }

  /**
   * Returns a reference to the buffer which backs this {@code OnnxTensor}. If the tensor is not
   * backed by a buffer (i.e., it is backed by memory allocated by ORT) this method returns an empty
   * {@link Optional}.
   *
   * <p>Changes to the buffer elements will be reflected in the native {@code OrtValue}, this can be
   * used to repeatedly update a single tensor for multiple different inferences without allocating
   * new tensors, though the inputs <b>must</b> remain the same size and shape.
   *
   * <p>Note: the tensor could refer to a contiguous range of elements in this buffer, not the whole
   * buffer. It is up to the user to manage this information by respecting the position and limit.
   * As a consequence, accessing this reference should be considered problematic when multiple
   * threads hold references to the buffer.
   *
   * @return A reference to the buffer.
   */
  public Optional<Buffer> getBufferRef() {
    return Optional.ofNullable(duplicate(buffer));
  }

  /**
   * Duplicates the buffer to ensure concurrent reads don't disrupt the buffer position. Concurrent
   * writes will modify the underlying memory in a racy way, don't do that.
   *
   * <p>Can be replaced to a call to buf.duplicate() in Java 9+.
   *
   * @param buf The buffer to duplicate.
   * @return A copy of the buffer which refers to the same underlying memory, but has an independent
   *     position, limit and mark.
   */
  private static Buffer duplicate(Buffer buf) {
    if (buf instanceof ByteBuffer) {
      return ((ByteBuffer) buf).duplicate().order(ByteOrder.nativeOrder());
    } else if (buf instanceof ShortBuffer) {
      return ((ShortBuffer) buf).duplicate();
    } else if (buf instanceof IntBuffer) {
      return ((IntBuffer) buf).duplicate();
    } else if (buf instanceof LongBuffer) {
      return ((LongBuffer) buf).duplicate();
    } else if (buf instanceof FloatBuffer) {
      return ((FloatBuffer) buf).duplicate();
    } else if (buf instanceof DoubleBuffer) {
      return ((DoubleBuffer) buf).duplicate();
    } else {
      throw new IllegalStateException("Unknown buffer type " + buf.getClass());
    }
  }

  /**
   * Checks that the buffer is the right type for the {@code info.type}, and if it's a {@link
   * ByteBuffer} then convert it to the right type. If it's not convertible it throws {@link
   * IllegalStateException}.
   *
   * <p>Note this method converts FP16 and BFLOAT16 ShortBuffers into FP32 FloatBuffers.
   *
   * @param buf The buffer to convert.
   * @return The buffer with the expected type.
   */
  private Buffer castBuffer(Buffer buf) {
    switch (info.type) {
      case FLOAT:
        if (buf instanceof FloatBuffer) {
          return buf;
        } else if (buf instanceof ByteBuffer) {
          return ((ByteBuffer) buf).asFloatBuffer();
        }
        break;
      case DOUBLE:
        if (buf instanceof DoubleBuffer) {
          return buf;
        } else if (buf instanceof ByteBuffer) {
          return ((ByteBuffer) buf).asDoubleBuffer();
        }
        break;
      case BOOL:
      case INT8:
      case UINT8:
        if (buf instanceof ByteBuffer) {
          return buf;
        }
        break;
      case BFLOAT16:
        if (buf instanceof ShortBuffer) {
          ShortBuffer bf16Buf = (ShortBuffer) buf;
          return Fp16Conversions.convertBf16BufferToFloatBuffer(bf16Buf);
        } else if (buf instanceof ByteBuffer) {
          ShortBuffer bf16Buf = ((ByteBuffer) buf).asShortBuffer();
          return Fp16Conversions.convertBf16BufferToFloatBuffer(bf16Buf);
        }
        break;
      case FLOAT16:
        if (buf instanceof ShortBuffer) {
          ShortBuffer fp16Buf = (ShortBuffer) buf;
          return Fp16Conversions.convertFp16BufferToFloatBuffer(fp16Buf);
        } else if (buf instanceof ByteBuffer) {
          ShortBuffer fp16Buf = ((ByteBuffer) buf).asShortBuffer();
          return Fp16Conversions.convertFp16BufferToFloatBuffer(fp16Buf);
        }
        break;
      case INT16:
        if (buf instanceof ShortBuffer) {
          return buf;
        } else if (buf instanceof ByteBuffer) {
          return ((ByteBuffer) buf).asShortBuffer();
        }
        break;
      case INT32:
        if (buf instanceof IntBuffer) {
          return buf;
        } else if (buf instanceof ByteBuffer) {
          return ((ByteBuffer) buf).asIntBuffer();
        }
        break;
      case INT64:
        if (buf instanceof LongBuffer) {
          return buf;
        } else if (buf instanceof ByteBuffer) {
          return ((ByteBuffer) buf).asLongBuffer();
        }
        break;
    }
    throw new IllegalStateException(
        "Invalid buffer type for cast operation, found "
            + buf.getClass()
            + " expected something convertible to "
            + info.type);
  }

  @Override
  public OnnxValueType getType() {
    return OnnxValueType.ONNX_TYPE_TENSOR;
  }

  /**
   * Either returns a boxed primitive if the Tensor is a scalar, or a multidimensional array of
   * primitives if it has multiple dimensions.
   *
   * <p>Java multidimensional arrays are quite slow for more than 2 dimensions, in that case it is
   * recommended you use the {@link java.nio.Buffer} extractors below (e.g., {@link
   * #getFloatBuffer}).
   *
   * @return A Java value.
   * @throws OrtException If the value could not be extracted as the Tensor is invalid, or if the
   *     native code encountered an error.
   */
  @Override
  public Object getValue() throws OrtException {
    checkClosed();
    if (info.isScalar()) {
      switch (info.type) {
        case FLOAT:
          return getFloat(OnnxRuntime.ortApiHandle, nativeHandle, info.onnxType.value);
        case DOUBLE:
          return getDouble(OnnxRuntime.ortApiHandle, nativeHandle);
        case UINT8:
        case INT8:
          return getByte(OnnxRuntime.ortApiHandle, nativeHandle, info.onnxType.value);
        case INT16:
          return getShort(OnnxRuntime.ortApiHandle, nativeHandle, info.onnxType.value);
        case INT32:
          return getInt(OnnxRuntime.ortApiHandle, nativeHandle, info.onnxType.value);
        case INT64:
          return getLong(OnnxRuntime.ortApiHandle, nativeHandle, info.onnxType.value);
        case BOOL:
          return getBool(OnnxRuntime.ortApiHandle, nativeHandle);
        case STRING:
          return getString(OnnxRuntime.ortApiHandle, nativeHandle);
        case FLOAT16:
          return Fp16Conversions.fp16ToFloat(
              getShort(OnnxRuntime.ortApiHandle, nativeHandle, info.onnxType.value));
        case BFLOAT16:
          return Fp16Conversions.bf16ToFloat(
              getShort(OnnxRuntime.ortApiHandle, nativeHandle, info.onnxType.value));
        case UNKNOWN:
        default:
          throw new OrtException("Extracting the value of an invalid Tensor.");
      }
    } else {
      Object carrier = info.makeCarrier();
      if (info.getNumElements() > 0) {
        // If the tensor has values copy them out
        if (info.type == OnnxJavaType.STRING) {
          // We read the strings out from native code in a flat array and then reshape
          // to the desired output shape if necessary.
          getStringArray(OnnxRuntime.ortApiHandle, nativeHandle, (String[]) carrier);
          if (info.shape.length != 1) {
            carrier = OrtUtil.reshape((String[]) carrier, info.shape);
          }
        } else {
          // Wrap ORT owned memory in buffer, otherwise use our reference
          Buffer buf;
          if (buffer == null) {
            buf = castBuffer(getBuffer());
          } else {
            buf = castBuffer(duplicate(buffer));
          }
          // Copy out buffer into arrays
          OrtUtil.fillArrayFromBuffer(info, buf, 0, carrier);
        }
      }
      return carrier;
    }
  }

  @Override
  public String toString() {
    return "OnnxTensor(info=" + info.toString() + ",closed=" + closed + ")";
  }

  /**
   * Closes the tensor, releasing its underlying memory (if it's not backed by an NIO buffer). If it
   * is backed by a buffer then the memory is released when the buffer is GC'd.
   */
  @Override
  public synchronized void close() {
    if (!closed) {
      close(OnnxRuntime.ortApiHandle, nativeHandle);
      closed = true;
    } else {
      logger.warning("Closing an already closed tensor.");
    }
  }

  /**
   * Returns a copy of the underlying OnnxTensor as a ByteBuffer.
   *
   * <p>This method returns null if the OnnxTensor contains Strings as they are stored externally to
   * the OnnxTensor.
   *
   * @return A ByteBuffer copy of the OnnxTensor.
   */
  public ByteBuffer getByteBuffer() {
    checkClosed();
    if (info.type != OnnxJavaType.STRING) {
      ByteBuffer buffer = getBuffer();
      ByteBuffer output = ByteBuffer.allocate(buffer.capacity()).order(ByteOrder.nativeOrder());
      output.put(buffer);
      output.rewind();
      return output;
    } else {
      return null;
    }
  }

  /**
   * Returns a copy of the underlying OnnxTensor as a FloatBuffer if it can be losslessly converted
   * into a float (i.e. it's a float, fp16 or bf16), otherwise it returns null.
   *
   * @return A FloatBuffer copy of the OnnxTensor.
   */
  public FloatBuffer getFloatBuffer() {
    checkClosed();
    if (info.type == OnnxJavaType.FLOAT) {
      // if it's fp32 use the efficient copy.
      FloatBuffer buffer = getBuffer().asFloatBuffer();
      FloatBuffer output = FloatBuffer.allocate(buffer.capacity());
      output.put(buffer);
      output.rewind();
      return output;
    } else if (info.type == OnnxJavaType.FLOAT16) {
      // if it's fp16 we need to convert it.
      ByteBuffer buf = getBuffer();
      ShortBuffer buffer = buf.asShortBuffer();
      return Fp16Conversions.convertFp16BufferToFloatBuffer(buffer);
    } else if (info.type == OnnxJavaType.BFLOAT16) {
      // if it's bf16 we need to convert it.
      ByteBuffer buf = getBuffer();
      ShortBuffer buffer = buf.asShortBuffer();
      return Fp16Conversions.convertBf16BufferToFloatBuffer(buffer);
    } else {
      return null;
    }
  }

  /**
   * Returns a copy of the underlying OnnxTensor as a DoubleBuffer if the underlying type is a
   * double, otherwise it returns null.
   *
   * @return A DoubleBuffer copy of the OnnxTensor.
   */
  public DoubleBuffer getDoubleBuffer() {
    checkClosed();
    if (info.type == OnnxJavaType.DOUBLE) {
      DoubleBuffer buffer = getBuffer().asDoubleBuffer();
      DoubleBuffer output = DoubleBuffer.allocate(buffer.capacity());
      output.put(buffer);
      output.rewind();
      return output;
    } else {
      return null;
    }
  }

  /**
   * Returns a copy of the underlying OnnxTensor as a ShortBuffer if the underlying type is int16,
   * uint16, fp16 or bf16, otherwise it returns null.
   *
   * @return A ShortBuffer copy of the OnnxTensor.
   */
  public ShortBuffer getShortBuffer() {
    checkClosed();
    if ((info.type == OnnxJavaType.INT16)
        || (info.type == OnnxJavaType.FLOAT16)
        || (info.type == OnnxJavaType.BFLOAT16)) {
      ShortBuffer buffer = getBuffer().asShortBuffer();
      ShortBuffer output = ShortBuffer.allocate(buffer.capacity());
      output.put(buffer);
      output.rewind();
      return output;
    } else {
      return null;
    }
  }

  /**
   * Returns a copy of the underlying OnnxTensor as an IntBuffer if the underlying type is int32 or
   * uint32, otherwise it returns null.
   *
   * @return An IntBuffer copy of the OnnxTensor.
   */
  public IntBuffer getIntBuffer() {
    checkClosed();
    if (info.type == OnnxJavaType.INT32) {
      IntBuffer buffer = getBuffer().asIntBuffer();
      IntBuffer output = IntBuffer.allocate(buffer.capacity());
      output.put(buffer);
      output.rewind();
      return output;
    } else {
      return null;
    }
  }

  /**
   * Returns a copy of the underlying OnnxTensor as a LongBuffer if the underlying type is int64 or
   * uint64, otherwise it returns null.
   *
   * @return A LongBuffer copy of the OnnxTensor.
   */
  public LongBuffer getLongBuffer() {
    checkClosed();
    if (info.type == OnnxJavaType.INT64) {
      LongBuffer buffer = getBuffer().asLongBuffer();
      LongBuffer output = LongBuffer.allocate(buffer.capacity());
      output.put(buffer);
      output.rewind();
      return output;
    } else {
      return null;
    }
  }

  /**
   * Wraps the OrtTensor pointer in a direct byte buffer of the native platform endian-ness. Unless
   * you really know what you're doing, you want this one rather than the native call {@link
   * OnnxTensor#getBuffer(long,long)}.
   *
   * @return A ByteBuffer wrapping the data.
   */
  private ByteBuffer getBuffer() {
    return getBuffer(OnnxRuntime.ortApiHandle, nativeHandle).order(ByteOrder.nativeOrder());
  }

  /**
   * Wraps the OrtTensor pointer in a direct byte buffer.
   *
   * @param apiHandle The OrtApi pointer.
   * @param nativeHandle The OrtTensor pointer.
   * @return A ByteBuffer wrapping the data.
   */
  private native ByteBuffer getBuffer(long apiHandle, long nativeHandle);

  private native float getFloat(long apiHandle, long nativeHandle, int onnxType)
      throws OrtException;

  private native double getDouble(long apiHandle, long nativeHandle) throws OrtException;

  private native byte getByte(long apiHandle, long nativeHandle, int onnxType) throws OrtException;

  private native short getShort(long apiHandle, long nativeHandle, int onnxType)
      throws OrtException;

  private native int getInt(long apiHandle, long nativeHandle, int onnxType) throws OrtException;

  private native long getLong(long apiHandle, long nativeHandle, int onnxType) throws OrtException;

  private native String getString(long apiHandle, long nativeHandle) throws OrtException;

  private native boolean getBool(long apiHandle, long nativeHandle) throws OrtException;

  private native void getStringArray(long apiHandle, long nativeHandle, String[] carrier)
      throws OrtException;

  private native void close(long apiHandle, long nativeHandle);

  /**
   * Create a Tensor from a Java primitive, primitive multidimensional array or String
   * multidimensional array. The shape is inferred from the object using reflection. The default
   * allocator is used.
   *
   * <p>Note: Java multidimensional arrays are not dense and this method requires traversing a large
   * number of pointers for high dimensional arrays. For types other than Strings it is recommended
   * to use one of the {@code createTensor} methods which accepts a {@link java.nio.Buffer}, e.g.
   * {@link #createTensor(OrtEnvironment, FloatBuffer, long[])} as those methods are zero copy to
   * transfer data into ORT when using direct buffers.
   *
   * @param env The current OrtEnvironment.
   * @param data The data to store in a tensor.
   * @return An OnnxTensor storing the data.
   * @throws OrtException If the onnx runtime threw an error.
   */
  public static OnnxTensor createTensor(OrtEnvironment env, Object data) throws OrtException {
    return createTensor(env, env.defaultAllocator, data);
  }

  /**
   * Create a Tensor from a Java primitive, String, primitive multidimensional array or String
   * multidimensional array. The shape is inferred from the object using reflection.
   *
   * @param env The current OrtEnvironment.
   * @param allocator The allocator to use.
   * @param data The data to store in a tensor.
   * @return An OnnxTensor storing the data.
   * @throws OrtException If the onnx runtime threw an error.
   */
  static OnnxTensor createTensor(OrtEnvironment env, OrtAllocator allocator, Object data)
      throws OrtException {
    if (!allocator.isClosed()) {
      TensorInfo info = TensorInfo.constructFromJavaArray(data);
      if (info.type == OnnxJavaType.STRING) {
        if (info.shape.length == 0) {
          return new OnnxTensor(
              createString(OnnxRuntime.ortApiHandle, allocator.handle, (String) data),
              allocator.handle,
              info);
        } else {
          return new OnnxTensor(
              createStringTensor(
                  OnnxRuntime.ortApiHandle,
                  allocator.handle,
                  OrtUtil.flattenString(data),
                  info.shape),
              allocator.handle,
              info);
        }
      } else {
        Buffer buf;
        if (info.shape.length == 0) {
          buf = OrtUtil.convertBoxedPrimitiveToBuffer(info.type, data);
          if (buf == null) {
            throw new OrtException(
                "Failed to convert a boxed primitive to an array, this is an error with the ORT Java API, please report this message & stack trace. JavaType = "
                    + info.type
                    + ", object = "
                    + data);
          }
        } else {
          buf = OrtUtil.convertArrayToBuffer(info, data);
        }
        return new OnnxTensor(
            createTensorFromBuffer(
                OnnxRuntime.ortApiHandle,
                allocator.handle,
                buf,
                0,
                info.type.size * info.numElements,
                info.shape,
                info.onnxType.value),
            allocator.handle,
            info,
            buf,
            true);
      }
    } else {
      throw new IllegalStateException("Trying to create an OnnxTensor with a closed OrtAllocator.");
    }
  }

  /**
   * Create a tensor from a flattened string array.
   *
   * <p>Requires the array to be flattened in row-major order. Uses the default allocator.
   *
   * @param env The current OrtEnvironment.
   * @param data The tensor data
   * @param shape the shape of the tensor
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  public static OnnxTensor createTensor(OrtEnvironment env, String[] data, long[] shape)
      throws OrtException {
    return createTensor(env, env.defaultAllocator, data, shape);
  }

  /**
   * Create a tensor from a flattened string array.
   *
   * <p>Requires the array to be flattened in row-major order.
   *
   * @param env The current OrtEnvironment.
   * @param allocator The allocator to use.
   * @param data The tensor data
   * @param shape the shape of the tensor
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  static OnnxTensor createTensor(
      OrtEnvironment env, OrtAllocator allocator, String[] data, long[] shape) throws OrtException {
    if (!allocator.isClosed()) {
      TensorInfo info =
          new TensorInfo(
              shape,
              OnnxJavaType.STRING,
              TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
      return new OnnxTensor(
          createStringTensor(OnnxRuntime.ortApiHandle, allocator.handle, data, shape),
          allocator.handle,
          info);
    } else {
      throw new IllegalStateException("Trying to create an OnnxTensor on a closed OrtAllocator.");
    }
  }

  /**
   * Create an OnnxTensor backed by a direct FloatBuffer. The buffer should be in nativeOrder.
   *
   * <p>If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
   * of the tensor. Uses the default allocator.
   *
   * @param env The current OrtEnvironment.
   * @param data The tensor data.
   * @param shape The shape of tensor.
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  public static OnnxTensor createTensor(OrtEnvironment env, FloatBuffer data, long[] shape)
      throws OrtException {
    return createTensor(env, env.defaultAllocator, data, shape);
  }

  /**
   * Create an OnnxTensor backed by a direct FloatBuffer. The buffer should be in nativeOrder.
   *
   * <p>If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
   * of the tensor.
   *
   * @param env The current OrtEnvironment.
   * @param allocator The allocator to use.
   * @param data The tensor data.
   * @param shape The shape of tensor.
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  static OnnxTensor createTensor(
      OrtEnvironment env, OrtAllocator allocator, FloatBuffer data, long[] shape)
      throws OrtException {
    if (!allocator.isClosed()) {
      OnnxJavaType type = OnnxJavaType.FLOAT;
      return createTensor(type, allocator, data, shape);
    } else {
      throw new IllegalStateException("Trying to create an OnnxTensor on a closed OrtAllocator.");
    }
  }

  /**
   * Create an OnnxTensor backed by a direct DoubleBuffer. The buffer should be in nativeOrder.
   *
   * <p>If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
   * of the tensor. Uses the default allocator.
   *
   * @param env The current OrtEnvironment.
   * @param data The tensor data.
   * @param shape The shape of tensor.
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  public static OnnxTensor createTensor(OrtEnvironment env, DoubleBuffer data, long[] shape)
      throws OrtException {
    return createTensor(env, env.defaultAllocator, data, shape);
  }

  /**
   * Create an OnnxTensor backed by a direct DoubleBuffer. The buffer should be in nativeOrder.
   *
   * <p>If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
   * of the tensor.
   *
   * @param env The current OrtEnvironment.
   * @param allocator The allocator to use.
   * @param data The tensor data.
   * @param shape The shape of tensor.
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  static OnnxTensor createTensor(
      OrtEnvironment env, OrtAllocator allocator, DoubleBuffer data, long[] shape)
      throws OrtException {
    if (!allocator.isClosed()) {
      OnnxJavaType type = OnnxJavaType.DOUBLE;
      return createTensor(type, allocator, data, shape);
    } else {
      throw new IllegalStateException("Trying to create an OnnxTensor on a closed OrtAllocator.");
    }
  }

  /**
   * Create an OnnxTensor backed by a direct ByteBuffer. The buffer should be in nativeOrder.
   *
   * <p>If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
   * of the tensor. Uses the default allocator. Tells the runtime it's {@link OnnxJavaType#INT8}.
   *
   * @param env The current OrtEnvironment.
   * @param data The tensor data.
   * @param shape The shape of tensor.
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  public static OnnxTensor createTensor(OrtEnvironment env, ByteBuffer data, long[] shape)
      throws OrtException {
    return createTensor(env, env.defaultAllocator, data, shape);
  }

  /**
   * Create an OnnxTensor backed by a direct ByteBuffer. The buffer should be in nativeOrder.
   *
   * <p>If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
   * of the tensor. Tells the runtime it's {@link OnnxJavaType#INT8}.
   *
   * @param env The current OrtEnvironment.
   * @param allocator The allocator to use.
   * @param data The tensor data.
   * @param shape The shape of tensor.
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  static OnnxTensor createTensor(
      OrtEnvironment env, OrtAllocator allocator, ByteBuffer data, long[] shape)
      throws OrtException {
    return createTensor(env, allocator, data, shape, OnnxJavaType.INT8);
  }

  /**
   * Create an OnnxTensor backed by a direct ByteBuffer. The buffer should be in nativeOrder.
   *
   * <p>If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
   * of the tensor. Uses the default allocator. Tells the runtime it's the specified type.
   *
   * @param env The current OrtEnvironment.
   * @param data The tensor data.
   * @param shape The shape of tensor.
   * @param type The type to use for the byte buffer elements.
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  public static OnnxTensor createTensor(
      OrtEnvironment env, ByteBuffer data, long[] shape, OnnxJavaType type) throws OrtException {
    return createTensor(env, env.defaultAllocator, data, shape, type);
  }

  /**
   * Create an OnnxTensor backed by a direct ByteBuffer. The buffer should be in nativeOrder.
   *
   * <p>If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
   * of the tensor. Tells the runtime it's the specified type.
   *
   * @param env The current OrtEnvironment.
   * @param allocator The allocator to use.
   * @param data The tensor data.
   * @param shape The shape of tensor.
   * @param type The type to use for the byte buffer elements.
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  static OnnxTensor createTensor(
      OrtEnvironment env, OrtAllocator allocator, ByteBuffer data, long[] shape, OnnxJavaType type)
      throws OrtException {
    if (!allocator.isClosed()) {
      return createTensor(type, allocator, data, shape);
    } else {
      throw new IllegalStateException("Trying to create an OnnxTensor on a closed OrtAllocator.");
    }
  }

  /**
   * Create an OnnxTensor backed by a direct ShortBuffer. The buffer should be in nativeOrder.
   *
   * <p>If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
   * of the tensor. Uses the default allocator.
   *
   * @param env The current OrtEnvironment.
   * @param data The tensor data.
   * @param shape The shape of tensor.
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  public static OnnxTensor createTensor(OrtEnvironment env, ShortBuffer data, long[] shape)
      throws OrtException {
    return createTensor(env, env.defaultAllocator, data, shape, OnnxJavaType.INT16);
  }

  /**
   * Create an OnnxTensor backed by a direct ShortBuffer. The buffer should be in nativeOrder.
   *
   * <p>If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
   * of the tensor. Uses the default allocator.
   *
   * @param env The current OrtEnvironment.
   * @param data The tensor data.
   * @param shape The shape of tensor.
   * @param type The type of the data in the buffer, can be either {@link OnnxJavaType#INT16},
   *     {@link OnnxJavaType#FLOAT16} or {@link OnnxJavaType#BFLOAT16}.
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  public static OnnxTensor createTensor(
      OrtEnvironment env, ShortBuffer data, long[] shape, OnnxJavaType type) throws OrtException {
    return createTensor(env, env.defaultAllocator, data, shape, type);
  }

  /**
   * Create an OnnxTensor backed by a direct ShortBuffer. The buffer should be in nativeOrder.
   *
   * <p>If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
   * of the tensor.
   *
   * @param env The current OrtEnvironment.
   * @param allocator The allocator to use.
   * @param data The tensor data.
   * @param shape The shape of tensor.
   * @param type The type of the data in the buffer, can be either {@link OnnxJavaType#INT16},
   *     {@link OnnxJavaType#FLOAT16} or {@link OnnxJavaType#BFLOAT16}.
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  static OnnxTensor createTensor(
      OrtEnvironment env, OrtAllocator allocator, ShortBuffer data, long[] shape, OnnxJavaType type)
      throws OrtException {
    if (!allocator.isClosed()) {
      if ((type == OnnxJavaType.BFLOAT16)
          || (type == OnnxJavaType.FLOAT16)
          || (type == OnnxJavaType.INT16)) {
        return createTensor(type, allocator, data, shape);
      } else {
        throw new IllegalArgumentException(
            "Only int16, float16 or bfloat16 tensors can be created from ShortBuffer.");
      }
    } else {
      throw new IllegalStateException("Trying to create an OnnxTensor on a closed OrtAllocator.");
    }
  }

  /**
   * Create an OnnxTensor backed by a direct IntBuffer. The buffer should be in nativeOrder.
   *
   * <p>If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
   * of the tensor. Uses the default allocator.
   *
   * @param env The current OrtEnvironment.
   * @param data The tensor data.
   * @param shape The shape of tensor.
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  public static OnnxTensor createTensor(OrtEnvironment env, IntBuffer data, long[] shape)
      throws OrtException {
    return createTensor(env, env.defaultAllocator, data, shape);
  }

  /**
   * Create an OnnxTensor backed by a direct IntBuffer. The buffer should be in nativeOrder.
   *
   * <p>If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
   * of the tensor.
   *
   * @param env The current OrtEnvironment.
   * @param allocator The allocator to use.
   * @param data The tensor data.
   * @param shape The shape of tensor.
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  static OnnxTensor createTensor(
      OrtEnvironment env, OrtAllocator allocator, IntBuffer data, long[] shape)
      throws OrtException {
    if (!allocator.isClosed()) {
      OnnxJavaType type = OnnxJavaType.INT32;
      return createTensor(type, allocator, data, shape);
    } else {
      throw new IllegalStateException("Trying to create an OnnxTensor on a closed OrtAllocator.");
    }
  }

  /**
   * Create an OnnxTensor backed by a direct LongBuffer. The buffer should be in nativeOrder.
   *
   * <p>If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
   * of the tensor. Uses the default allocator.
   *
   * @param env The current OrtEnvironment.
   * @param data The tensor data.
   * @param shape The shape of tensor.
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  public static OnnxTensor createTensor(OrtEnvironment env, LongBuffer data, long[] shape)
      throws OrtException {
    return createTensor(env, env.defaultAllocator, data, shape);
  }

  /**
   * Create an OnnxTensor backed by a direct LongBuffer. The buffer should be in nativeOrder.
   *
   * <p>If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
   * of the tensor. Uses the supplied allocator.
   *
   * @param env The current OrtEnvironment.
   * @param allocator The allocator to use.
   * @param data The tensor data.
   * @param shape The shape of tensor.
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  static OnnxTensor createTensor(
      OrtEnvironment env, OrtAllocator allocator, LongBuffer data, long[] shape)
      throws OrtException {
    if (!allocator.isClosed()) {
      OnnxJavaType type = OnnxJavaType.INT64;
      return createTensor(type, allocator, data, shape);
    } else {
      throw new IllegalStateException("Trying to create an OnnxTensor on a closed OrtAllocator.");
    }
  }

  /**
   * Creates a tensor by wrapping the data in a direct byte buffer and passing it to JNI.
   *
   * <p>Throws IllegalStateException if the buffer is too large to create a direct byte buffer copy,
   * which is more than approximately (Integer.MAX_VALUE - 5) / type.size elements.
   *
   * @param type The buffer type.
   * @param allocator The OrtAllocator.
   * @param data The data.
   * @param shape The tensor shape.
   * @return An OnnxTensor instance.
   * @throws OrtException If the create call failed.
   */
  private static OnnxTensor createTensor(
      OnnxJavaType type, OrtAllocator allocator, Buffer data, long[] shape) throws OrtException {
    OrtUtil.BufferTuple tuple = OrtUtil.prepareBuffer(data, type);
    TensorInfo info = TensorInfo.constructFromBuffer(tuple.data, shape, type);
    return new OnnxTensor(
        createTensorFromBuffer(
            OnnxRuntime.ortApiHandle,
            allocator.handle,
            tuple.data,
            tuple.pos,
            tuple.byteSize,
            shape,
            info.onnxType.value),
        allocator.handle,
        info,
        tuple.data,
        tuple.isCopy);
  }

  private static native long createTensorFromBuffer(
      long apiHandle,
      long allocatorHandle,
      Buffer data,
      int bufferPos,
      long bufferSize,
      long[] shape,
      int onnxType)
      throws OrtException;

  private static native long createString(long apiHandle, long allocatorHandle, String data)
      throws OrtException;

  private static native long createStringTensor(
      long apiHandle, long allocatorHandle, Object[] data, long[] shape) throws OrtException;
}
