/*
 * Copyright (c) 2019, 2023, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.reflect.Array;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.ShortBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;

/** Util code for interacting with Java arrays. */
public final class OrtUtil {
  private static final Logger logger = Logger.getLogger(OrtUtil.class.getName());

  private static final MethodHandle fp16ToFp32;
  private static final MethodHandle fp32ToFp16;

  static {
    MethodHandle tmp16 = null;
    MethodHandle tmp32 = null;
    MethodHandles.Lookup lookup = MethodHandles.lookup();
    try {
      // Attempt to lookup the Java 20 fp16 conversion methods which can use SIMD intrinsics.
      tmp16 =
          lookup.findStatic(
              Float.class, "float16ToFloat", MethodType.methodType(float.class, short.class));
      tmp32 =
          lookup.findStatic(
              Float.class, "floatToFloat16", MethodType.methodType(short.class, float.class));
    } catch (IllegalAccessException | NoSuchMethodException e) {
      // Must be on Java 19 or earlier, create handles for our methods.
      try {
        tmp16 =
            lookup.findStatic(
                OrtUtil.class, "mlasFp16ToFloat", MethodType.methodType(float.class, short.class));
        tmp32 =
            lookup.findStatic(
                OrtUtil.class, "mlasFloatToFp16", MethodType.methodType(short.class, float.class));
      } catch (IllegalAccessException | NoSuchMethodException ex) {
        // Should not happen
        logger.log(Level.SEVERE, "Failed to find fp16 conversion methods on OnnxTensor", e);
      }
    }
    fp16ToFp32 = tmp16;
    fp32ToFp16 = tmp32;
  }

  /** Private constructor for static util class. */
  private OrtUtil() {}

  /**
   * Converts an long shape into a int shape.
   *
   * <p>Validates that the shape has more than 1 elements, less than 9 elements, each element is
   * less than {@link Integer#MAX_VALUE} and that each entry is non-negative.
   *
   * @param shape The long shape.
   * @return The int shape.
   */
  public static int[] transformShape(long[] shape) {
    if (shape.length == 0 || shape.length > TensorInfo.MAX_DIMENSIONS) {
      throw new IllegalArgumentException(
          "Arrays with less than 1 and greater than "
              + TensorInfo.MAX_DIMENSIONS
              + " dimensions are not supported.");
    }
    int[] newShape = new int[shape.length];
    for (int i = 0; i < shape.length; i++) {
      long curDim = shape[i];
      if (curDim < 0 || curDim > Integer.MAX_VALUE) {
        throw new IllegalArgumentException(
            "Invalid shape for a Java array, expected non-negative entries smaller than Integer.MAX_VALUE. Found "
                + Arrays.toString(shape));
      } else {
        newShape[i] = (int) curDim;
      }
    }
    return newShape;
  }

  /**
   * Converts an int shape into a long shape.
   *
   * <p>Validates that the shape has more than 1 element, less than 9 elements and that each entry
   * is non-negative.
   *
   * @param shape The int shape.
   * @return The long shape.
   */
  public static long[] transformShape(int[] shape) {
    if (shape.length == 0 || shape.length > 8) {
      throw new IllegalArgumentException(
          "Arrays with less than 1 and greater than "
              + TensorInfo.MAX_DIMENSIONS
              + " dimensions are not supported.");
    }
    long[] newShape = new long[shape.length];
    for (int i = 0; i < shape.length; i++) {
      long curDim = shape[i];
      if (curDim < 1) {
        throw new IllegalArgumentException(
            "Invalid shape for a Java array, expected positive entries smaller than Integer.MAX_VALUE. Found "
                + Arrays.toString(shape));
      } else {
        newShape[i] = curDim;
      }
    }
    return newShape;
  }

  /**
   * Creates a new primitive boolean array of up to 8 dimensions, using the supplied shape.
   *
   * <p>
   *
   * @param shape The shape of array to create.
   * @return A boolean array.
   */
  public static Object newBooleanArray(long[] shape) {
    int[] intShape = transformShape(shape);
    return Array.newInstance(boolean.class, intShape);
  }

  /**
   * Creates a new primitive byte array of up to 8 dimensions, using the supplied shape.
   *
   * <p>
   *
   * @param shape The shape of array to create.
   * @return A byte array.
   */
  public static Object newByteArray(long[] shape) {
    int[] intShape = transformShape(shape);
    return Array.newInstance(byte.class, intShape);
  }

  /**
   * Creates a new primitive short array of up to 8 dimensions, using the supplied shape.
   *
   * <p>
   *
   * @param shape The shape of array to create.
   * @return A short array.
   */
  public static Object newShortArray(long[] shape) {
    int[] intShape = transformShape(shape);
    return Array.newInstance(short.class, intShape);
  }

  /**
   * Creates a new primitive int array of up to 8 dimensions, using the supplied shape.
   *
   * <p>
   *
   * @param shape The shape of array to create.
   * @return A int array.
   */
  public static Object newIntArray(long[] shape) {
    int[] intShape = transformShape(shape);
    return Array.newInstance(int.class, intShape);
  }

  /**
   * Creates a new primitive long array of up to 8 dimensions, using the supplied shape.
   *
   * <p>
   *
   * @param shape The shape of array to create.
   * @return A long array.
   */
  public static Object newLongArray(long[] shape) {
    int[] intShape = transformShape(shape);
    return Array.newInstance(long.class, intShape);
  }

  /**
   * Creates a new primitive float array of up to 8 dimensions, using the supplied shape.
   *
   * <p>
   *
   * @param shape The shape of array to create.
   * @return A float array.
   */
  public static Object newFloatArray(long[] shape) {
    int[] intShape = transformShape(shape);
    return Array.newInstance(float.class, intShape);
  }

  /**
   * Creates a new primitive double array of up to 8 dimensions, using the supplied shape.
   *
   * <p>
   *
   * @param shape The shape of array to create.
   * @return A double array.
   */
  public static Object newDoubleArray(long[] shape) {
    int[] intShape = transformShape(shape);
    return Array.newInstance(double.class, intShape);
  }

  /**
   * Creates a new String array of up to 8 dimensions, using the supplied shape.
   *
   * <p>
   *
   * @param shape The shape of array to create.
   * @return A double array.
   */
  public static Object newStringArray(long[] shape) {
    int[] intShape = transformShape(shape);
    return Array.newInstance(String.class, intShape);
  }

  /**
   * Reshapes a boolean array into the desired n-dimensional array assuming the boolean array is
   * stored in n-dimensional row-major order. Throws {@link IllegalArgumentException} if the number
   * of elements doesn't match between the shape and the input or the shape is invalid.
   *
   * @param input The boolean array.
   * @param shape The desired shape.
   * @return An n-dimensional boolean array.
   */
  public static Object reshape(boolean[] input, long[] shape) {
    Object output = OrtUtil.newBooleanArray(shape);
    reshape(input, output, 0);
    return output;
  }

  /**
   * Reshapes a byte array into the desired n-dimensional array assuming the byte array is stored in
   * n-dimensional row-major order. Throws {@link IllegalArgumentException} if the number of
   * elements doesn't match between the shape and the input or the shape is invalid.
   *
   * @param input The byte array.
   * @param shape The desired shape.
   * @return An n-dimensional byte array.
   */
  public static Object reshape(byte[] input, long[] shape) {
    Object output = OrtUtil.newByteArray(shape);
    reshape(input, output, 0);
    return output;
  }

  /**
   * Reshapes a short array into the desired n-dimensional array assuming the short array is stored
   * in n-dimensional row-major order. Throws {@link IllegalArgumentException} if the number of
   * elements doesn't match between the shape and the input or the shape is invalid.
   *
   * @param input The short array.
   * @param shape The desired shape.
   * @return An n-dimensional short array.
   */
  public static Object reshape(short[] input, long[] shape) {
    Object output = OrtUtil.newShortArray(shape);
    reshape(input, output, 0);
    return output;
  }

  /**
   * Reshapes an int array into the desired n-dimensional array, assuming the int array is stored in
   * n-dimensional row-major order. Throws {@link IllegalArgumentException} if the number of
   * elements doesn't match between the shape and the input or the shape is invalid.
   *
   * @param input The int array.
   * @param shape The desired shape.
   * @return An n-dimensional int array.
   */
  public static Object reshape(int[] input, long[] shape) {
    Object output = OrtUtil.newIntArray(shape);
    reshape(input, output, 0);
    return output;
  }

  /**
   * Reshapes a long array into the desired n-dimensional array, assuming the long array is stored
   * in n-dimensional row-major order. Throws {@link IllegalArgumentException} if the number of
   * elements doesn't match between the shape and the input or the shape is invalid.
   *
   * @param input The long array.
   * @param shape The desired shape.
   * @return An n-dimensional long array.
   */
  public static Object reshape(long[] input, long[] shape) {
    Object output = OrtUtil.newLongArray(shape);
    reshape(input, output, 0);
    return output;
  }

  /**
   * Reshapes a float array into the desired n-dimensional array assuming the float array is stored
   * in n-dimensional row-major order. Throws {@link IllegalArgumentException} if the number of
   * elements doesn't match between the shape and the input or the shape is invalid.
   *
   * @param input The float array.
   * @param shape The desired shape.
   * @return An n-dimensional float array.
   */
  public static Object reshape(float[] input, long[] shape) {
    Object output = OrtUtil.newFloatArray(shape);
    reshape(input, output, 0);
    return output;
  }

  /**
   * Reshapes a double array into the desired n-dimensional array assuming the double array is
   * stored in n-dimensional row-major order. Throws {@link IllegalArgumentException} if the number
   * of elements doesn't match between the shape and the input or the shape is invalid.
   *
   * @param input The double array.
   * @param shape The desired shape.
   * @return An n-dimensional double array.
   */
  public static Object reshape(double[] input, long[] shape) {
    Object output = OrtUtil.newDoubleArray(shape);
    reshape(input, output, 0);
    return output;
  }

  /**
   * Reshapes a String array into the desired n-dimensional array assuming the String array is
   * stored in n-dimensional row-major order. Throws {@link IllegalArgumentException} if the number
   * of elements doesn't match between the shape and the input or the shape is invalid.
   *
   * @param input The double array.
   * @param shape The desired shape.
   * @return An n-dimensional String array.
   */
  public static Object reshape(String[] input, long[] shape) {
    Object output = OrtUtil.newStringArray(shape);
    reshape(input, output, 0);
    return output;
  }

  /**
   * Copies elements from the flat input array to the appropriate primitive array of the output.
   * Recursively calls itself as it traverses the output array.
   *
   * @param input The input array.
   * @param output The output multidimensional array.
   * @param position The current position in the input array.
   * @return The new position in the input array.
   */
  private static int reshape(Object input, Object output, int position) {
    if (output.getClass().isArray()) {
      Object[] outputArray = (Object[]) output;
      for (Object outputElement : outputArray) {
        Class<?> outputElementClass = outputElement.getClass();
        if (outputElementClass.isArray()) {
          Class<?> componentType = outputElementClass.getComponentType();
          if (componentType.isPrimitive() || componentType == String.class) {
            int length = Array.getLength(outputElement);
            System.arraycopy(input, position, outputElement, 0, length);
            position += length;
          } else {
            position = reshape(input, outputElement, position);
          }
        } else {
          throw new IllegalStateException(
              "Found element type when expecting an array. Class " + outputElementClass);
        }
      }
    } else {
      throw new IllegalStateException(
          "Found element type when expecting an array. Class " + output.getClass());
    }

    return position;
  }

  /**
   * Counts the number of elements stored in a Tensor of this shape.
   *
   * <p>Multiplies all the elements together if they are non-negative, throws an {@link
   * IllegalArgumentException} otherwise.
   *
   * @param shape The shape to use.
   * @return The number of elements.
   */
  public static long elementCount(long[] shape) {
    // Java side tensors must be less than Integer.MAX_VALUE,
    // tensors created in native code can be larger, but are not usable in Java.
    // Tensors should not be able to be created which will overflow a 64-bit long.
    long count = 1;
    for (int i = 0; i < shape.length; i++) {
      if (shape[i] >= 0) {
        count *= shape[i];
      } else {
        throw new IllegalArgumentException(
            "Received negative value in shape " + Arrays.toString(shape) + " .");
      }
    }
    return count;
  }

  /**
   * Checks that the shape is a valid shape for a Java array (i.e. that the values are all positive
   * and representable by an int).
   *
   * @param shape The shape to check.
   * @return True if the shape is valid.
   */
  public static boolean validateShape(long[] shape) {
    boolean valid = true;
    for (int i = 0; i < shape.length; i++) {
      valid &= shape[i] > 0;
      valid &= ((int) shape[i]) == shape[i];
    }
    return valid && shape.length <= TensorInfo.MAX_DIMENSIONS;
  }

  /**
   * Flatten a multidimensional String array into a single dimensional String array, reading it in a
   * multidimensional row-major order.
   *
   * @param o A multidimensional String array.
   * @return A single dimensional String array.
   */
  public static String[] flattenString(Object o) {
    if (o instanceof String[]) {
      return (String[]) o;
    } else {
      ArrayList<String> output = new ArrayList<>();

      flattenString((Object[]) o, output);

      return output.toArray(new String[0]);
    }
  }

  /**
   * Flattens a multidimensional String array into the ArrayList.
   *
   * @param input The multidimensional String array.
   * @param output The output ArrayList.
   */
  private static void flattenString(Object[] input, ArrayList<String> output) {
    for (Object i : input) {
      Class<?> iClazz = i.getClass();
      if (iClazz.isArray()) {
        if (iClazz.getComponentType().isArray()) {
          flattenString((Object[]) i, output);
        } else if (iClazz.getComponentType().equals(String.class)) {
          output.addAll(Arrays.asList((String[]) i));
        } else {
          throw new IllegalStateException("Found a non-String, non-array element type, " + iClazz);
        }
      } else {
        throw new IllegalStateException(
            "Found an element type where there should have been an array. Class = " + iClazz);
      }
    }
  }

  /**
   * Stores a boxed primitive in a single element array of the unboxed type.
   *
   * <p>If it's not a boxed primitive then it returns null.
   *
   * @param javaType The type of the boxed primitive.
   * @param data The boxed primitive.
   * @return The primitive in an array.
   */
  static Object convertBoxedPrimitiveToArray(OnnxJavaType javaType, Object data) {
    switch (javaType) {
      case FLOAT:
        float[] floatArr = new float[1];
        floatArr[0] = (Float) data;
        return floatArr;
      case DOUBLE:
        double[] doubleArr = new double[1];
        doubleArr[0] = (Double) data;
        return doubleArr;
      case UINT8:
      case INT8:
        byte[] byteArr = new byte[1];
        byteArr[0] = (Byte) data;
        return byteArr;
      case INT16:
        short[] shortArr = new short[1];
        shortArr[0] = (Short) data;
        return shortArr;
      case INT32:
        int[] intArr = new int[1];
        intArr[0] = (Integer) data;
        return intArr;
      case INT64:
        long[] longArr = new long[1];
        longArr[0] = (Long) data;
        return longArr;
      case BOOL:
        boolean[] booleanArr = new boolean[1];
        booleanArr[0] = (Boolean) data;
        return booleanArr;
      case STRING:
      case UNKNOWN:
      default:
        return null;
    }
  }

  /**
   * Returns expected JDK map capacity for a given size, this factors in the default JDK load factor
   *
   * @param size The expected map size
   * @return The capacity for a map that guarantees no resizing
   */
  static int capacityFromSize(int size) {
    // 0.75 is the default JDK load factor
    return (int) (size / 0.75 + 1);
  }

  /**
   * Prepares a buffer, either copying it if it's not direct, or computing it's size and position if
   * it is.
   *
   * @param data The buffer to prepare.
   * @param type The Java-side type.
   * @return The prepared buffer tuple.
   */
  static BufferTuple prepareBuffer(Buffer data, OnnxJavaType type) {
    if (type == OnnxJavaType.STRING || type == OnnxJavaType.UNKNOWN) {
      throw new IllegalStateException("Cannot create a " + type + " tensor from a buffer");
    }
    int bufferPos;
    long bufferSizeLong = data.remaining() * (long) type.size;
    if (bufferSizeLong > (Integer.MAX_VALUE - (8 * type.size))) {
      // The maximum direct byte buffer size is a little below Integer.MAX_VALUE depending
      // on the JVM, so we check for something 8 elements below the maximum size which
      // should be allocatable (assuming there is enough memory) on all 64-bit JVMs.
      throw new IllegalStateException(
          "Cannot allocate a direct buffer of the requested size and type, size "
              + data.remaining()
              + ", type = "
              + type);
    }
    // Now we know we're in range
    int bufferSize = data.remaining() * type.size;
    Buffer tmp;
    if (data.isDirect()) {
      tmp = data;
      bufferPos = data.position() * type.size;
    } else {
      // Copy the data to a new direct buffer, then restore the state of the input.
      int origPosition = data.position();
      ByteBuffer buffer = ByteBuffer.allocateDirect(bufferSize).order(ByteOrder.nativeOrder());
      switch (type) {
        case FLOAT:
          tmp = buffer.asFloatBuffer().put((FloatBuffer) data);
          break;
        case DOUBLE:
          tmp = buffer.asDoubleBuffer().put((DoubleBuffer) data);
          break;
        case BOOL:
        case UINT8:
        case INT8:
          // buffer is already a ByteBuffer, no cast needed.
          tmp = buffer.put((ByteBuffer) data);
          break;
        case INT16:
        case FLOAT16:
        case BFLOAT16:
          tmp = buffer.asShortBuffer().put((ShortBuffer) data);
          break;
        case INT32:
          tmp = buffer.asIntBuffer().put((IntBuffer) data);
          break;
        case INT64:
          tmp = buffer.asLongBuffer().put((LongBuffer) data);
          break;
        default:
          throw new IllegalStateException(
              "Impossible to reach here, managed to cast a buffer as an incorrect type, found "
                  + type);
      }
      data.position(origPosition);
      tmp.rewind();
      bufferPos = 0;
    }

    return new BufferTuple(tmp, bufferPos, bufferSize, data.remaining(), tmp != data);
  }

  /**
   * Rounds a buffer of floats into a buffer containing fp16 values (stored as shorts in Java).
   *
   * <p>Respects the position and limit of the input buffer.
   *
   * @param buf The buffer of floats.
   * @return A buffer of fp16 values stored as shorts.
   */
  public static ShortBuffer convertFloatBufferToFp16Buffer(FloatBuffer buf) {
    int pos = buf.position();
    int remaining = buf.remaining();
    ShortBuffer output =
        ByteBuffer.allocateDirect(remaining * 2).order(ByteOrder.nativeOrder()).asShortBuffer();
    for (int i = 0; i < remaining; i++) {
      output.put(i, floatToFp16(buf.get(i + pos)));
    }
    return output;
  }

  /**
   * Casts a buffer of fp16 values stored as shorts into a buffer of floats.
   *
   * <p>Respects the position and limit of the input buffer.
   *
   * @param buf The buffer of fp16 values stored as shorts.
   * @return A buffer of float values.
   */
  public static FloatBuffer convertFp16BufferToFloatBuffer(ShortBuffer buf) {
    int pos = buf.position();
    int remaining = buf.remaining();
    FloatBuffer output =
        ByteBuffer.allocateDirect(remaining * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
    for (int i = 0; i < remaining; i++) {
      output.put(i, fp16ToFloat(buf.get(i + pos)));
    }
    return output;
  }

  /**
   * Rounds a buffer of floats into a buffer containing bf16 values (stored as shorts in Java).
   *
   * <p>Respects the position and limit of the input buffer.
   *
   * @param buf The buffer of floats.
   * @return A buffer of bf16 values stored as shorts.
   */
  public static ShortBuffer convertFloatBufferToBf16Buffer(FloatBuffer buf) {
    int pos = buf.position();
    int remaining = buf.remaining();
    ShortBuffer output =
        ByteBuffer.allocateDirect(remaining * 2).order(ByteOrder.nativeOrder()).asShortBuffer();
    for (int i = 0; i < remaining; i++) {
      output.put(i, floatToBf16(buf.get(i + pos)));
    }
    return output;
  }

  /**
   * Casts a buffer of bf16 values stored as shorts into a buffer of floats.
   *
   * <p>Respects the position and limit of the input buffer.
   *
   * @param buf The buffer of bf16 values stored as shorts.
   * @return A buffer of float values.
   */
  public static FloatBuffer convertBf16BufferToFloatBuffer(ShortBuffer buf) {
    int pos = buf.position();
    int remaining = buf.remaining();
    FloatBuffer output =
        ByteBuffer.allocateDirect(remaining * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
    for (int i = 0; i < remaining; i++) {
      output.put(i, bf16ToFloat(buf.get(i + pos)));
    }
    return output;
  }

  /**
   * Converts a fp16 value stored in a short into a float value.
   *
   * <p>Note on Java 20 or newer this uses {@code Float.float16ToFloat} which may use CPU specific
   * instructions for the conversion, otherwise it uses the conversion operation from ORT's native
   * implementation.
   *
   * @param input The fp16 value.
   * @return The float value.
   */
  public static float fp16ToFloat(short input) {
    try {
      float ret = (float) fp16ToFp32.invokeExact(input);
      return ret;
    } catch (Throwable e) {
      throw new AssertionError("Should not reach here", e);
    }
  }

  /**
   * Converts a float value into a fp16 value stored in a short.
   *
   * <p>Note on Java 20 or newer this uses {@code Float.floatToFloat16} which may use CPU specific
   * instructions for the conversion, otherwise it uses the conversion operation from ORT's native
   * implementation.
   *
   * @param input The float value.
   * @return The fp16 value.
   */
  public static short floatToFp16(float input) {
    try {
      short ret = (short) fp32ToFp16.invokeExact(input);
      return ret;
    } catch (Throwable e) {
      throw new AssertionError("Should not reach here", e);
    }
  }

  /**
   * Upcasts a fp16 value to a float. Mirrors the conversion in MLAS.
   *
   * @param input A uint16_t representing an IEEE half precision float.
   * @return A float.
   */
  static float mlasFp16ToFloat(short input) {
    // Port of MLAS_Half2Float from onnxruntime/core/mlas/inc/mlas_float16.h
    final int MAGIC = 113 << 23;
    // exponent mask after shift
    final int SHIFTED_EXP = 0x7c00 << 13;

    // exponent/mantissa bits
    int bits = (input & 0x7fff) << 13;
    // just the exponent
    final int exp = SHIFTED_EXP & bits;
    // exponent adjust
    bits += (127 - 15) << 23;

    // handle exponent special cases
    if (exp == SHIFTED_EXP) {
      // Inf/NaN?
      // extra exp adjust
      bits += (128 - 16) << 23;
    } else if (exp == 0) {
      // Zero/Denormal?
      // extra exp adjust
      bits += (1 << 23);
      // renormalize
      float tmp = Float.intBitsToFloat(bits) - Float.intBitsToFloat(MAGIC);
      bits = Float.floatToIntBits(tmp);
    }

    // sign bit
    bits |= (input & 0x8000) << 16;

    return Float.intBitsToFloat(bits);
  }

  /**
   * Rounds a float value to fp16. Mirrors the conversion in MLAS.
   *
   * @param input A float value.
   * @return The value rounded to an IEEE half precision value.
   */
  static short mlasFloatToFp16(float input) {
    // Port of MLAS_Float2Half from onnxruntime/core/mlas/inc/mlas_float16.h
    int bits = Float.floatToIntBits(input);
    final int F32_INFINITY = Float.floatToIntBits(Float.POSITIVE_INFINITY);
    final int F16_MAX = (127 + 16) << 23;
    final int DENORM_MAGIC = ((127 - 15) + (23 - 10) + 1) << 23;
    final int SIGN_MASK = 0x80000000;
    final int ROUNDING_CONST = ((15 - 127) << 23) + 0xfff;

    int sign = bits & SIGN_MASK;
    // mask out sign bit
    bits ^= sign;

    short output;
    if (bits >= F16_MAX) {
      // Inf or NaN (all exponent bits set)
      output = (bits > F32_INFINITY) ? (short) 0x7e00 : (short) 0x7c00;
    } else {
      if (bits < (113 << 23)) {
        // Subnormal or zero
        // use a magic value to align our 10 mantissa bits at the bottom of
        // the float. as long as FP addition is round-to-nearest-even this
        // just works.
        float tmp = Float.intBitsToFloat(bits) + Float.intBitsToFloat(DENORM_MAGIC);

        // and one integer subtract of the bias later, we have our final float!
        output = (short) (Float.floatToIntBits(tmp) - DENORM_MAGIC);
      } else {
        int mant_odd = (bits >> 13) & 1; // resulting mantissa is odd

        // update exponent, rounding bias part 1
        bits += ROUNDING_CONST;
        // rounding bias part 2
        bits += mant_odd;
        // take the bits!
        output = (short) (bits >> 13);
      }
    }

    // Add the sign back in
    output = (short) (output | ((short) (sign >> 16)));

    return output;
  }

  /**
   * Converts a bf16 value stored in a short into a float value.
   *
   * @param input A uint16_t representing a bfloat16 value.
   * @return A float.
   */
  public static float bf16ToFloat(short input) {
    int bits = input << 16;
    return Float.intBitsToFloat(bits);
  }

  /**
   * Converts a float into bf16. May not produce correct values for subnormal floats.
   *
   * <p>Rounds to nearest even.
   *
   * @param input The float input.
   * @return A bfloat16 value which is closest to the float.
   */
  public static short floatToBf16(float input) {
    int bits = Float.floatToIntBits(input);
    int lsb = (bits >> 16) & 1;
    int roundingBias = 0x7fff + lsb;
    bits += roundingBias;
    return (short) (bits >> 16);
  }

  static final class BufferTuple {
    final Buffer data;
    final int pos;
    final long byteSize;
    final long size;
    final boolean isCopy;

    BufferTuple(Buffer data, int pos, long byteSize, long size, boolean isCopy) {
      this.data = data;
      this.pos = pos;
      this.byteSize = byteSize;
      this.size = size;
      this.isCopy = isCopy;
    }
  }
}
