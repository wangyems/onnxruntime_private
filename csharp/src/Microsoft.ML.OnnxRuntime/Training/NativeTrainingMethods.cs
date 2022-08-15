﻿// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntime
    {
        // OrtTrainingApi  (onnxruntime_training_c_api.cc)
        [StructLayout(LayoutKind.Sequential)]
        public struct OrtTrainingApi
        {
            public IntPtr LoadCheckpoint;
            public IntPtr SaveCheckpoint;
            public IntPtr CreateTrainingSession;
            public IntPtr TrainingSessionGetTrainModelOutputCount;
            public IntPtr TrainingSessionGetEvalModelOutputCount;
            public IntPtr TrainingSessionGetTrainModelOutputName;
            public IntPtr TrainingSessionGetEvalModelOutputName;
            public IntPtr ResetGrad;
            public IntPtr TrainStep;
            public IntPtr EvalStep;
            public IntPtr OptimizerStep;
            public IntPtr ReleaseTrainingSession;
            public IntPtr ReleaseCheckpointState;
        }

        internal static class NativeTrainingMethods
        {
            static OrtApi api_;
            static OrtTrainingApi trainingApi_;
            static IntPtr trainingApiPtr;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate ref OrtApi DOrtGetApi(UInt32 version);

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /* OrtTrainingApi* */ DOrtGetTrainingApi(UInt32 version);
            public static DOrtGetTrainingApi OrtGetTrainingApi;

        static NativeTrainingMethods()
            {
                DOrtGetApi OrtGetApi = (DOrtGetApi)Marshal.GetDelegateForFunctionPointer(NativeMethods.OrtGetApiBase().GetApi, typeof(DOrtGetApi));

                // TODO: Make this save the pointer, and not copy the whole structure across
                api_ = (OrtApi)OrtGetApi(4 /*ORT_API_VERSION*/);
 
                OrtGetTrainingApi = (DOrtGetTrainingApi)Marshal.GetDelegateForFunctionPointer(api_.GetTrainingApi, typeof(DOrtGetTrainingApi));
                trainingApiPtr = OrtGetTrainingApi(4 /*ORT_API_VERSION*/);
                if (trainingApiPtr != IntPtr.Zero)
                {
                    trainingApi_ = (OrtTrainingApi)Marshal.PtrToStructure(trainingApiPtr, typeof(OrtTrainingApi));
                    OrtLoadCheckpoint = (DOrtLoadCheckpoint)Marshal.GetDelegateForFunctionPointer(trainingApi_.LoadCheckpoint, typeof(DOrtLoadCheckpoint));
                    OrtSaveCheckpoint = (DOrtSaveCheckpoint)Marshal.GetDelegateForFunctionPointer(trainingApi_.SaveCheckpoint, typeof(DOrtSaveCheckpoint));
                    OrtCreateTrainingSession = (DOrtCreateTrainingSession)Marshal.GetDelegateForFunctionPointer(trainingApi_.CreateTrainingSession, typeof(DOrtCreateTrainingSession));
                    OrtGetTrainModelOutputCount = (DOrtGetTrainModelOutputCount)Marshal.GetDelegateForFunctionPointer(trainingApi_.TrainingSessionGetTrainModelOutputCount, typeof(DOrtGetTrainModelOutputCount));
                    OrtGetEvalModelOutputCount = (DOrtGetEvalModelOutputCount)Marshal.GetDelegateForFunctionPointer(trainingApi_.TrainingSessionGetEvalModelOutputCount, typeof(DOrtGetEvalModelOutputCount));
                    OrtGetTrainModelOutputName = (DOrtGetTrainModelOutputName)Marshal.GetDelegateForFunctionPointer(trainingApi_.TrainingSessionGetTrainModelOutputName, typeof(DOrtGetTrainModelOutputName));
                    OrtGetEvalModelOutputName = (DOrtGetEvalModelOutputName)Marshal.GetDelegateForFunctionPointer(trainingApi_.TrainingSessionGetEvalModelOutputName, typeof(DOrtGetEvalModelOutputName));
                    OrtResetGrad = (DOrtResetGrad)Marshal.GetDelegateForFunctionPointer(trainingApi_.ResetGrad, typeof(DOrtResetGrad));
                    OrtTrainStep = (DOrtTrainStep)Marshal.GetDelegateForFunctionPointer(trainingApi_.TrainStep, typeof(DOrtTrainStep));
                    OrtEvalStep = (DOrtEvalStep)Marshal.GetDelegateForFunctionPointer(trainingApi_.EvalStep, typeof(DOrtEvalStep));
                    OrtOptimizerStep = (DOrtOptimizerStep)Marshal.GetDelegateForFunctionPointer(trainingApi_.OptimizerStep, typeof(DOrtOptimizerStep));
                    OrtReleaseTrainingSession = (DOrtReleaseTrainingSession)Marshal.GetDelegateForFunctionPointer(trainingApi_.ReleaseTrainingSession, typeof(DOrtReleaseTrainingSession));
                    OrtReleaseCheckpointState = (DOrtReleaseCheckpointState)Marshal.GetDelegateForFunctionPointer(trainingApi_.ReleaseCheckpointState, typeof(DOrtReleaseCheckpointState));
                }

            }

            #region TrainingSession API

            /// <summary>
            /// Creates an instance of OrtSession with provided parameters
            /// </summary>
            /// <param name="checkpointPath">UTF-8 bytes corresponding to checkpoint string path</param>
            /// <param name="checkpointState">(Output) Loaded OrtCheckpointState instance</param>
            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /* OrtStatus* */DOrtLoadCheckpoint(
                                            byte[] checkpointPath,
                                            out IntPtr /* (OrtCheckpointState**) */ checkpointState);

            public static DOrtLoadCheckpoint OrtLoadCheckpoint;

            /// <summary>
            /// Creates an instance of OrtSession with provided parameters
            /// </summary>
            /// <param name="checkpointPath">UTF-8 bytes corresponding to checkpoint string path</param>
            /// <param name="checkpointState">(Output) Loaded OrtCheckpointState instance</param>
            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /* OrtStatus* */DOrtSaveCheckpoint(
                                            byte[] checkpointPath,
                                            IntPtr /*(OrtTrainingSession*)*/ session,
                                            bool saveOptimizerState);

            public static DOrtSaveCheckpoint OrtSaveCheckpoint;

            /// <summary>
            /// Creates an instance of OrtSession with provided parameters
            /// </summary>
            /// <param name="environment">Native OrtEnv instance</param>
            /// <param name="sessionOptions">Native SessionOptions instance</param>
            /// <param name="checkpointState">Loaded OrtCheckpointState instance</param>
            /// <param name="trainModelPath">UTF-8 bytes corresponding to model string path</param>
            /// <param name="evalModelPath">UTF-8 bytes corresponding to model string path</param>
            /// <param name="optimizerModelPath">UTF-8 bytes corresponding to model string path</param>
            /// <param name="session">(Output) Created native OrtTrainingSession instance</param>
            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /* OrtStatus* */DOrtCreateTrainingSession(
                                            IntPtr /* (OrtEnv*) */ environment,
                                            IntPtr /* (OrtSessionOptions*) */ sessionOptions,
                                            IntPtr /* (OrtCheckpointState*) */ checkpointState,
                                            byte[] trainModelPath,
                                            byte[] evalModelPath,
                                            byte[] optimizerModelPath,
                                            out IntPtr /* (OrtTrainingSession**) */ session);

            public static DOrtCreateTrainingSession OrtCreateTrainingSession;


            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /*(OrtStatus*)*/ DOrtGetTrainModelOutputCount(
                                                    IntPtr /*(OrtSession*)*/ session,
                                                    out UIntPtr count);

            public static DOrtGetTrainModelOutputCount OrtGetTrainModelOutputCount;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /*(OrtStatus*)*/ DOrtGetEvalModelOutputCount(
                                                    IntPtr /*(OrtSession*)*/ session,
                                                    out UIntPtr count);

            public static DOrtGetEvalModelOutputCount OrtGetEvalModelOutputCount;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /*(OrtStatus*)*/ DOrtGetTrainModelOutputName(
                                                    IntPtr /*(OrtSession*)*/ session,
                                                    UIntPtr index,
                                                    IntPtr /*(OrtAllocator*)*/ allocator,
                                                    out IntPtr /*(char**)*/name);

            public static DOrtGetTrainModelOutputName OrtGetTrainModelOutputName;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /*(OrtStatus*)*/ DOrtGetEvalModelOutputName(
                                                    IntPtr /*(OrtSession*)*/ session,
                                                    UIntPtr index,
                                                    IntPtr /*(OrtAllocator*)*/ allocator,
                                                    out IntPtr /*(char**)*/name);

            public static DOrtGetEvalModelOutputName OrtGetEvalModelOutputName;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /*(OrtStatus*)*/ DOrtResetGrad(
                                                    IntPtr /*(OrtSession*)*/ session);

            public static DOrtResetGrad OrtResetGrad;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /*(ONNStatus*)*/ DOrtTrainStep(
                                                    IntPtr /*(OrtTrainingSession*)*/ session,
                                                    IntPtr /*(OrtSessionRunOptions*)*/ runOptions,  // can be null to use the default options
                                                    UIntPtr inputCount,
                                                    IntPtr[] /* (OrtValue*[])*/ inputValues,
                                                    UIntPtr outputCount,
                                                    IntPtr[] outputValues /* An array of output value pointers. Array must be allocated by the caller */
                                                    );

            public static DOrtTrainStep OrtTrainStep;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /*(ONNStatus*)*/ DOrtEvalStep(
                                                    IntPtr /*(OrtTrainingSession*)*/ session,
                                                    IntPtr /*(OrtSessionRunOptions*)*/ runOptions,  // can be null to use the default options
                                                    UIntPtr inputCount,
                                                    IntPtr[] /* (OrtValue*[])*/ inputValues,
                                                    UIntPtr outputCount,
                                                    IntPtr[] outputValues /* An array of output value pointers. Array must be allocated by the caller */
                                                    );

            public static DOrtEvalStep OrtEvalStep;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /*(ONNStatus*)*/ DOrtOptimizerStep(
                                                    IntPtr /*(OrtTrainingSession*)*/ session,
                                                    IntPtr /*(OrtSessionRunOptions*)*/ runOptions  // can be null to use the default options
                                                    );

            public static DOrtOptimizerStep OrtOptimizerStep;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate void DOrtReleaseTrainingSession(IntPtr /*(OrtSession*)*/session);
            public static DOrtReleaseTrainingSession OrtReleaseTrainingSession;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate void DOrtReleaseCheckpointState(IntPtr /*(OrtSession*)*/session);
            public static DOrtReleaseCheckpointState OrtReleaseCheckpointState;

            #endregion TrainingSession API

            public static byte[] GetPlatformSerializedString(string str)
            {
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                    return System.Text.Encoding.Unicode.GetBytes(str + Char.MinValue);
                else
                    return System.Text.Encoding.UTF8.GetBytes(str + Char.MinValue);
            }

            public static bool TrainingEnabled()
            {
                if (trainingApiPtr == IntPtr.Zero)
                {
                    return false;
                }
                return true;
            }
        } //class NativeTrainingMethods
    } //namespace