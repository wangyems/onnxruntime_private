﻿// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
using System;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    ///  Holds the Checkpoint State as generated/consumed by on-device training APIs
    /// </summary>
    public class CheckpointState : SafeHandle
    {
        internal IntPtr Handle
        {
            get
            {
                return handle;
            }
        }

        /// <summary>
        /// Default __ctor. Creates default CheckpointState
        /// </summary>
        public CheckpointState()
            : base(IntPtr.Zero, true)
        {
            var envHandle = OrtEnv.Handle; // just so it is initialized
        }

        /// <summary>
        /// Overrides SafeHandle.IsInvalid
        /// </summary>
        /// <value>returns true if handle is equal to Zero</value>
        public override bool IsInvalid { get { return handle == IntPtr.Zero; } }

        /// <summary>
        /// Loads Checkpoint state from path
        /// </summary>
        /// <param name="checkpointPath"> absolute path to checkpoint</param>
        public void LoadCheckpoint(string checkpointPath)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtLoadCheckpoint(NativeMethods.GetPlatformSerializedString(checkpointPath), out handle));
        }

        #region SafeHandle
        /// <summary>
        /// Overrides SafeHandle.ReleaseHandle() to properly dispose of
        /// the native instance of CheckpointState
        /// </summary>
        /// <returns>always returns true</returns>
        protected override bool ReleaseHandle()
        {
            NativeMethods.OrtReleaseCheckpointState(handle);
            handle = IntPtr.Zero;
            return true;
        }

        #endregion
    }
}
