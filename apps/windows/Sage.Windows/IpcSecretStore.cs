using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;

namespace Sage.Windows;

// Uses the same Win32 Credential Manager target convention as keyring-rs:
// "{username}.{service}". This lets the C# UI and Rust core independently
// retrieve the installation IPC key without placing it on disk or a command line.
internal static class IpcSecretStore
{
    private const string Service = "com.ivanpadeliya.sage";
    private const string UserName = "local-ipc-v1";
    private const string Target = UserName + "." + Service;
    private const uint CredentialTypeGeneric = 1;
    private const uint CredentialPersistEnterprise = 3;
    private const int ErrorNotFound = 1168;

    public static byte[] LoadOrCreate()
    {
        if (TryRead(out var encoded))
        {
            var existing = Convert.FromBase64String(encoded);
            if (existing.Length != 32)
            {
                throw new CryptographicException("Credential Manager IPC key has an invalid length");
            }
            return existing;
        }

        var secret = RandomNumberGenerator.GetBytes(32);
        Write(Convert.ToBase64String(secret));
        return secret;
    }

    private static bool TryRead(out string secret)
    {
        if (!CredRead(Target, CredentialTypeGeneric, 0, out var pointer))
        {
            var error = Marshal.GetLastWin32Error();
            if (error == ErrorNotFound)
            {
                secret = string.Empty;
                return false;
            }
            throw new Win32Exception(error, "Credential Manager could not read the SAGE IPC key");
        }
        try
        {
            var credential = Marshal.PtrToStructure<NativeCredential>(pointer);
            var bytes = new byte[credential.CredentialBlobSize];
            if (bytes.Length > 0)
            {
                Marshal.Copy(credential.CredentialBlob, bytes, 0, bytes.Length);
            }
            secret = Encoding.UTF8.GetString(bytes);
            CryptographicOperations.ZeroMemory(bytes);
            return true;
        }
        finally
        {
            CredFree(pointer);
        }
    }

    private static void Write(string secret)
    {
        var bytes = Encoding.UTF8.GetBytes(secret);
        var blob = Marshal.AllocCoTaskMem(bytes.Length);
        try
        {
            Marshal.Copy(bytes, 0, blob, bytes.Length);
            var credential = new NativeCredential
            {
                Type = CredentialTypeGeneric,
                TargetName = Target,
                CredentialBlobSize = bytes.Length,
                CredentialBlob = blob,
                Persist = CredentialPersistEnterprise,
                UserName = UserName,
            };
            if (!CredWrite(ref credential, 0))
            {
                throw new Win32Exception(
                    Marshal.GetLastWin32Error(),
                    "Credential Manager could not store the SAGE IPC key"
                );
            }
        }
        finally
        {
            CryptographicOperations.ZeroMemory(bytes);
            Marshal.FreeCoTaskMem(blob);
        }
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    private struct NativeCredential
    {
        public uint Flags;
        public uint Type;
        [MarshalAs(UnmanagedType.LPWStr)] public string TargetName;
        [MarshalAs(UnmanagedType.LPWStr)] public string? Comment;
        public System.Runtime.InteropServices.ComTypes.FILETIME LastWritten;
        public int CredentialBlobSize;
        public IntPtr CredentialBlob;
        public uint Persist;
        public uint AttributeCount;
        public IntPtr Attributes;
        [MarshalAs(UnmanagedType.LPWStr)] public string? TargetAlias;
        [MarshalAs(UnmanagedType.LPWStr)] public string UserName;
    }

    [DllImport("advapi32.dll", EntryPoint = "CredReadW", CharSet = CharSet.Unicode, SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool CredRead(
        string target,
        uint type,
        uint flags,
        out IntPtr credential
    );

    [DllImport("advapi32.dll", EntryPoint = "CredWriteW", CharSet = CharSet.Unicode, SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool CredWrite(ref NativeCredential credential, uint flags);

    [DllImport("advapi32.dll", SetLastError = false)]
    private static extern void CredFree(IntPtr buffer);
}
