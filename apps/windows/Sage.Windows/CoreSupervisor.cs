using System.Diagnostics;

namespace Sage.Windows;

internal sealed class CoreSupervisor
{
    private Process? _process;

    public void StartIfNeeded(byte[] secret)
    {
        if (_process is { HasExited: false }) return;
        var overridePath = Environment.GetEnvironmentVariable("SAGE_CORE_EXECUTABLE");
        var executable = string.IsNullOrWhiteSpace(overridePath)
            ? Path.Combine(AppContext.BaseDirectory, "sage-core.exe")
            : overridePath;
        if (!File.Exists(executable))
        {
            throw new FileNotFoundException(
                "sage-core.exe is missing beside the WinUI app; set SAGE_CORE_EXECUTABLE for source development",
                executable
            );
        }
        _process = Process.Start(new ProcessStartInfo
        {
            FileName = executable,
            Arguments = "--bootstrap-stdin",
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardInput = true,
            RedirectStandardOutput = false,
            RedirectStandardError = false,
        }) ?? throw new InvalidOperationException("Windows could not launch SAGE Core");
        _process.StandardInput.WriteLine(Convert.ToBase64String(secret));
        _process.StandardInput.Close();
    }
}
