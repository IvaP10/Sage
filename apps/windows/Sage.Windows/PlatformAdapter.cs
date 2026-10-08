using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using FlaUI.Core.AutomationElements;
using FlaUI.UIA3;
using Sage.Ipc.V2;

namespace Sage.Windows;

internal sealed class PlatformAdapter : IDisposable
{
    private readonly SemaphoreSlim _serial = new(1, 1);
    private readonly WinEventDelegate _focusCallback;
    private readonly nint _focusHook;
    private nint _lastWindow;

    public PlatformAdapter()
    {
        _lastWindow = GetForegroundWindow();
        _focusCallback = (_, _, window, _, _, _, _) =>
        {
            GetWindowThreadProcessId(window, out var process);
            if (process != Environment.ProcessId && window != 0) _lastWindow = window;
        };
        _focusHook = SetWinEventHook(3, 3, 0, _focusCallback, 0, 0, 0);
    }

    public async Task<AdapterResult> HandleAsync(AdapterRequest request, CancellationToken cancellationToken)
    {
        await _serial.WaitAsync(cancellationToken);
        try
        {
            return await Task.Run(() =>
            {
                var response = new AdapterResult { RequestId = request.RequestId };
                try
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    if (request.ExpiresAtUnixMs <= DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()) throw new InvalidOperationException("Adapter request expired");
                    using var payload = JsonDocument.Parse(request.Json);
                    using var automation = new UIA3Automation();
                    object data = request.Operation switch
                    {
                        "reference" => Reference(automation, cancellationToken),
                        "observe" => Observe(automation, payload.RootElement, cancellationToken),
                        "execute" or "application_identity" or "observe_application" => throw new InvalidOperationException("Signed Windows application identity is not implemented; native execution is unavailable"),
                        _ => throw new InvalidOperationException("Unknown adapter operation"),
                    };
                    response.Success = true; response.Json = JsonSerializer.Serialize(data);
                }
                catch (Exception error) { response.Error = error.Message; }
                return response;
            });
        }
        finally { _serial.Release(); }
    }

    private object Reference(UIA3Automation automation, CancellationToken cancellationToken)
    {
        var window = GetForegroundWindow();
        if (window == 0) return new { available = false, reason = "No foreground application is available." };
        GetWindowThreadProcessId(window, out var pid);
        if (pid == Environment.ProcessId)
        {
            window = _lastWindow;
            if (window == 0) return new { available = false, reason = "No foreground application is available." };
            GetWindowThreadProcessId(window, out pid);
        }
        using var process = Process.GetProcessById((int)pid);
        var root = automation.FromHandle(window);
        var elements = Descendants(root, 80, cancellationToken);
        var focused = elements.FirstOrDefault(e => e.Properties.HasKeyboardFocus.ValueOrDefault && !e.Properties.IsPassword.ValueOrDefault);
        var selection = focused?.Patterns.Text.IsSupported == true
            ? string.Join(" ", focused.Patterns.Text.Pattern.GetSelection().Take(2).Select(range => range.GetText(1000)))
            : "";
        return new
        {
            available = true,
            active_application = process.ProcessName,
            application_name = process.ProcessName,
            active_window = root.Name,
            selected_text = selection.Length > 4000 ? selection[..4000] : selection,
            selection = focused is null ? null : new { role = focused.ControlType.ToString(), label = focused.Name },
        };
    }

    private object Observe(UIA3Automation automation, JsonElement payload, CancellationToken cancellationToken)
    {
        var condition = payload.GetProperty("condition");
        if (Text(condition, "kind") == "application_running") return new { running = Running(Text(condition, "application")) is not null };
        if (Text(condition, "kind") != "element_present") throw new InvalidOperationException("Unsupported native observation");
        using var process = Running(Text(payload, "application"));
        if (process is null || process.MainWindowHandle == 0) return new { present = false };
        return new { present = Matching(automation.FromHandle(process.MainWindowHandle), condition.GetProperty("selector"), cancellationToken).Count == 1 };
    }

    private static List<AutomationElement> Matching(AutomationElement root, JsonElement selector, CancellationToken cancellationToken)
    {
        var role = Text(selector, "role"); var label = Text(selector, "label"); var id = Text(selector, "automation_id");
        if (role.Length + label.Length + id.Length == 0) throw new InvalidOperationException("Explicit semantic selector required");
        return Descendants(root, 500, cancellationToken).Where(e => !e.Properties.IsPassword.ValueOrDefault && !e.Properties.IsOffscreen.ValueOrDefault
            && (role.Length == 0 || e.ControlType.ToString().Equals(role, StringComparison.OrdinalIgnoreCase))
            && (label.Length == 0 || e.Name == label) && (id.Length == 0 || e.AutomationId == id)).ToList();
    }

    private static List<AutomationElement> Descendants(AutomationElement root, int limit, CancellationToken cancellationToken)
    {
        var result = new List<AutomationElement>(); var queue = new Queue<(AutomationElement, int)>(); queue.Enqueue((root, 0));
        var timer = Stopwatch.StartNew();
        while (queue.Count > 0 && result.Count < limit && timer.ElapsedMilliseconds < 1500)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var (element, depth) = queue.Dequeue(); result.Add(element);
            if (depth < 8) foreach (var child in element.FindAllChildren().Take(limit - result.Count)) queue.Enqueue((child, depth + 1));
        }
        return result;
    }

    private static Process? Running(string identifier)
    {
        if (string.IsNullOrWhiteSpace(identifier)) return null;
        var matches = Process.GetProcessesByName(Path.GetFileNameWithoutExtension(identifier));
        if (matches.Length == 1) return matches[0];
        foreach (var match in matches) match.Dispose(); return null;
    }
    private static string Text(JsonElement element, string key) => element.ValueKind == JsonValueKind.Object && element.TryGetProperty(key, out var value) && value.ValueKind == JsonValueKind.String ? value.GetString() ?? "" : "";
    public void Dispose() { if (_focusHook != 0) UnhookWinEvent(_focusHook); }
    private delegate void WinEventDelegate(nint hook, uint eventType, nint window, int objectId, int childId, uint thread, uint time);
    [DllImport("user32.dll")] private static extern nint SetWinEventHook(uint min, uint max, nint module, WinEventDelegate callback, uint process, uint thread, uint flags);
    [DllImport("user32.dll")] private static extern bool UnhookWinEvent(nint hook);
    [DllImport("user32.dll")] private static extern nint GetForegroundWindow();
    [DllImport("user32.dll")] private static extern uint GetWindowThreadProcessId(nint window, out uint process);
}
