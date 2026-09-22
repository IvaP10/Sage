using System.Text.Json.Nodes;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using FlaUI.Core.AutomationElements;
using FlaUI.UIA3;
using Sage.Ipc.V2;

namespace Sage.Windows;

internal sealed class PlatformAdapter : IDisposable
{
    private readonly Dictionary<string, DateTimeOffset> _usedGrants = [];
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

    public async Task<AdapterResult> HandleAsync(AdapterRequest request)
    {
        await _serial.WaitAsync();
        try
        {
            return await Task.Run(() =>
            {
                var response = new AdapterResult { RequestId = request.RequestId };
                try
                {
                    if (request.ExpiresAtUnixMs <= DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()) throw new InvalidOperationException("Adapter request expired");
                    var body = JsonNode.Parse(request.Json)!.AsObject();
                    if (request.Operation == "execute")
                    {
                        var grant = request.Grant ?? throw new InvalidOperationException("Missing typed grant");
                        if (grant.PolicyVersion != 2 || grant.Domain != "native" || grant.ResourceCase != ExecutionGrant.ResourceOneofCase.Application
                            || grant.ExpiresAtUnixMs <= DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()) throw new InvalidOperationException("Invalid typed grant");
                        body["capability"] = JsonSerializer.SerializeToNode(new { id = grant.GrantId, task_id = grant.RunId, action_id = grant.ActionId,
                            domain = grant.Domain, remaining_uses = 0, revoked = false,
                            expires_at = DateTimeOffset.FromUnixTimeMilliseconds(grant.ExpiresAtUnixMs).ToString("O"),
                            resource = new { kind = "application", identifier = grant.Application } });
                    }
                    using var payload = JsonDocument.Parse(body.ToJsonString());
                    using var automation = new UIA3Automation();
                    object data = request.Operation switch
                    {
                        "context" => Context(automation),
                        "observe" => Observe(automation, payload.RootElement),
                        "execute" => Execute(automation, payload.RootElement),
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

    private object Context(UIA3Automation automation)
    {
        var window = GetForegroundWindow();
        GetWindowThreadProcessId(window, out var pid);
        if (pid == Environment.ProcessId) window = _lastWindow;
        if (window == 0) return new { available = false };
        GetWindowThreadProcessId(window, out pid);
        using var process = Process.GetProcessById((int)pid);
        var root = automation.FromHandle(window);
        var elements = Descendants(root, 80);
        var focused = elements.FirstOrDefault(e => e.Properties.HasKeyboardFocus.ValueOrDefault && !e.Properties.IsPassword.ValueOrDefault);
        string selection = "";
        if (focused?.Patterns.Text.IsSupported == true)
            selection = string.Join(" ", focused.Patterns.Text.Pattern.GetSelection().Take(2).Select(range => range.GetText(1000)));
        return new
        {
            active_application = process.ProcessName,
            active_window = root.Name,
            selected_text = selection,
            screen_state = new { width = GetSystemMetrics(0), height = GetSystemMetrics(1) },
            accessibility_tree = elements.Where(e => !e.Properties.IsPassword.ValueOrDefault).Select(e => new { role = e.ControlType.ToString(), label = e.Name, automation_id = e.AutomationId }).ToArray(),
        };
    }

    private object Observe(UIA3Automation automation, JsonElement payload)
    {
        var condition = payload.GetProperty("condition");
        if (Text(condition, "kind") == "application_running") return new { running = Running(Text(condition, "application")) is not null };
        if (Text(condition, "kind") != "element_present") throw new InvalidOperationException("Unsupported native observation");
        using var process = Running(Text(payload, "application"));
        if (process is null || process.MainWindowHandle == 0) return new { present = false };
        return new { present = Matching(automation.FromHandle(process.MainWindowHandle), condition.GetProperty("selector")).Count == 1 };
    }

    private object Execute(UIA3Automation automation, JsonElement payload)
    {
        var action = payload.GetProperty("action"); var grant = payload.GetProperty("capability"); var resource = grant.GetProperty("resource");
        var identifier = Text(action, "application"); var id = Text(grant, "id");
        var expires = DateTimeOffset.Parse(Text(grant, "expires_at"));
        if (id.Length == 0 || Text(grant, "domain") != "native" || grant.GetProperty("remaining_uses").GetInt32() != 0 || grant.GetProperty("revoked").GetBoolean()
            || Text(resource, "kind") != "application" || Text(resource, "identifier") != identifier || expires <= DateTimeOffset.UtcNow)
            throw new InvalidOperationException("Native capability does not match this action");
        foreach (var stale in _usedGrants.Where(pair => pair.Value <= DateTimeOffset.UtcNow).Select(pair => pair.Key).ToArray()) _usedGrants.Remove(stale);
        if (!_usedGrants.TryAdd(id, expires)) throw new InvalidOperationException("Native capability already used");
        var kind = Text(action, "type");
        if (kind != "open_application") throw new InvalidOperationException("This native operation has no enabled feature contract");
        using var process = Running(identifier);
        if (kind == "open_application")
        {
            if (process is not null) { SetForegroundWindow(process.MainWindowHandle); return new { }; }
            if (!Path.IsPathFullyQualified(identifier) || !File.Exists(identifier) || !identifier.EndsWith(".exe", StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException("Opening an application requires its installed executable path");
            Process.Start(new ProcessStartInfo(identifier) { UseShellExecute = false }); return new { };
        }
        if (process is null || process.MainWindowHandle == 0 || process.Id == Environment.ProcessId) throw new InvalidOperationException("Target application is not available");
        if (kind == "close_application")
        {
            if (!process.CloseMainWindow()) throw new InvalidOperationException("Application declined to close"); return new { };
        }
        var root = automation.FromHandle(process.MainWindowHandle);
        if (kind == "press_shortcut")
        {
            var keys = action.GetProperty("keys").EnumerateArray().Select(e => e.GetString()!.ToLowerInvariant()).Order().ToArray();
            var allowed = new Dictionary<string, byte> { ["control+s"] = 0x53, ["control+f"] = 0x46, ["a+control"] = 0x41, ["c+control"] = 0x43, ["control+z"] = 0x5A };
            if (!allowed.TryGetValue(string.Join("+", keys), out var key)) throw new InvalidOperationException("Shortcut is not in the native allowlist");
            if (!SetForegroundWindow(process.MainWindowHandle) || GetForegroundWindow() != process.MainWindowHandle) throw new InvalidOperationException("Target lost focus");
            keybd_event(0x11, 0, 0, 0); keybd_event(key, 0, 0, 0); keybd_event(key, 0, 2, 0); keybd_event(0x11, 0, 2, 0);
            return new { };
        }
        var matches = Matching(root, action.GetProperty("selector"));
        if (matches.Count != 1) throw new InvalidOperationException("Semantic target is missing or ambiguous; observe again");
        var element = matches[0];
        if (kind == "click_element")
        {
            if (!element.Patterns.Invoke.IsSupported) throw new InvalidOperationException("Element does not support UI Automation Invoke");
            element.Patterns.Invoke.Pattern.Invoke();
        }
        else if (kind == "type_text")
        {
            if (action.TryGetProperty("sensitive", out var sensitive) && sensitive.GetBoolean()) throw new InvalidOperationException("Use native secure input for credentials");
            if (!element.Patterns.Value.IsSupported || element.Patterns.Value.Pattern.IsReadOnly.Value) throw new InvalidOperationException("Element does not support editable UI Automation Value");
            element.Patterns.Value.Pattern.SetValue(Text(action, "text"));
        }
        else throw new InvalidOperationException("No structured adapter is installed for this application action");
        return new { };
    }

    private static List<AutomationElement> Matching(AutomationElement root, JsonElement selector)
    {
        var role = Text(selector, "role"); var label = Text(selector, "label"); var id = Text(selector, "automation_id");
        if (role.Length + label.Length + id.Length == 0) throw new InvalidOperationException("Explicit semantic selector required");
        return Descendants(root, 500).Where(e => !e.Properties.IsPassword.ValueOrDefault && !e.Properties.IsOffscreen.ValueOrDefault
            && (role.Length == 0 || e.ControlType.ToString().Equals(role, StringComparison.OrdinalIgnoreCase))
            && (label.Length == 0 || e.Name == label) && (id.Length == 0 || e.AutomationId == id)).ToList();
    }

    private static List<AutomationElement> Descendants(AutomationElement root, int limit)
    {
        var result = new List<AutomationElement>(); var queue = new Queue<(AutomationElement, int)>(); queue.Enqueue((root, 0));
        var timer = Stopwatch.StartNew();
        while (queue.Count > 0 && result.Count < limit && timer.ElapsedMilliseconds < 1500)
        {
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
    [DllImport("user32.dll")] private static extern bool SetForegroundWindow(nint window);
    [DllImport("user32.dll")] private static extern uint GetWindowThreadProcessId(nint window, out uint process);
    [DllImport("user32.dll")] private static extern int GetSystemMetrics(int index);
    [DllImport("user32.dll")] private static extern void keybd_event(byte key, byte scan, uint flags, nuint extra);
}
