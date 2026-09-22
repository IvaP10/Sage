using System.Text.Json;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Sage.Ipc.V2;

namespace Sage.Windows;

public sealed partial class MainWindow
{
    private static string J(JsonElement record, string key) => record.ValueKind == JsonValueKind.Object && record.TryGetProperty(key, out var value) && value.ValueKind == JsonValueKind.String ? value.GetString() ?? "" : "";

    private void LoadKnowledge(string json)
    {
        try
        {
            using var document = JsonDocument.Parse(json); var data = document.RootElement;
            _conversationRecords.Clear(); _conversationRecords.AddRange(data.GetProperty("conversations").EnumerateArray().Select(e => e.Clone()));
            _memories.Clear(); _memories.AddRange(data.GetProperty("memories").EnumerateArray().Select(e => e.Clone()));
            _memoryEnabled = data.GetProperty("memory_enabled").GetBoolean();
            var messages = data.GetProperty("messages").EnumerateArray().Where(e => J(e, "conversation_id") == _selectedConversationId).Select(e => e.Clone()).ToList();
            if (messages.Count > 0) { _messages.Clear(); _messages.AddRange(messages); }
            RenderMemories();
        }
        catch (JsonException) { }
    }

    private async Task ChangeMemory(string operation, string id = "", string content = "", bool enabled = true)
    {
        await _client.KnowledgeAsync(new KnowledgeCommand { Operation = operation, Id = id, Content = content, Enabled = enabled, ConversationId = _selectedConversationId ?? "" });
    }

    private void RenderMemories()
    {
        if (_memoryPanel is null) return;
        _memoryPanel.Children.Clear();
        _memoryPanel.Children.Add(new TextBlock { Text = "Memory", FontSize = 18 });
        var toggle = new ToggleSwitch { Header = "Use memory", IsOn = _memoryEnabled };
        toggle.Toggled += async (_, _) => await ChangeMemory("configure", enabled: toggle.IsOn);
        _memoryPanel.Children.Add(toggle);
        foreach (var memory in _memories)
        {
            var id = J(memory, "id");
            var row = new StackPanel { Spacing = 6 };
            row.Children.Add(new TextBlock { Text = $"{J(memory, "subject")} · {J(memory, "kind")}" });
            var content = new TextBox { Text = J(memory, "content"), AcceptsReturn = true, TextWrapping = TextWrapping.Wrap, MaxLength = 4096 };
            row.Children.Add(content);
            var controls = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 8 };
            var enabled = new ToggleSwitch { IsOn = memory.GetProperty("enabled").GetBoolean(), Header = "Enabled" };
            enabled.Toggled += async (_, _) => await ChangeMemory(enabled.IsOn ? "enable" : "disable", id);
            var save = new Button { Content = "Save", IsEnabled = _memoryEnabled };
            save.Click += async (_, _) => await ChangeMemory("edit", id, content.Text);
            var delete = new Button { Content = "Delete" };
            var confirm = new MenuFlyout(); var confirmDelete = new MenuFlyoutItem { Text = "Delete this memory" };
            confirmDelete.Click += async (_, _) => await ChangeMemory("delete", id); confirm.Items.Add(confirmDelete); delete.Flyout = confirm;
            controls.Children.Add(enabled); controls.Children.Add(save); controls.Children.Add(delete); row.Children.Add(controls); _memoryPanel.Children.Add(row);
        }
        var draft = new TextBox { PlaceholderText = "Remember that…", MaxLength = 4096 };
        var remember = new Button { Content = "Remember", IsEnabled = _memoryEnabled };
        remember.Click += async (_, _) => { if (!string.IsNullOrWhiteSpace(draft.Text)) await ChangeMemory("remember", content: draft.Text); };
        _memoryPanel.Children.Add(draft); _memoryPanel.Children.Add(remember);
    }

    private void LoadWorkflows(string json)
    {
        if (_workflowPanel is null) return;
        using var document = JsonDocument.Parse(json);
        _workflowPanel.Children.Clear();
        _workflowPanel.Children.Add(new TextBlock { Text = "Skills and background tasks", FontSize = 18 });
        var selectedSkills = new List<string>();
        foreach (var kind in new[] { "skills", "workflows", "schedules" })
        {
            foreach (var item in document.RootElement.GetProperty(kind).EnumerateArray())
            {
                var id = J(item, "id"); var row = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 8 };
                if (kind == "skills")
                {
                    var pick = new CheckBox { Content = J(item, "name"), IsEnabled = item.GetProperty("enabled").GetBoolean() };
                    pick.Checked += (_, _) => selectedSkills.Add(id); pick.Unchecked += (_, _) => selectedSkills.Remove(id); row.Children.Add(pick);
                }
                else row.Children.Add(new TextBlock { Text = J(item, "name") });
                if (kind != "schedules")
                {
                    var run = new Button { Content = "Run", IsEnabled = item.GetProperty("enabled").GetBoolean() };
                    run.Click += async (_, _) => await _client.WorkflowAsync(new WorkflowCommand { Operation = kind == "skills" ? "run_skill" : "run_workflow", Id = id, ConversationId = _selectedConversationId ?? "" }); row.Children.Add(run);
                    if (kind == "skills" && !item.GetProperty("enabled").GetBoolean())
                    {
                        var preview = J(item, "preview"); var digest = J(item, "review_digest_candidate"); var title = J(item, "name");
                        var review = new Button { Content = "Review draft" };
                        review.Click += async (_, _) => {
                            var dialog = new ContentDialog { XamlRoot = Content.XamlRoot, Title = title,
                                Content = new ScrollViewer { Content = new TextBlock { Text = "Each run still needs permission for its resources and effects.\n\n" + preview, TextWrapping = TextWrapping.Wrap, IsTextSelectionEnabled = true }, MaxHeight = 400 },
                                PrimaryButtonText = "Enable skill", CloseButtonText = "Cancel" };
                            if (await dialog.ShowAsync() == ContentDialogResult.Primary)
                                await _client.WorkflowAsync(new WorkflowCommand { Operation = "review_skill", Id = id, Json = JsonSerializer.Serialize(new { digest }) });
                        }; row.Children.Add(review);
                    }
                }
                var delete = new Button { Content = "Delete" };
                delete.Click += async (_, _) => await _client.WorkflowAsync(new WorkflowCommand { Operation = "delete_" + kind[..^1], Id = id }); row.Children.Add(delete);
                _workflowPanel.Children.Add(row);
            }
        }
        var workflowName = new TextBox { PlaceholderText = "Workflow name" };
        var createWorkflow = new Button { Content = "Create workflow from selected skills" };
        createWorkflow.Click += async (_, _) => {
            if (selectedSkills.Count == 0 || string.IsNullOrWhiteSpace(workflowName.Text)) return;
            await _client.WorkflowAsync(new WorkflowCommand { Operation = "save_workflow", Json = JsonSerializer.Serialize(new { id = Guid.NewGuid(), name = workflowName.Text, skill_ids = selectedSkills, enabled = true }) });
        };
        _workflowPanel.Children.Add(workflowName); _workflowPanel.Children.Add(createWorkflow);
        var name = new TextBox { PlaceholderText = "Task name" }; var request = new TextBox { PlaceholderText = "What should happen?" };
        var when = new CalendarDatePicker { Date = DateTimeOffset.Now.AddDays(1) }; var time = new TimePicker { Time = DateTime.Now.TimeOfDay };
        var repeat = new ComboBox { ItemsSource = new[] { "Once", "Every hour", "Every day" }, SelectedIndex = 0 };
        var folder = new TextBox { PlaceholderText = "Watch a folder (optional)" };
        var schedule = new Button { Content = "Schedule" };
        schedule.Click += async (_, _) => {
            if (string.IsNullOrWhiteSpace(name.Text) || string.IsNullOrWhiteSpace(request.Text) || when.Date is null) return;
            var local = when.Date.Value.Date + time.Time;
            var at = new DateTimeOffset(local, TimeZoneInfo.Local.GetUtcOffset(local)).ToUniversalTime();
            object trigger = folder.Text.Length > 0 ? new { kind = "folder_changed", path = folder.Text } : repeat.SelectedIndex == 0 ? new { kind = "once", at = at.ToString("O") } : new { kind = "interval", seconds = repeat.SelectedIndex == 1 ? 3600 : 86400 };
            var command = new WorkflowCommand { Operation = "save_schedule", MaximumRuns = 100,
                BackgroundExpiresAtUnixMs = DateTimeOffset.UtcNow.AddDays(30).ToUnixTimeMilliseconds(),
                Json = JsonSerializer.Serialize(new { id = Guid.NewGuid(), name = name.Text, request = request.Text, conversation_id = Guid.NewGuid(), trigger, enabled = true, next_run_at = at.ToString("O"), last_condition = false }) };
            if (folder.Text.Length > 0) { var scope = new ResourceScope { Root = folder.Text }; scope.Effects.Add(ResourceEffect.Read); command.BackgroundResources.Add(scope); }
            await _client.WorkflowAsync(command);
        };
        foreach (var control in new UIElement[] { name, request, when, time, repeat, folder, schedule }) _workflowPanel.Children.Add(control);
        _workflowPanel.Children.Add(new TextBlock { Text = "Expires after 30 days or 100 runs. A watched folder may be read in the background. Additional access pauses for approval.", TextWrapping = TextWrapping.Wrap });
    }

    private async void Resume_Click(object sender, RoutedEventArgs e)
    {
        if (SelectedTask is { } row) await _client.ControlTaskAsync(row.TaskId, ControlTask.Types.Operation.Resume);
    }
    private async void SaveSkill_Click(object sender, RoutedEventArgs e)
    {
        if (SelectedTask is { } row) await _client.WorkflowAsync(new WorkflowCommand { Operation = "capture_skill", Id = row.TaskId, Name = row.Request });
    }
}
