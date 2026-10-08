using System.Text.Json;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Sage.Ipc.V2;
using Windows.UI;

namespace Sage.Windows;

public sealed partial class MainWindow
{
    private static string J(JsonElement record, string key) => record.ValueKind == JsonValueKind.Object && record.TryGetProperty(key, out var value) && value.ValueKind == JsonValueKind.String ? value.GetString() ?? "" : "";
    private static string[] Strings(JsonElement record, string key) =>
        record.ValueKind == JsonValueKind.Object && record.TryGetProperty(key, out var values) && values.ValueKind == JsonValueKind.Array
            ? values.EnumerateArray().Where(value => value.ValueKind == JsonValueKind.String).Select(value => value.GetString() ?? "").ToArray()
            : Array.Empty<string>();
    private static string EffectClasses(JsonElement record, string key)
    {
        var values = Strings(record, key);
        return values.Length == 0 ? "none" : string.Join(", ", values);
    }

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
            if (_connected) _ = _client.WorkflowAsync(new WorkflowCommand { Operation = "list" });
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
        var root = document.RootElement;
        _routineLearningEnabled = root.TryGetProperty("routine_learning_enabled", out var learning) && learning.GetBoolean();
        _routines.Clear();
        if (root.TryGetProperty("routines", out var routineList) && routineList.ValueKind == JsonValueKind.Array)
            _routines.AddRange(routineList.EnumerateArray().Select(item => item.Clone()));
        _routineFamilies.Clear();
        if (root.TryGetProperty("routine_families", out var familyList) && familyList.ValueKind == JsonValueKind.Array)
            _routineFamilies.AddRange(familyList.EnumerateArray().Select(item => item.Clone()));
        _workflowPanel.Children.Clear();
        _workflowPanel.Children.Add(new TextBlock { Text = "Skills and background tasks", FontSize = 18 });
        var learningCard = new StackPanel { Spacing = 8, Margin = new Thickness(0, 8, 0, 12) };
        learningCard.Children.Add(new TextBlock { Text = "Learn and reuse routines", FontSize = 16, FontWeight = Microsoft.UI.Text.FontWeights.SemiBold });
        var learn = new ToggleSwitch { Header = "Use reviewed routines", IsOn = _routineLearningEnabled, IsEnabled = _memoryEnabled };
        learn.Toggled += async (_, _) => await _client.WorkflowAsync(new WorkflowCommand {
            Operation = "configure_learning", Json = JsonSerializer.Serialize(new { enabled = learn.IsOn })
        });
        learningCard.Children.Add(learn);
        learningCard.Children.Add(new TextBlock {
            Text = _memoryEnabled
                ? "Sage can suggest repeated steps after three verified runs. Review each routine before use. Every run checks folder access again."
                : "Turn on Memory to learn routines. Examples stay on this device; every run still checks folder access.",
            TextWrapping = TextWrapping.Wrap
        });
        foreach (var routine in _routines)
        {
            var requests = routine.TryGetProperty("requests", out var requestArray) && requestArray.ValueKind == JsonValueKind.Array
                ? requestArray.EnumerateArray().Where(value => value.ValueKind == JsonValueKind.String).Select(value => value.GetString() ?? "").ToArray()
                : Array.Empty<string>();
            var steps = routine.TryGetProperty("steps", out var stepArray) && stepArray.ValueKind == JsonValueKind.Array
                ? stepArray.EnumerateArray().Where(value => value.ValueKind == JsonValueKind.String).Select(value => value.GetString() ?? "").ToArray()
                : Array.Empty<string>();
            var ready = routine.TryGetProperty("ready", out var readyValue) && readyValue.GetBoolean();
            var enabled = routine.TryGetProperty("enabled", out var enabledValue) && enabledValue.GetBoolean();
            var card = new StackPanel { Spacing = 7 };
            card.Children.Add(new TextBlock { Text = requests.FirstOrDefault() ?? "Repeated task", FontWeight = Microsoft.UI.Text.FontWeights.SemiBold, TextWrapping = TextWrapping.Wrap });
            if (requests.Length > 1) card.Children.Add(new TextBlock { Text = $"Also seen as: {string.Join(" · ", requests.Skip(1))}", TextWrapping = TextWrapping.Wrap });
            var count = routine.TryGetProperty("verified_runs", out var countValue) ? countValue.GetInt32() : 0;
            card.Children.Add(new TextBlock { Text = $"Observed in {count} verified tasks", Opacity = 0.75 });
            if (routine.TryGetProperty("evolution", out var evolution) && evolution.ValueKind == JsonValueKind.Object)
            {
                var sourceRuns = evolution.TryGetProperty("source_verified_runs", out var sourceCount) ? sourceCount.GetInt32() : 0;
                card.Children.Add(new TextBlock {
                    Text = $"Corrected variation of ‘{J(evolution, "source_request")}’ ({sourceRuns} source runs).",
                    Opacity = 0.8, TextWrapping = TextWrapping.Wrap
                });
                card.Children.Add(new TextBlock {
                    Text = $"Effect classes — same: {EffectClasses(evolution, "unchanged_effect_classes")}; added: {EffectClasses(evolution, "added_effect_classes")}; removed: {EffectClasses(evolution, "removed_effect_classes")}. This stays a draft until three independent runs and review.",
                    Opacity = 0.8, TextWrapping = TextWrapping.Wrap
                });
            }
            if (!ready)
                card.Children.Add(new TextBlock { Text = "Review opens after three verified tasks.", Opacity = 0.75, TextWrapping = TextWrapping.Wrap });
            else if (!_routineLearningEnabled)
                card.Children.Add(new TextBlock { Text = "Turn on routine learning to review and use this routine.", Opacity = 0.75, TextWrapping = TextWrapping.Wrap });
            for (var index = 0; index < steps.Length; index++)
                card.Children.Add(new TextBlock { Text = $"{index + 1}. {steps[index]}", TextWrapping = TextWrapping.Wrap });
            if (enabled)
                card.Children.Add(new TextBlock { Text = "Reviewed and ready · fresh access is still required", Opacity = 0.8 });
            else
            {
                var review = new Button { Content = "Review and enable", IsEnabled = ready && _routineLearningEnabled };
                review.Click += async (_, _) =>
                {
                    var preview = string.Join("\n", steps.Select((step, index) => $"{index + 1}. {step}"));
                    var dialog = new ContentDialog {
                        XamlRoot = Content.XamlRoot,
                        Title = "Review repeated steps",
                        Content = new ScrollViewer { Content = new TextBlock {
                            Text = $"Sage observed this request in {count} verified tasks:\n\n{string.Join("\n", requests)}\n\nSteps:\n{preview}\n\nEach run asks for fresh access to its resources.",
                            TextWrapping = TextWrapping.Wrap, IsTextSelectionEnabled = true
                        }, MaxHeight = 400 },
                        PrimaryButtonText = "Enable routine", CloseButtonText = "Cancel"
                    };
                    if (await dialog.ShowAsync() == ContentDialogResult.Primary)
                        await _client.WorkflowAsync(new WorkflowCommand {
                            Operation = "review_routine", Id = J(routine, "id"),
                            Json = JsonSerializer.Serialize(new { digest = J(routine, "review_digest") })
                        });
                };
                card.Children.Add(review);
            }
            var forget = new Button { Content = "Forget routine" };
            forget.Click += async (_, _) =>
            {
                var dialog = new ContentDialog {
                    XamlRoot = Content.XamlRoot,
                    Title = "Forget this routine?",
                    Content = "This removes the observed examples and your review from this device.",
                    PrimaryButtonText = "Forget routine", CloseButtonText = "Cancel"
                };
                if (await dialog.ShowAsync() == ContentDialogResult.Primary)
                    await _client.WorkflowAsync(new WorkflowCommand { Operation = "forget_routine", Id = J(routine, "id") });
            };
            card.Children.Add(forget);
            learningCard.Children.Add(new Border {
                Padding = new Thickness(10), Margin = new Thickness(0, 6, 0, 0), CornerRadius = new CornerRadius(8),
                Background = new Microsoft.UI.Xaml.Media.SolidColorBrush(Color.FromArgb(10, 255, 255, 255)),
                BorderBrush = new Microsoft.UI.Xaml.Media.SolidColorBrush(Color.FromArgb(24, 255, 255, 255)),
                BorderThickness = new Thickness(1), Child = card
            });
        }
        foreach (var family in _routineFamilies)
        {
            var branches = family.TryGetProperty("branches", out var branchValues)
                && branchValues.ValueKind == JsonValueKind.Array ? branchValues : default;
            var card = new StackPanel { Spacing = 8 };
            card.Children.Add(new TextBlock {
                Text = $"{(branches.ValueKind == JsonValueKind.Array ? branches.GetArrayLength() : 0)} reviewed routines share these steps",
                FontSize = 14, FontWeight = Microsoft.UI.Text.FontWeights.SemiBold, TextWrapping = TextWrapping.Wrap
            });
            card.Children.Add(new TextBlock {
                Text = "Sage found the same verified steps across these routines. Different next steps stay separate from the skill draft.",
                Opacity = 0.75, TextWrapping = TextWrapping.Wrap
            });
            card.Children.Add(new TextBlock {
                Text = "COMMON STEPS", FontSize = 11, FontWeight = Microsoft.UI.Text.FontWeights.SemiBold, Opacity = 0.75
            });
            var shared = Strings(family, "shared_steps");
            for (var index = 0; index < shared.Length; index++)
                card.Children.Add(new TextBlock { Text = $"{index + 1}. {shared[index]}", TextWrapping = TextWrapping.Wrap });
            card.Children.Add(new TextBlock {
                Text = "DIFFERENT NEXT STEPS", FontSize = 11, FontWeight = Microsoft.UI.Text.FontWeights.SemiBold, Opacity = 0.75,
                Margin = new Thickness(0, 4, 0, 0)
            });
            if (branches.ValueKind == JsonValueKind.Array)
            {
                foreach (var branch in branches.EnumerateArray())
                {
                    var branchCard = new StackPanel { Spacing = 4 };
                    var requests = Strings(branch, "requests");
                    var count = branch.TryGetProperty("verified_runs", out var branchCount) ? branchCount.GetInt32() : 0;
                    branchCard.Children.Add(new TextBlock {
                        Text = $"When you ask · {count} verified runs",
                        FontWeight = Microsoft.UI.Text.FontWeights.SemiBold, TextWrapping = TextWrapping.Wrap
                    });
                    branchCard.Children.Add(new TextBlock { Text = string.Join(" · ", requests), TextWrapping = TextWrapping.Wrap });
                    var next = Strings(branch, "next_steps");
                    if (next.Length == 0)
                        branchCard.Children.Add(new TextBlock { Text = "No additional steps in this branch.", Opacity = 0.75 });
                    for (var index = 0; index < next.Length; index++)
                        branchCard.Children.Add(new TextBlock { Text = $"Then {index + 1}. {next[index]}", TextWrapping = TextWrapping.Wrap });
                    card.Children.Add(new Border {
                        Padding = new Thickness(9), CornerRadius = new CornerRadius(8),
                        Background = new Microsoft.UI.Xaml.Media.SolidColorBrush(Color.FromArgb(10, 255, 255, 255)),
                        BorderBrush = new Microsoft.UI.Xaml.Media.SolidColorBrush(Color.FromArgb(24, 255, 255, 255)),
                        BorderThickness = new Thickness(1), Child = branchCard
                    });
                }
            }
            var skillName = new TextBox { PlaceholderText = "Name this shared skill", MaxLength = 100 };
            var createDraft = new Button { Content = "Create skill draft", IsEnabled = false };
            skillName.TextChanged += (_, _) => createDraft.IsEnabled = !string.IsNullOrWhiteSpace(skillName.Text);
            createDraft.Click += async (_, _) => await _client.WorkflowAsync(new WorkflowCommand {
                Operation = "synthesize_skill", Id = J(family, "id"), Name = skillName.Text,
                Json = JsonSerializer.Serialize(new { digest = J(family, "review_digest") })
            });
            card.Children.Add(skillName);
            card.Children.Add(createDraft);
            card.Children.Add(new TextBlock {
                Text = "Review the draft before enabling it. Each run checks access again.",
                Opacity = 0.75, TextWrapping = TextWrapping.Wrap
            });
            learningCard.Children.Add(new Border {
                Padding = new Thickness(12), Margin = new Thickness(0, 8, 0, 0), CornerRadius = new CornerRadius(10),
                Background = new Microsoft.UI.Xaml.Media.SolidColorBrush(Color.FromArgb(14, 255, 255, 255)),
                BorderBrush = new Microsoft.UI.Xaml.Media.SolidColorBrush(Color.FromArgb(32, 255, 255, 255)),
                BorderThickness = new Thickness(1), Child = card
            });
        }
        if (_routines.Count == 0)
            learningCard.Children.Add(new TextBlock {
                Text = _routineLearningEnabled
                    ? "Repeated tasks will appear here after Sage verifies them three times."
                    : "Turn on reviewed routines to let Sage learn from successful tasks.",
                Opacity = 0.75, TextWrapping = TextWrapping.Wrap
            });
        _workflowPanel.Children.Add(new Border {
            Padding = new Thickness(12), Margin = new Thickness(0, 8, 0, 12), CornerRadius = new CornerRadius(12),
            Background = new Microsoft.UI.Xaml.Media.SolidColorBrush(Color.FromArgb(14, 255, 255, 255)),
            BorderBrush = new Microsoft.UI.Xaml.Media.SolidColorBrush(Color.FromArgb(32, 255, 255, 255)),
            BorderThickness = new Thickness(1), Child = learningCard
        });
        var selectedSkills = new List<string>();
        foreach (var kind in new[] { "skills", "workflows", "schedules" })
        {
            foreach (var item in document.RootElement.GetProperty(kind).EnumerateArray())
            {
                var id = J(item, "id"); var row = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 8 };
                var sourcePaused = item.TryGetProperty("source_paused", out var sourcePausedValue)
                    && sourcePausedValue.ValueKind == JsonValueKind.True;
                if (kind == "skills")
                {
                    var pick = new CheckBox { Content = J(item, "name"), IsEnabled = item.GetProperty("enabled").GetBoolean() };
                    pick.Checked += (_, _) => selectedSkills.Add(id); pick.Unchecked += (_, _) => selectedSkills.Remove(id); row.Children.Add(pick);
                }
                else row.Children.Add(new TextBlock { Text = J(item, "name") });
                if (sourcePaused)
                    row.Children.Add(new TextBlock { Text = "Paused: review source routines", Opacity = 0.75, TextWrapping = TextWrapping.Wrap });
                if (kind == "schedules" && item.TryGetProperty("enabled", out var scheduleEnabled))
                    row.Children.Add(new TextBlock { Text = scheduleEnabled.GetBoolean() ? "Scheduled" : "Inactive", Opacity = 0.75 });
                if (kind != "schedules")
                {
                    var run = new Button { Content = "Run", IsEnabled = item.GetProperty("enabled").GetBoolean() };
                    run.Click += async (_, _) => await _client.WorkflowAsync(new WorkflowCommand { Operation = kind == "skills" ? "run_skill" : "run_workflow", Id = id, ConversationId = _selectedConversationId ?? "" }); row.Children.Add(run);
                    if (kind == "skills" && !item.GetProperty("enabled").GetBoolean() && !sourcePaused)
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
