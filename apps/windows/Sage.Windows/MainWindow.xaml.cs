using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.UI;
using Microsoft.UI.Input;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using Sage.Ipc.V1;
using Windows.System;
using Windows.UI.Core;
using WireTaskStatus = Sage.Ipc.V1.TaskStatus;

namespace Sage.Windows;

public sealed partial class MainWindow : Window
{
    private readonly CoreSupervisor _supervisor = new();
    private readonly SageCoreClient _client = new();
    private readonly TaskMetadataStore _taskMetadata = new();
    private readonly ObservableCollection<TaskRow> _tasks = [];
    private readonly ObservableCollection<AgentEvent> _timeline = [];
    private readonly Dictionary<string, List<AgentEvent>> _timelineByTaskId = [];
    private string? _selectedTaskId;
    private bool _draftActive;
    private bool _isSubmitting;
    private ProviderSettings? _providerSettings;
    private TaskCompletionSource<ProviderConnectionResult>? _providerTestWaiter;

    public MainWindow()
    {
        InitializeComponent();
        TaskList.ItemsSource = _tasks;
        Timeline.ItemsSource = _timeline;
        _client.EventReceived += OnCoreEvent;
        Closed += (_, _) => _client.Dispose();
        TaskToolbar.Visibility = Visibility.Collapsed;
        EmptyState.Visibility = Visibility.Visible;
        Composer_TextChanged(Composer, null!);
        _ = StartAsync();
    }

    private async Task StartAsync()
    {
        try
        {
            var secret = IpcSecretStore.LoadOrCreate();
            try
            {
                await _client.ConnectAsync();
                await _client.RequestStateAsync(includeCompleted: true);
                return;
            }
            catch
            {
                _supervisor.StartIfNeeded(secret);
            }
            Exception? finalError = null;
            for (var attempt = 0; attempt < 30; attempt++)
            {
                try
                {
                    await _client.ConnectAsync();
                    await _client.RequestStateAsync(includeCompleted: true);
                    return;
                }
                catch (Exception error)
                {
                    finalError = error;
                    await Task.Delay(150);
                }
            }
            throw finalError ?? new InvalidOperationException("SAGE Core did not open its named pipe");
        }
        catch (Exception error)
        {
            await ShowErrorAsync(error.Message);
        }
    }

    private void OnCoreEvent(object? sender, CoreEvent coreEvent)
    {
        DispatcherQueue.TryEnqueue(async () =>
        {
            switch (coreEvent.EventCase)
            {
                case CoreEvent.EventOneofCase.StateSnapshot:
                    LoadSnapshot(coreEvent.StateSnapshot);
                    break;
                case CoreEvent.EventOneofCase.TaskUpdate:
                    UpsertTask(coreEvent.TaskUpdate);
                    break;
                case CoreEvent.EventOneofCase.AgentEvent:
                    AddAgentEvent(coreEvent.AgentEvent);
                    await _client.RequestStateAsync(includeCompleted: true);
                    break;
                case CoreEvent.EventOneofCase.ApprovalRequest:
                    await ShowApprovalAsync(coreEvent.ApprovalRequest);
                    break;
                case CoreEvent.EventOneofCase.QuestionRequest:
                    await ShowQuestionAsync(coreEvent.QuestionRequest);
                    break;
                case CoreEvent.EventOneofCase.Error:
                    _isSubmitting = false;
                    UpdateComposerActions();
                    await ShowErrorAsync(coreEvent.Error.Message);
                    await _client.RequestStateAsync(includeCompleted: true);
                    break;
                case CoreEvent.EventOneofCase.ProviderConnectionResult:
                    _providerTestWaiter?.TrySetResult(coreEvent.ProviderConnectionResult);
                    break;
            }
        });
    }

    private void LoadSnapshot(StateSnapshot snapshot)
    {
        _providerSettings = snapshot.ProviderSettings.FirstOrDefault(item => item.Role == "reasoning");
        _tasks.Clear();
        foreach (var task in snapshot.Tasks)
        {
            if (!_taskMetadata.IsDeleted(task.TaskId))
            {
                _tasks.Add(new TaskRow(task, _taskMetadata.Get(task.TaskId)));
            }
        }
        ReorderTasks();

        if (_selectedTaskId is not null)
        {
            var selected = _tasks.FirstOrDefault(task => task.TaskId == _selectedTaskId);
            if (selected is not null)
            {
                ApplySelection(selected, focusComposer: false);
                return;
            }
        }

        if (!_draftActive && _tasks.Count > 0)
        {
            ApplySelection(_tasks[0], focusComposer: false);
        }
        else
        {
            SetNewTaskState(focusComposer: false);
        }
    }

    private void UpsertTask(TaskUpdate task)
    {
        if (_taskMetadata.IsDeleted(task.TaskId)) return;

        var existing = _tasks.FirstOrDefault(item => item.TaskId == task.TaskId);
        if (existing is null)
        {
            _tasks.Insert(0, new TaskRow(task, _taskMetadata.Get(task.TaskId)));
        }
        else
        {
            existing.Update(task, _taskMetadata.Get(task.TaskId));
        }
        ReorderTasks();

        if (_selectedTaskId == task.TaskId)
        {
            var selected = _tasks.First(item => item.TaskId == task.TaskId);
            SetTaskHeader(selected);
            RefreshTimeline(selected);
        }
        else if (_selectedTaskId is null && !_draftActive)
        {
            ApplySelection(_tasks.First(item => item.TaskId == task.TaskId), focusComposer: false);
        }
        _isSubmitting = false;
        UpdateComposerActions();
    }

    private void AddAgentEvent(AgentEvent agentEvent)
    {
        if (!string.IsNullOrEmpty(agentEvent.TaskId))
        {
            if (!_timelineByTaskId.TryGetValue(agentEvent.TaskId, out var events))
            {
                events = [];
                _timelineByTaskId[agentEvent.TaskId] = events;
            }
            events.Insert(0, agentEvent);
            if (events.Count > 200) events.RemoveAt(events.Count - 1);

            if (_selectedTaskId is null && !_draftActive)
            {
                _selectedTaskId = agentEvent.TaskId;
            }
            if (_selectedTaskId == agentEvent.TaskId)
            {
                var selected = _tasks.FirstOrDefault(task => task.TaskId == agentEvent.TaskId);
                if (selected is not null)
                {
                    TaskList.SelectedItem = selected;
                    SetTaskHeader(selected);
                }
                RefreshTimeline(selected);
            }
            _isSubmitting = false;
            UpdateComposerActions();
        }
        else
        {
            _timeline.Insert(0, agentEvent);
            while (_timeline.Count > 200) _timeline.RemoveAt(_timeline.Count - 1);
        }
    }

    private void ApplySelection(TaskRow row, bool focusComposer)
    {
        _draftActive = false;
        _selectedTaskId = row.TaskId;
        TaskList.SelectedItem = row;
        foreach (var task in _tasks) task.SetSelected(task.TaskId == _selectedTaskId);
        SetTaskHeader(row);
        RefreshTimeline(row);
        if (focusComposer) Composer.Focus(FocusState.Programmatic);
        UpdateComposerActions();
    }

    private void SetNewTaskState(bool focusComposer)
    {
        _draftActive = true;
        _selectedTaskId = null;
        TaskList.SelectedItem = null;
        foreach (var task in _tasks) task.SetSelected(false);
        _timeline.Clear();
        TaskToolbar.Visibility = Visibility.Collapsed;
        EmptyState.Visibility = Visibility.Visible;
        TaskTitle.Text = string.Empty;
        UndoButton.Visibility = Visibility.Collapsed;
        StopButton.Visibility = Visibility.Collapsed;
        SendButton.Visibility = Visibility.Visible;
        if (focusComposer)
        {
            Composer.Text = string.Empty;
            Composer.Focus(FocusState.Programmatic);
        }
        UpdateComposerActions();
    }

    private void SetTaskHeader(TaskRow row)
    {
        TaskToolbar.Visibility = Visibility.Visible;
        EmptyState.Visibility = Visibility.Collapsed;
        TaskTitle.Text = row.Request;
        UndoButton.Visibility = row.Task.UndoAvailable ? Visibility.Visible : Visibility.Collapsed;
        StopButton.Visibility = IsFinished(row.Task.Status) ? Visibility.Collapsed : Visibility.Visible;
    }

    private void RefreshTimeline(TaskRow? row)
    {
        _timeline.Clear();
        if (row is null)
        {
            EmptyState.Visibility = Visibility.Visible;
            return;
        }

        EmptyState.Visibility = Visibility.Collapsed;

        if (_timelineByTaskId.TryGetValue(row.TaskId, out var events))
        {
            foreach (var agentEvent in events) _timeline.Add(agentEvent);
        }

        if (_timeline.Count == 0)
        {
            var detail = row.Task.Summary;
            if (!string.IsNullOrWhiteSpace(row.Task.FinalOutcome))
            {
                detail = string.IsNullOrWhiteSpace(detail)
                    ? row.Task.FinalOutcome
                    : $"{detail}\n\n{row.Task.FinalOutcome}";
            }
            if (string.IsNullOrWhiteSpace(detail)) detail = row.Task.Request;
            _timeline.Add(new AgentEvent { Title = row.StatusLabel, Detail = detail });
        }
    }

    private void TaskList_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (TaskList.SelectedItem is TaskRow row)
        {
            ApplySelection(row, focusComposer: false);
        }
        else if (_draftActive)
        {
            // The draft state is already applied by New task or Delete.
        }
    }

    private void TaskRow_Tapped(object sender, TappedRoutedEventArgs e)
    {
        if ((sender as FrameworkElement)?.DataContext is TaskRow row)
        {
            ApplySelection(row, focusComposer: false);
        }
    }

    private void NewTask_Click(object sender, RoutedEventArgs e) => SetNewTaskState(focusComposer: true);

    private async void Send_Click(object sender, RoutedEventArgs e) => await SubmitAsync();

    private async void Stop_Click(object sender, RoutedEventArgs e)
    {
        var row = SelectedTask;
        if (row is null || IsFinished(row.Task.Status)) return;
        try
        {
            await _client.ControlTaskAsync(row.TaskId, ControlTask.Types.Operation.Cancel);
        }
        catch (Exception error)
        {
            await ShowErrorAsync(error.Message);
        }
    }

    private async void Undo_Click(object sender, RoutedEventArgs e)
    {
        var row = SelectedTask;
        if (row is null || !row.Task.UndoAvailable) return;
        try
        {
            await _client.UndoAsync(row.TaskId);
        }
        catch (Exception error)
        {
            await ShowErrorAsync(error.Message);
        }
    }

    private void Composer_TextChanged(object sender, TextChangedEventArgs e)
    {
        UpdateComposerActions();
    }

    private async void Composer_KeyDown(object sender, KeyRoutedEventArgs e)
    {
        if (e.Key == VirtualKey.Enter && !IsShiftDown())
        {
            e.Handled = true;
            await SubmitAsync();
        }
    }

    private async Task SubmitAsync()
    {
        var request = Composer.Text.Trim();
        if (request.Length == 0 || _isSubmitting || IsSelectedTaskActive()) return;
        _isSubmitting = true;
        _draftActive = false;
        Composer.Text = string.Empty;
        UpdateComposerActions();
        try
        {
            await _client.SubmitTaskAsync(request);
        }
        catch (Exception error)
        {
            _isSubmitting = false;
            _draftActive = true;
            Composer.Text = request;
            UpdateComposerActions();
            await ShowErrorAsync(error.Message);
        }
    }

    private void TaskRow_PointerEntered(object sender, PointerRoutedEventArgs e)
    {
        if ((sender as FrameworkElement)?.DataContext is TaskRow row) row.SetHovering(true);
    }

    private void TaskRow_PointerExited(object sender, PointerRoutedEventArgs e)
    {
        if ((sender as FrameworkElement)?.DataContext is TaskRow row) row.SetHovering(false);
    }

    private async void RenameTask_Click(object sender, RoutedEventArgs e)
    {
        if ((sender as MenuFlyoutItem)?.Tag is not TaskRow row) return;
        var editor = new TextBox
        {
            Text = row.Request,
            PlaceholderText = "Task title",
            MaxLength = 240,
        };
        var panel = new StackPanel { Spacing = 8 };
            panel.Children.Add(editor);
        var dialog = new ContentDialog
        {
            XamlRoot = Root.XamlRoot,
            Title = "Rename task",
            Content = panel,
            PrimaryButtonText = "Save",
            CloseButtonText = "Cancel",
            DefaultButton = ContentDialogButton.Primary,
        };
        editor.Focus(FocusState.Programmatic);
        if (await dialog.ShowAsync() != ContentDialogResult.Primary) return;
        var title = editor.Text.Trim();
        if (title.Length == 0) return;
        _taskMetadata.SetTitle(row.TaskId, title);
        row.SetPresentation(_taskMetadata.Get(row.TaskId));
        ReorderTasks();
        if (_selectedTaskId == row.TaskId) SetTaskHeader(row);
    }

    private void TogglePin_Click(object sender, RoutedEventArgs e)
    {
        if ((sender as MenuFlyoutItem)?.Tag is not TaskRow row) return;
        _taskMetadata.TogglePinned(row.TaskId);
        row.SetPresentation(_taskMetadata.Get(row.TaskId));
        ReorderTasks();
    }

    private async void DeleteTask_Click(object sender, RoutedEventArgs e)
    {
        if ((sender as MenuFlyoutItem)?.Tag is not TaskRow row) return;
        var dialog = new ContentDialog
        {
            XamlRoot = Root.XamlRoot,
            Title = "Delete conversation?",
            Content = $"Remove “{row.Request}” from Recent chats?",
            PrimaryButtonText = "Delete",
            CloseButtonText = "Cancel",
            DefaultButton = ContentDialogButton.Close,
        };
        if (await dialog.ShowAsync() != ContentDialogResult.Primary) return;
        if (!IsFinished(row.Task.Status))
        {
            try { await _client.ControlTaskAsync(row.TaskId, ControlTask.Types.Operation.Cancel); }
            catch { /* The history entry can still be removed locally. */ }
        }
        _taskMetadata.MarkDeleted(row.TaskId);
        var wasSelected = _selectedTaskId == row.TaskId;
        if (wasSelected)
        {
            _draftActive = true;
            _selectedTaskId = null;
            TaskList.SelectedItem = null;
        }
        _tasks.Remove(row);
        if (wasSelected) SetNewTaskState(focusComposer: true);
    }

    private async void Settings_Click(object sender, RoutedEventArgs e)
    {
        var provider = new ComboBox
        {
            ItemsSource = new[] { "openai", "openai-compatible" },
            PlaceholderText = "Provider",
        };
        provider.SelectedItem = _providerSettings?.Provider is "openai" or "openai-compatible"
            ? _providerSettings.Provider
            : null;
        var model = new TextBox
        {
            Text = _providerSettings?.Model ?? "gpt-5.4",
            PlaceholderText = "Model",
        };
        var endpoint = new TextBox
        {
            Text = _providerSettings?.Endpoint ?? string.Empty,
            PlaceholderText = "Endpoint (HTTPS, or localhost HTTP)",
        };
        var apiKey = new PasswordBox { PlaceholderText = "API key (optional for local runtimes)" };
        var removeKey = new CheckBox { Content = "Remove the saved credential", IsChecked = false };
        if (_providerSettings?.HasApiKey == true) removeKey.Visibility = Visibility.Visible;
        var status = new TextBlock { TextWrapping = TextWrapping.Wrap, Opacity = 0.8 };
        var test = new Button { Content = "Test connection", HorizontalAlignment = HorizontalAlignment.Left };
        var panel = new StackPanel { Spacing = 10, MinWidth = 380 };
        panel.Children.Add(new TextBlock { Text = "Provider" });
        panel.Children.Add(provider);
        panel.Children.Add(new TextBlock { Text = "Model" });
        panel.Children.Add(model);
        panel.Children.Add(new TextBlock { Text = "Endpoint" });
        panel.Children.Add(endpoint);
        panel.Children.Add(new TextBlock { Text = "API key" });
        panel.Children.Add(apiKey);
        panel.Children.Add(removeKey);
        panel.Children.Add(test);
        panel.Children.Add(status);
        var dialog = new ContentDialog
        {
            XamlRoot = Root.XamlRoot,
            Title = "Settings",
            Content = panel,
            PrimaryButtonText = "Save",
            CloseButtonText = "Done",
            DefaultButton = ContentDialogButton.Primary,
        };
        test.Click += async (_, _) =>
        {
            var selectedProvider = provider.SelectedItem as string;
            if (string.IsNullOrWhiteSpace(selectedProvider) || string.IsNullOrWhiteSpace(model.Text))
            {
                status.Text = "Choose a provider and model first.";
                return;
            }
            status.Text = "Testing…";
            _providerTestWaiter = new TaskCompletionSource<ProviderConnectionResult>(
                TaskCreationOptions.RunContinuationsAsynchronously);
            try
            {
                await _client.TestProviderConnectionAsync(
                    selectedProvider,
                    model.Text.Trim(),
                    endpoint.Text.Trim(),
                    apiKey.Password.Trim());
                var result = await _providerTestWaiter.Task.WaitAsync(TimeSpan.FromSeconds(95));
                status.Text = result.Success ? result.Message : $"Test failed: {result.Message}";
            }
            catch (Exception error)
            {
                status.Text = $"Test failed: {error.Message}";
            }
            finally
            {
                _providerTestWaiter = null;
            }
        };
        if (await dialog.ShowAsync() != ContentDialogResult.Primary) return;
        var providerValue = provider.SelectedItem as string;
        if (string.IsNullOrWhiteSpace(providerValue) || string.IsNullOrWhiteSpace(model.Text))
        {
            await ShowErrorAsync("Choose a provider and model before saving.");
            return;
        }
        var secret = apiKey.Password.Trim();
        var changedCredential = removeKey.IsChecked == true || secret.Length > 0;
        var authenticated = false;
        if (changedCredential)
        {
            authenticated = await NativeAuthentication.AuthenticateAsync(
                "Save this model provider credential in Windows Credential Manager");
        }
        await _client.SaveProviderSettingsAsync(
            providerValue,
            model.Text.Trim(),
            endpoint.Text.Trim(),
            secret,
            removeKey.IsChecked == true,
            authenticated);
    }

    private async Task ShowApprovalAsync(ApprovalRequest approval)
    {
        var content = $"{approval.Explanation}\n\nResource: {approval.Resource}\nRisk: {approval.Risk}";
        if (approval.RequiresNativeAuthentication)
        {
            content += "\n\nWindows device authentication is required.";
        }
        if (!approval.Reversible)
        {
            content += "\n\nThis action may not be reversible.";
        }
        var dialog = new ContentDialog
        {
            XamlRoot = Root.XamlRoot,
            Title = approval.Title,
            Content = content,
            PrimaryButtonText = "Approve once",
            CloseButtonText = "Deny",
            DefaultButton = ContentDialogButton.Close,
        };
        var result = await dialog.ShowAsync();
        var approve = result == ContentDialogResult.Primary;
        var authenticated = false;
        if (approve && approval.RequiresNativeAuthentication)
        {
            authenticated = await NativeAuthentication.AuthenticateAsync(approval.Explanation);
            approve = authenticated;
        }
        await _client.ResolveApprovalAsync(approval, approve, authenticated);
    }

    private async Task ShowQuestionAsync(QuestionRequest question)
    {
        var answer = new TextBox { AcceptsReturn = true, TextWrapping = TextWrapping.Wrap };
        var panel = new StackPanel { Spacing = 12 };
        panel.Children.Add(new TextBlock { Text = question.Question, TextWrapping = TextWrapping.Wrap });
        panel.Children.Add(answer);
        var dialog = new ContentDialog
        {
            XamlRoot = Root.XamlRoot,
            Title = "More information needed",
            Content = panel,
            PrimaryButtonText = "Send",
            CloseButtonText = "Cancel",
            DefaultButton = ContentDialogButton.Primary,
        };
        if (await dialog.ShowAsync() == ContentDialogResult.Primary && !string.IsNullOrWhiteSpace(answer.Text))
        {
            await _client.AnswerAsync(question, answer.Text);
        }
    }

    private async Task ShowErrorAsync(string message)
    {
        var dialog = new ContentDialog
        {
            XamlRoot = Root.XamlRoot,
            Title = "Error",
            Content = message,
            CloseButtonText = "OK",
        };
        await dialog.ShowAsync();
    }

    private TaskRow? SelectedTask => _tasks.FirstOrDefault(task => task.TaskId == _selectedTaskId);

    private bool IsSelectedTaskActive() => SelectedTask is { } task && !IsFinished(task.Task.Status);

    private void UpdateComposerActions()
    {
        var active = IsSelectedTaskActive();
        StopButton.Visibility = active ? Visibility.Visible : Visibility.Collapsed;
        SendButton.Visibility = active ? Visibility.Collapsed : Visibility.Visible;
        SendButton.IsEnabled = !_isSubmitting
            && !string.IsNullOrWhiteSpace(Composer.Text)
            && !IsSelectedTaskActive();
        if (_isSubmitting) SendButton.IsEnabled = false;
    }

    private void ReorderTasks()
    {
        var ordered = _tasks.OrderByDescending(task => task.IsPinned).ToList();
        for (var index = 0; index < ordered.Count; index++)
        {
            var currentIndex = _tasks.IndexOf(ordered[index]);
            if (currentIndex != index) _tasks.Move(currentIndex, index);
        }
    }

    private static bool IsShiftDown() =>
        InputKeyboardSource.GetKeyStateForCurrentThread(VirtualKey.Shift).HasFlag(CoreVirtualKeyStates.Down);

    private static bool IsFinished(WireTaskStatus status) => status is
        WireTaskStatus.Succeeded or WireTaskStatus.Failed or WireTaskStatus.Cancelled or WireTaskStatus.Interrupted;
}

public sealed class TaskRow : INotifyPropertyChanged
{
    private bool _selected;
    private bool _hovering;
    private string? _title;
    private bool _pinned;

    public TaskRow(TaskUpdate task, TaskMetadataStore.Entry metadata)
    {
        Task = task;
        _title = metadata.Title;
        _pinned = metadata.Pinned;
        UpdateVisuals();
    }

    public TaskUpdate Task { get; private set; }
    public string TaskId => Task.TaskId;
    public string Request => string.IsNullOrWhiteSpace(_title) ? Task.Request : _title;
    public string StatusLabel => FormatStatus(Task.Status);
    public Brush RowBackground { get; private set; } = new SolidColorBrush(Colors.Transparent);
    public bool IsPinned => _pinned;

    public event PropertyChangedEventHandler? PropertyChanged;

    public void Update(TaskUpdate task, TaskMetadataStore.Entry metadata)
    {
        Task = task;
        _title = metadata.Title;
        _pinned = metadata.Pinned;
        UpdateVisuals();
        NotifyPresentationChanged();
    }

    public void SetPresentation(TaskMetadataStore.Entry metadata)
    {
        _title = metadata.Title;
        _pinned = metadata.Pinned;
        UpdateVisuals();
        NotifyPresentationChanged();
    }

    public void SetSelected(bool selected)
    {
        _selected = selected;
        UpdateVisuals();
        OnPropertyChanged(nameof(RowBackground));
    }

    public void SetHovering(bool hovering)
    {
        _hovering = hovering;
        UpdateVisuals();
        OnPropertyChanged(nameof(RowBackground));
    }

    private void UpdateVisuals()
    {
        RowBackground = new SolidColorBrush(_selected ? Color.FromArgb(30, 255, 255, 255) : _hovering ? Color.FromArgb(16, 255, 255, 255) : Colors.Transparent);
    }

    private void NotifyPresentationChanged()
    {
        OnPropertyChanged(nameof(Task));
        OnPropertyChanged(nameof(Request));
        OnPropertyChanged(nameof(StatusLabel));
        OnPropertyChanged(nameof(RowBackground));
        OnPropertyChanged(nameof(IsPinned));
    }

    private static string FormatStatus(WireTaskStatus status) => status switch
    {
        WireTaskStatus.WaitingForApproval => "Needs approval",
        WireTaskStatus.WaitingForUser => "Waiting for you",
        WireTaskStatus.Succeeded => "Completed",
        WireTaskStatus.Failed => "Failed",
        WireTaskStatus.Cancelled => "Stopped",
        WireTaskStatus.Interrupted => "Interrupted",
        _ => status.ToString().Replace("_", " "),
    };

    private void OnPropertyChanged([CallerMemberName] string? propertyName = null) =>
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
}

public sealed class TaskMetadataStore
{
    public sealed class Entry
    {
        public string? Title { get; set; }
        public bool Pinned { get; set; }
        public bool Deleted { get; set; }
    }

    private readonly string _path = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "Sage",
        "ui-task-metadata.json");
    private readonly Dictionary<string, Entry> _entries;

    public TaskMetadataStore()
    {
        try
        {
            var json = File.Exists(_path) ? File.ReadAllText(_path) : string.Empty;
            _entries = string.IsNullOrWhiteSpace(json)
                ? []
                : JsonSerializer.Deserialize<Dictionary<string, Entry>>(json) ?? [];
        }
        catch
        {
            _entries = [];
        }
    }

    public Entry Get(string taskId) => _entries.TryGetValue(taskId, out var entry)
        ? new Entry { Title = entry.Title, Pinned = entry.Pinned, Deleted = entry.Deleted }
        : new Entry();

    public bool IsDeleted(string taskId) => _entries.TryGetValue(taskId, out var entry) && entry.Deleted;

    public void SetTitle(string taskId, string title)
    {
        var entry = Get(taskId);
        entry.Title = title;
        entry.Deleted = false;
        _entries[taskId] = entry;
        Save();
    }

    public void TogglePinned(string taskId)
    {
        var entry = Get(taskId);
        entry.Pinned = !entry.Pinned;
        _entries[taskId] = entry;
        Save();
    }

    public void MarkDeleted(string taskId)
    {
        var entry = Get(taskId);
        entry.Deleted = true;
        _entries[taskId] = entry;
        Save();
    }

    private void Save()
    {
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(_path)!);
            File.WriteAllText(_path, JsonSerializer.Serialize(_entries, new JsonSerializerOptions { WriteIndented = true }));
        }
        catch
        {
            // UI metadata is best-effort and never blocks the local core.
        }
    }
}
