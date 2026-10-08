using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.UI;
using Microsoft.UI.Dispatching;
using Microsoft.UI.Input;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using Sage.Ipc.V2;
using Windows.System;
using Windows.UI;
using Windows.UI.Core;
using WireTaskStatus = Sage.Ipc.V2.TaskStatus;

namespace Sage.Windows;

public sealed partial class MainWindow : Window
{
    private readonly CoreSupervisor _supervisor = new();
    private readonly SageCoreClient _client = new();
    private readonly PlatformAdapter _platformAdapter = new();
    private readonly TaskMetadataStore _taskMetadata = new();
    private readonly ObservableCollection<TaskRow> _tasks = [];
    private readonly ObservableCollection<TaskRow> _visibleTasks = [];
    private readonly ObservableCollection<TimelineRow> _timeline = [];
    private readonly ObservableCollection<TimelineRow> _activity = [];
    private readonly SnapshotRefreshGate _snapshotRefresh = new();
    private readonly DispatcherQueueTimer _snapshotTimer;
    private readonly DispatcherQueueTimer _responseTimer;
    private readonly object _responseLock = new();
    private readonly Dictionary<string, ModelResponseDelta> _pendingResponses = [];
    private bool _responseDispatchQueued;
    private bool _closed;
    private bool _synchronizingTasks;
    private bool _followLatest = true;
    private ScrollViewer? _timelineScrollViewer;
    private readonly Dictionary<string, List<AgentEvent>> _timelineByTaskId = [];
    private string? _selectedTaskId;
    private string? _selectedConversationId;
    private readonly List<JsonElement> _conversationRecords = [];
    private readonly List<JsonElement> _messages = [];
    private readonly List<JsonElement> _memories = [];
    private bool _memoryEnabled = true;
    private bool _routineLearningEnabled;
    private readonly List<JsonElement> _routines = [];
    private readonly List<JsonElement> _routineFamilies = [];
    private StackPanel? _memoryPanel;
    private StackPanel? _workflowPanel;
    private bool _draftActive;
    private bool _isSubmitting;
    private bool _reconnecting;
    private bool _connected;
    private bool _storageLocked = true;
    private string? _unconfirmedRequest;
    private readonly string _intentStreamId = Guid.NewGuid().ToString();
    private ulong _intentRevision;
    private CancellationTokenSource? _intentPreparation;
    private sealed record DecisionPrompt(string Id, string TaskId, long ExpiresAt, ApprovalRequest? Approval, QuestionRequest? Question);
    private readonly Dictionary<string, DecisionPrompt> _decisionInbox = new();
    private readonly HashSet<string> _deferredDecisions = new();
    private readonly HashSet<string> _submittedDecisions = new();
    private readonly SemaphoreSlim _dialogGate = new(1, 1);
    private bool _processingDecisions;
    private string? _activeDecisionId;
    private ContentDialog? _activeDecisionDialog;

    public MainWindow()
    {
        InitializeComponent();
        TaskList.ItemsSource = _visibleTasks;
        Timeline.ItemsSource = _timeline;
        ActivityList.ItemsSource = _activity;
        _snapshotTimer = DispatcherQueue.CreateTimer();
        _snapshotTimer.Interval = TimeSpan.FromMilliseconds(180);
        _snapshotTimer.Tick += SnapshotTimer_Tick;
        _responseTimer = DispatcherQueue.CreateTimer();
        _responseTimer.Interval = TimeSpan.FromMilliseconds(33);
        _responseTimer.Tick += ResponseTimer_Tick;
        _client.EventReceived += OnCoreEvent;
        _client.Disconnected += (_, _) => DispatcherQueue.TryEnqueue(async () => await StartAsync());
        _client.AdapterHandler = _platformAdapter.HandleAsync;
        Closed += (_, _) =>
        {
            _closed = true;
            _snapshotTimer.Stop();
            _responseTimer.Stop();
            _client.EventReceived -= OnCoreEvent;
            _client.Dispose();
            _platformAdapter.Dispose();
        };
        TaskToolbar.Visibility = Visibility.Collapsed;
        EmptyState.Visibility = Visibility.Visible;
        Composer_TextChanged(Composer, null!);
        _ = StartAsync();
    }

    private async Task StartAsync()
    {
        if (_reconnecting || _closed) return;
        _reconnecting = true;
        _snapshotRefresh.Reset();
        _snapshotTimer.Stop();
        _submittedDecisions.Clear();
        _connected = false;
        ReconnectButton.Visibility = Visibility.Visible;
        ReconnectButton.Content = "Connecting…";
        ReconnectButton.IsEnabled = false;
        UpdateComposerActions();
        try
        {
            var secret = IpcSecretStore.LoadOrCreate();
            try
            {
                await _client.ConnectAsync();
                await RequestSnapshotAsync();
                _connected = true;
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
                    await RequestSnapshotAsync();
                    _connected = true;
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
            _isSubmitting = false;
            if (string.IsNullOrWhiteSpace(Composer.Text) && _unconfirmedRequest is not null) Composer.Text = _unconfirmedRequest;
            await ShowErrorAsync(error.Message);
        }
        finally
        {
            _reconnecting = false;
            ReconnectButton.Visibility = _connected ? Visibility.Collapsed : Visibility.Visible;
            ReconnectButton.Content = "Reconnect";
            ReconnectButton.IsEnabled = true;
            UpdateComposerActions();
        }
    }

    private async void Reconnect_Click(object sender, RoutedEventArgs e) => await StartAsync();

    private void OnCoreEvent(object? sender, CoreEvent coreEvent)
    {
        if (_closed) return;
        if (coreEvent.EventCase == CoreEvent.EventOneofCase.ModelResponseDelta)
        {
            // A cumulative delta supersedes earlier text from the same task. Keep at most one
            // dispatcher wakeup per render interval, even when generation outruns the UI.
            lock (_responseLock)
            {
                if (_pendingResponses.Count < 256 || _pendingResponses.ContainsKey(coreEvent.ModelResponseDelta.TaskId))
                    _pendingResponses[coreEvent.ModelResponseDelta.TaskId] = coreEvent.ModelResponseDelta;
                if (_responseDispatchQueued) return;
                _responseDispatchQueued = true;
            }
            DispatcherQueue.TryEnqueue(() => { if (!_closed) _responseTimer.Start(); });
            return;
        }
        DispatcherQueue.TryEnqueue(async () =>
        {
            if (_closed) return;
            switch (coreEvent.EventCase)
            {
                case CoreEvent.EventOneofCase.IntentPreview:
                    var preview = coreEvent.IntentPreview;
                    if (preview.StreamId != _intentStreamId || preview.Revision != _intentRevision) break;
                    IntentCard.Visibility = preview.Steps.Count > 0 || preview.RoutineSuggestions.Count > 0 || preview.Status is "paused" or "stopping" ? Visibility.Visible : Visibility.Collapsed;
                    IntentHeading.Text = preview.Status switch
                    {
                        "unavailable" => "Action unavailable",
                        "paused" => "Paused for your correction",
                        "stopping" => "Stopping",
                        "suggested" => "Possible learned path",
                        "prepared" => "Ready to open",
                        "preparing" => "Preparing your request",
                        _ => "Ready on this device"
                    };
                    IntentSteps.Text = string.Join("\n", preview.Steps.Select((step, index) => $"{index + 1}  {step}"));
                    IntentDetail.Text = preview.Detail;
                    IntentSuggestionDetail.Text = preview.RoutineSuggestionDetail;
                    IntentSuggestionPanel.Visibility = preview.RoutineSuggestions.Count > 0 ? Visibility.Visible : Visibility.Collapsed;
                    IntentSuggestions.Children.Clear();
                    foreach (var suggestion in preview.RoutineSuggestions.Take(2))
                    {
                        var request = suggestion;
                        var button = new Button
                        {
                            Content = new TextBlock { Text = $"Use: {request}", TextWrapping = TextWrapping.Wrap, MaxWidth = 260, MaxLines = 2 },
                            MaxWidth = 280
                        };
                        AutomationProperties.SetName(button, $"Use learned request: {request}");
                        ToolTipService.SetToolTip(button, request);
                        button.Click += (_, _) =>
                        {
                            Composer.Text = request;
                            Composer.Focus(FocusState.Programmatic);
                            Composer.SelectionStart = Composer.Text.Length;
                        };
                        IntentSuggestions.Children.Add(button);
                    }
                    break;
                case CoreEvent.EventOneofCase.TaskAccepted:
                    _selectedTaskId = coreEvent.TaskAccepted.TaskId;
                    _isSubmitting = false;
                    _draftActive = false;
                    if (Composer.Text == _unconfirmedRequest) Composer.Text = "";
                    _unconfirmedRequest = null;
                    UpdateComposerActions();
                    ScheduleSnapshotRefresh();
                    break;
                case CoreEvent.EventOneofCase.StateSnapshot:
                    _snapshotRefresh.Complete();
                    LoadSnapshot(coreEvent.StateSnapshot);
                    break;
                case CoreEvent.EventOneofCase.TaskUpdate:
                    UpsertTask(coreEvent.TaskUpdate);
                    break;
                case CoreEvent.EventOneofCase.AgentEvent:
                    AddAgentEvent(coreEvent.AgentEvent);
                    if (!_tasks.Any(task => task.TaskId == coreEvent.AgentEvent.TaskId)) ScheduleSnapshotRefresh();
                    break;
                case CoreEvent.EventOneofCase.ApprovalRequest:
                    QueueDecision(coreEvent.ApprovalRequest);
                    break;
                case CoreEvent.EventOneofCase.QuestionRequest:
                    QueueDecision(coreEvent.QuestionRequest);
                    break;
                case CoreEvent.EventOneofCase.DecisionResolved:
                    RemoveDecision(coreEvent.DecisionResolved.DecisionId);
                    break;
                case CoreEvent.EventOneofCase.Error:
                    _submittedDecisions.Clear();
                    _isSubmitting = false;
                    UpdateComposerActions();
                    await ShowErrorAsync(coreEvent.Error.Message);
                    ScheduleSnapshotRefresh();
                    break;
                case CoreEvent.EventOneofCase.KnowledgeState:
                    LoadKnowledge(coreEvent.KnowledgeState.Json);
                    if (SelectedTask is { } selectedTask) RefreshTimeline(selectedTask);
                    break;
                case CoreEvent.EventOneofCase.WorkflowState:
                    LoadWorkflows(coreEvent.WorkflowState.Json);
                    break;
                case CoreEvent.EventOneofCase.Notification:
                    _isSubmitting = false;
                    ScheduleSnapshotRefresh();
                    await _client.KnowledgeAsync(new KnowledgeCommand { Operation = "list", ConversationId = _selectedConversationId ?? "" });
                    break;
            }
        });
    }

    private void LoadSnapshot(StateSnapshot snapshot)
    {
        _storageLocked = snapshot.StorageLocked;
        _decisionInbox.Clear();
        foreach (var approval in snapshot.PendingApprovals) AddDecision(new(approval.ApprovalId, approval.TaskId, approval.ExpiresAtUnixMs, approval, null));
        foreach (var question in snapshot.PendingQuestions) AddDecision(new(question.QuestionId, question.TaskId, question.ExpiresAtUnixMs, null, question));
        RefreshDecisionQueue();
        UnlockHistoryButton.Visibility = snapshot.StorageLocked ? Visibility.Visible : Visibility.Collapsed;
        if (snapshot.Knowledge is not null) LoadKnowledge(snapshot.Knowledge.Json);
        var desired = new List<TaskRow>();
        var existingRows = _tasks.ToDictionary(row => ConversationKey(row.Task));
        foreach (var task in snapshot.Tasks.GroupBy(ConversationKey).Select(group => group.First()))
        {
            if (_taskMetadata.IsDeleted(task.TaskId)) continue;
            if (_conversationRecords.Count > 0 && !_conversationRecords.Any(record => J(record, "id") == task.ConversationId)) continue;
            var metadata = _taskMetadata.Get(task.TaskId);
            var record = _conversationRecords.FirstOrDefault(record => J(record, "id") == task.ConversationId);
            if (record.ValueKind == JsonValueKind.Object)
            {
                metadata.Title = J(record, "title");
                metadata.Pinned = record.TryGetProperty("pinned", out var pinned) && pinned.ValueKind == JsonValueKind.True;
            }
            if (existingRows.TryGetValue(ConversationKey(task), out var existing))
            {
                existing.Update(task, metadata);
                desired.Add(existing);
            }
            else desired.Add(new TaskRow(task, metadata));
        }
        SynchronizeRows(_tasks, desired);
        if (_selectedConversationId is not null) _selectedTaskId = _tasks.FirstOrDefault(row => row.Task.ConversationId == _selectedConversationId)?.TaskId ?? _selectedTaskId;
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

        var existing = _tasks.FirstOrDefault(item => ConversationKey(item.Task) == ConversationKey(task));
        if (existing is null)
        {
            _tasks.Insert(0, new TaskRow(task, _taskMetadata.Get(task.TaskId)));
        }
        else
        {
            // A conversation is a stable row across follow-up tasks and fresh recovery runs.
            var wasSelected = _selectedTaskId == existing.TaskId;
            existing.Update(task, _taskMetadata.Get(task.TaskId));
            if (wasSelected) _selectedTaskId = task.TaskId;
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
        if (IsFinished(task.Status) && task.TaskId == _selectedTaskId)
            _ = RequestConversationAsync(_selectedConversationId ?? "");
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
            if (_timelineByTaskId.Count > 256)
            {
                var oldest = _timelineByTaskId.Keys.FirstOrDefault(id => id != _selectedTaskId && id != agentEvent.TaskId);
                if (oldest is not null) _timelineByTaskId.Remove(oldest);
            }
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
            // Global events do not belong to whichever conversation happens to be open.
            ScheduleSnapshotRefresh();
        }
    }

    private void ApplySelection(TaskRow row, bool focusComposer)
    {
        _draftActive = false;
        _selectedTaskId = row.TaskId;
        var changedConversation = _selectedConversationId != row.Task.ConversationId;
        _selectedConversationId = row.Task.ConversationId;
        if (changedConversation)
        {
            _followLatest = true;
            _messages.Clear();
            _ = RequestConversationAsync(_selectedConversationId ?? "");
        }
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
        _selectedConversationId = null;
        _messages.Clear();
        _followLatest = true;
        _taskFolder = null;
        UpdateFolderScope();
        TaskList.SelectedItem = null;
        foreach (var task in _tasks) task.SetSelected(false);
        _timeline.Clear();
        _activity.Clear();
        ActivityExpander.Visibility = Visibility.Collapsed;
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
        TaskStatusText.Text = row.StatusLabel;
        var verified = row.Task.ExecutionFacts?.Verified ?? row.Task.CompletedActions;
        ActionProgressText.Text = row.Task.TotalActions > 0 ? $"{verified} of {row.Task.TotalActions} actions verified" : "";
        IntentChangeText.Text = row.Task.IntentChangeSummary;
        IntentChangeText.Visibility = string.IsNullOrEmpty(row.Task.IntentChangeSummary) ? Visibility.Collapsed : Visibility.Visible;
        RoutineText.Text = row.Task.RoutineSummary;
        RoutineText.Visibility = string.IsNullOrEmpty(row.Task.RoutineSummary) ? Visibility.Collapsed : Visibility.Visible;
        var stages = $"Preparing {row.Task.Actions.Count(a => a.Status == "compiling")} · Running {row.Task.Actions.Count(a => a.Status == "running")} · Verifying {row.Task.Actions.Count(a => a.Status == "verifying")}";
        CurrentActionText.Text = row.Task.TotalActions > 0 ? $"{row.Task.CurrentAction}\n{stages}" : row.Task.CurrentAction;
        CurrentActionText.Visibility = !IsFinished(row.Task.Status) && !string.IsNullOrWhiteSpace(row.Task.CurrentAction) ? Visibility.Visible : Visibility.Collapsed;
        UndoButton.Visibility = row.Task.UndoAvailable ? Visibility.Visible : Visibility.Collapsed;
        UndoButton.Content = row.Task.UndoState is "dispatched" or "uncertain" ? "Check Undo" : row.Task.UndoState == "prepared" ? "Retry Undo" : "Undo";
        ToolTipService.SetToolTip(UndoButton, string.IsNullOrEmpty(row.Task.UndoSummary) ? "Undo the last reversible action" : row.Task.UndoSummary);
        StopButton.Visibility = IsFinished(row.Task.Status) ? Visibility.Collapsed : Visibility.Visible;
        ResumeButton.Visibility = string.IsNullOrEmpty(row.Task.UndoState) && string.IsNullOrEmpty(row.Task.ContinuedTaskId) && (row.Task.Status is WireTaskStatus.Interrupted or WireTaskStatus.Paused) ? Visibility.Visible : Visibility.Collapsed;
        SaveSkillButton.Visibility = string.IsNullOrEmpty(row.Task.UndoState) && row.Task.Status == WireTaskStatus.Succeeded && row.Task.TotalActions > 0 ? Visibility.Visible : Visibility.Collapsed;
    }

    private void RefreshTimeline(TaskRow? row)
    {
        if (row is null) { EmptyState.Visibility = Visibility.Visible; return; }
        EmptyState.Visibility = Visibility.Collapsed;
        var messages = new List<TimelineRow>();
        var messageIndex = 0;
        foreach (var message in _messages)
        {
            var user = J(message, "role") == "user";
            var id = J(message, "id");
            messages.Add(new TimelineRow($"message:{(id.Length > 0 ? id : (++messageIndex).ToString())}", user ? "You" : "Sage", J(message, "content"), user));
        }
        if (!_messages.Any(message => J(message, "task_id") == row.TaskId && J(message, "role") == "user"))
            messages.Add(new TimelineRow($"request:{row.TaskId}", "You", row.Task.Request, user: true));
        var response = IsFinished(row.Task.Status) && !string.IsNullOrWhiteSpace(row.Task.FinalOutcome)
            ? row.Task.FinalOutcome : _streamedResponses.GetValueOrDefault(row.TaskId, row.Task.FinalOutcome);
        if (!string.IsNullOrWhiteSpace(response) &&
            !_messages.Any(message => J(message, "task_id") == row.TaskId && J(message, "role") == "assistant"))
            messages.Add(new TimelineRow($"response:{row.TaskId}", "Sage", response));
        if (string.IsNullOrWhiteSpace(row.Task.FinalOutcome) && IsFinished(row.Task.Status) && !string.IsNullOrWhiteSpace(row.Task.Summary) &&
            !_messages.Any(message => J(message, "task_id") == row.TaskId && J(message, "role") == "assistant"))
            messages.Add(new TimelineRow($"summary:{row.TaskId}", row.StatusLabel, row.Task.Summary));
        SynchronizeTimeline(_timeline, messages);

        var activity = new List<TimelineRow>();
        foreach (var action in row.Task.Actions)
            activity.Add(new TimelineRow($"action:{action.ActionId}", action.Summary, action.Status, activity: true));
        if (_timelineByTaskId.TryGetValue(row.TaskId, out var events))
        {
            var eventIndex = 0;
            foreach (var item in events.AsEnumerable().Reverse().Where(item => item.Kind != "notification"))
                activity.Add(new TimelineRow($"event:{row.TaskId}:{item.Kind}:{eventIndex++}", item.Title, item.Detail, activity: true));
        }
        SynchronizeTimeline(_activity, activity);
        ActivityExpander.Header = $"Activity ({activity.Count})";
        ActivityExpander.Visibility = activity.Count > 0 ? Visibility.Visible : Visibility.Collapsed;
        if (_followLatest && _timeline.Count > 0) Timeline.ScrollIntoView(_timeline[^1]);
    }

    private void TaskList_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_synchronizingTasks) return;
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
            await _client.ControlTaskAsync(ControlScopeFor(row.TaskId), ControlTask.Types.Operation.Cancel);
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
            await _client.UndoAsync(row.TaskId, row.Task.UndoActionId);
        }
        catch (Exception error)
        {
            await ShowErrorAsync(error.Message);
        }
    }

    private void Composer_TextChanged(object sender, TextChangedEventArgs e)
    {
        UpdateComposerActions();
        PrepareIntent();
    }

    private async void PrepareIntent()
    {
        _intentPreparation?.Cancel();
        _intentPreparation?.Dispose();
        var revision = ++_intentRevision;
        if (IntentCard is null || Composer is null) return;
        IntentCard.Visibility = Visibility.Collapsed;
        var text = Composer.Text;
        if (!_connected || _closed) return;
        if (string.IsNullOrWhiteSpace(text) || System.Text.Encoding.UTF8.GetByteCount(text) > 4096) text = "";
        _intentPreparation = new CancellationTokenSource();
        var token = _intentPreparation.Token;
        var input = new UpdateIntent { StreamId = _intentStreamId, Revision = revision, Text = text };
        if (_taskFolder is { } folder) input.FolderRoots.Add(folder);
        try
        {
            if (text.Length > 0) await Task.Delay(120, token);
            token.ThrowIfCancellationRequested();
            await _client.UpdateIntentAsync(input);
        }
        catch (OperationCanceledException) { }
        catch { /* Optional preparation leaves the draft and submission intact. */ }
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
        if (!_connected || request.Length == 0 || _isSubmitting) return;
        var supersedesTaskId = IsSelectedTaskActive() ? _selectedTaskId ?? "" : "";
        _isSubmitting = true;
        _unconfirmedRequest = request;
        _draftActive = false;
        Composer.Text = string.Empty;
        UpdateComposerActions();
        try
        {
            _selectedConversationId ??= Guid.NewGuid().ToString();
            await _client.SubmitTaskAsync(request, _selectedConversationId, _taskFolder, supersedesTaskId);
            _taskFolder = null;
            UpdateFolderScope();
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

    private async void UnlockHistory_Click(object sender, RoutedEventArgs e)
    {
        try { await _client.UnlockStorageAsync(); } catch (Exception error) { await ShowErrorAsync(error.Message); }
    }
    private string? _taskFolder;
    private async void Scope_Click(object sender, RoutedEventArgs e)
    {
        var picker = new Windows.Storage.Pickers.FolderPicker();
        picker.FileTypeFilter.Add("*");
        WinRT.Interop.InitializeWithWindow.Initialize(picker, WinRT.Interop.WindowNative.GetWindowHandle(this));
        var folder = await picker.PickSingleFolderAsync();
        if (folder is null) return;
        var dialog = new ContentDialog { XamlRoot = Root.XamlRoot, Title = "Allow this task to read files?",
            Content = folder.Path + "\nChanges still need approval.", PrimaryButtonText = "Allow reading", CloseButtonText = "Cancel" };
        if (await ShowDialogAsync(dialog) == ContentDialogResult.Primary) { _taskFolder = folder.Path; UpdateFolderScope(); }
    }

    private void TaskRow_PointerEntered(object sender, PointerRoutedEventArgs e)
    {
        if ((sender as FrameworkElement)?.DataContext is TaskRow row) row.SetHovering(true);
    }

    private void TaskRow_PointerExited(object sender, PointerRoutedEventArgs e)
    {
        if ((sender as FrameworkElement)?.DataContext is TaskRow row) row.SetHovering(false);
    }

    private void TaskOptionsButton_GotFocus(object sender, RoutedEventArgs e)
    {
        if ((sender as FrameworkElement)?.DataContext is TaskRow row) row.SetOptionsFocused(true);
    }

    private void TaskOptionsButton_LostFocus(object sender, RoutedEventArgs e)
    {
        if ((sender as FrameworkElement)?.DataContext is TaskRow row) row.SetOptionsFocused(false);
    }

    private void TaskOptions_Click(object sender, RoutedEventArgs e)
    {
        if (sender is not FrameworkElement button || button.DataContext is not TaskRow) return;
        FindAncestor<Border>(button)?.ContextFlyout?.ShowAt(button);
        e.Handled = true;
    }

    private static T? FindAncestor<T>(DependencyObject? element) where T : DependencyObject
    {
        while (element is not null)
        {
            if (element is T ancestor) return ancestor;
            element = VisualTreeHelper.GetParent(element);
        }
        return null;
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
        if (await ShowDialogAsync(dialog) != ContentDialogResult.Primary) return;
        var title = editor.Text.Trim();
        if (title.Length == 0) return;
        _taskMetadata.SetTitle(row.TaskId, title);
        await _client.KnowledgeAsync(new KnowledgeCommand { Operation = "conversation", Id = row.Task.ConversationId, Content = title, Pinned = row.IsPinned });
        row.SetPresentation(_taskMetadata.Get(row.TaskId));
        ReorderTasks();
        if (_selectedTaskId == row.TaskId) SetTaskHeader(row);
    }

    private async void TogglePin_Click(object sender, RoutedEventArgs e)
    {
        if ((sender as MenuFlyoutItem)?.Tag is not TaskRow row) return;
        _taskMetadata.TogglePinned(row.TaskId);
        await _client.KnowledgeAsync(new KnowledgeCommand { Operation = "conversation", Id = row.Task.ConversationId, Content = row.Request, Pinned = !row.IsPinned });
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
        if (await ShowDialogAsync(dialog) != ContentDialogResult.Primary) return;
        if (!IsFinished(row.Task.Status))
        {
            try { await _client.ControlTaskAsync(ControlScopeFor(row.TaskId), ControlTask.Types.Operation.Cancel); }
            catch { /* The history entry can still be removed locally. */ }
        }
        _taskMetadata.MarkDeleted(row.TaskId);
        await _client.KnowledgeAsync(new KnowledgeCommand { Operation = "conversation", Id = row.Task.ConversationId, Content = row.Request, Pinned = row.IsPinned, Archived = true });
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
        var panel = new StackPanel { Spacing = 10, MinWidth = 380 };
        panel.Children.Add(new TextBlock
        {
            Text = "Sage first-party local inference is in development. The selected target is Qwen3.5-4B. No hosted model API or external inference executable is enabled; open-ended model tasks remain unavailable until numerical, quality, memory, latency, and hardware qualification completes.",
            TextWrapping = TextWrapping.Wrap,
        });
        _memoryPanel = new StackPanel { Spacing = 10 };
        _workflowPanel = new StackPanel { Spacing = 10 };
        panel.Children.Add(_memoryPanel);
        panel.Children.Add(_workflowPanel);
        RenderMemories();
        await _client.KnowledgeAsync(new KnowledgeCommand { Operation = "list", ConversationId = _selectedConversationId ?? "" });
        await _client.WorkflowAsync(new WorkflowCommand { Operation = "list" });
        var dialog = new ContentDialog
        {
            XamlRoot = Root.XamlRoot,
            Title = "Settings",
            Content = new ScrollViewer { Content = panel, MaxHeight = 600 },
            PrimaryButtonText = "Done",
            CloseButtonText = "Back",
            DefaultButton = ContentDialogButton.Primary,
        };
        await ShowDialogAsync(dialog);
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
        var panel = new StackPanel { Spacing = 12 };
        panel.Children.Add(new TextBlock { Text = content, TextWrapping = TextWrapping.Wrap });
        var stop = new Button { Content = "Stop task" };
        stop.Click += async (_, _) => {
            try
            {
                await _client.ControlTaskAsync(ControlScopeFor(approval.TaskId), ControlTask.Types.Operation.Cancel);
                ScheduleSnapshotRefresh();
            }
            catch (Exception error)
            {
                panel.Children.Add(new TextBlock { Text = error.Message, TextWrapping = TextWrapping.Wrap });
            }
        };
        panel.Children.Add(stop);
        var dialog = new ContentDialog
        {
            XamlRoot = Root.XamlRoot,
            Title = approval.Title,
            Content = panel,
            PrimaryButtonText = "Approve once",
            SecondaryButtonText = "Later",
            CloseButtonText = "Deny",
            DefaultButton = ContentDialogButton.Close,
        };
        var result = await ShowDialogAsync(dialog, approval.ApprovalId);
        if (!_decisionInbox.ContainsKey(approval.ApprovalId)) return;
        if (result == ContentDialogResult.Secondary) { _deferredDecisions.Add(approval.ApprovalId); return; }
        var approve = result == ContentDialogResult.Primary;
        var authenticated = false;
        if (approve && approval.RequiresNativeAuthentication)
        {
            authenticated = await NativeAuthentication.AuthenticateAsync(approval.Explanation);
            approve = authenticated;
        }
        await _client.ResolveApprovalAsync(approval, approve, authenticated);
        _submittedDecisions.Add(approval.ApprovalId);
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
            SecondaryButtonText = "Later",
            CloseButtonText = "Stop task",
            DefaultButton = ContentDialogButton.Primary,
        };
        dialog.IsPrimaryButtonEnabled = false;
        answer.TextChanged += (_, _) => dialog.IsPrimaryButtonEnabled = !string.IsNullOrWhiteSpace(answer.Text);
        var result = await ShowDialogAsync(dialog, question.QuestionId);
        if (!_decisionInbox.ContainsKey(question.QuestionId)) return;
        if (result == ContentDialogResult.Secondary) { _deferredDecisions.Add(question.QuestionId); return; }
        if (result == ContentDialogResult.Primary)
        {
            await _client.AnswerAsync(question, answer.Text);
        }
        else
        {
            await _client.ControlTaskAsync(ControlScopeFor(question.TaskId), ControlTask.Types.Operation.Cancel);
            ScheduleSnapshotRefresh();
        }
        _submittedDecisions.Add(question.QuestionId);
    }

    private void QueueDecision(ApprovalRequest approval)
    {
        AddDecision(new(approval.ApprovalId, approval.TaskId, approval.ExpiresAtUnixMs, approval, null));
        RefreshDecisionQueue();
    }

    private void QueueDecision(QuestionRequest question)
    {
        AddDecision(new(question.QuestionId, question.TaskId, question.ExpiresAtUnixMs, null, question));
        RefreshDecisionQueue();
    }

    private void AddDecision(DecisionPrompt decision)
    {
        if (decision.ExpiresAt > DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() && !_submittedDecisions.Contains(decision.Id))
            _decisionInbox[decision.Id] = decision;
    }

    private void RemoveDecision(string id)
    {
        _decisionInbox.Remove(id);
        _deferredDecisions.Remove(id);
        _submittedDecisions.Remove(id);
        RefreshDecisionQueue();
    }

    private void RefreshDecisionQueue()
    {
        if (_activeDecisionId is not null && !_decisionInbox.ContainsKey(_activeDecisionId)) _activeDecisionDialog?.Hide();
        ReviewDecisionsButton.Content = $"Review requests ({_decisionInbox.Count})";
        ReviewDecisionsButton.Visibility = _decisionInbox.Count == 0 ? Visibility.Collapsed : Visibility.Visible;
        _ = ProcessDecisionQueueAsync();
    }

    private void ReviewDecisions_Click(object sender, RoutedEventArgs e)
    {
        _deferredDecisions.Clear();
        RefreshDecisionQueue();
    }

    private async Task ProcessDecisionQueueAsync()
    {
        if (_processingDecisions) return;
        _processingDecisions = true;
        try
        {
            while (_decisionInbox.Values.Where(item => !_deferredDecisions.Contains(item.Id)).OrderBy(item => item.ExpiresAt).FirstOrDefault() is { } decision)
            {
                if (decision.ExpiresAt <= DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()) { _decisionInbox.Remove(decision.Id); continue; }
                try
                {
                    if (decision.Approval is not null) await ShowApprovalAsync(decision.Approval);
                    else if (decision.Question is not null) await ShowQuestionAsync(decision.Question);
                    if (!_deferredDecisions.Contains(decision.Id)) _decisionInbox.Remove(decision.Id);
                }
                catch (Exception error)
                {
                    _deferredDecisions.Add(decision.Id);
                    await ShowErrorAsync(error.Message);
                }
            }
        }
        finally
        {
            _processingDecisions = false;
            ReviewDecisionsButton.Content = $"Review requests ({_decisionInbox.Count})";
            ReviewDecisionsButton.Visibility = _decisionInbox.Count == 0 ? Visibility.Collapsed : Visibility.Visible;
        }
    }

    private async Task<ContentDialogResult> ShowDialogAsync(ContentDialog dialog, string? decisionId = null)
    {
        await _dialogGate.WaitAsync();
        try
        {
            if (decisionId is not null)
            {
                if (!_decisionInbox.ContainsKey(decisionId)) return ContentDialogResult.None;
                _activeDecisionId = decisionId;
                _activeDecisionDialog = dialog;
            }
            return await dialog.ShowAsync();
        }
        finally
        {
            if (decisionId is not null) { _activeDecisionId = null; _activeDecisionDialog = null; }
            _dialogGate.Release();
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
        await ShowDialogAsync(dialog);
    }

    private string ControlScopeFor(string taskId)
    {
        var scope = _tasks.FirstOrDefault(row => row.TaskId == taskId)?.Task.ControlScopeId;
        return string.IsNullOrEmpty(scope) ? taskId : scope;
    }

    private TaskRow? SelectedTask => _tasks.FirstOrDefault(task => task.TaskId == _selectedTaskId);

    private bool IsSelectedTaskActive() => SelectedTask is { } task && !IsFinished(task.Task.Status);

    private void UpdateComposerActions()
    {
        var active = IsSelectedTaskActive();
        StopButton.Visibility = active ? Visibility.Visible : Visibility.Collapsed;
        SendButton.Visibility = active && string.IsNullOrWhiteSpace(Composer.Text) ? Visibility.Collapsed : Visibility.Visible;
        SendButton.IsEnabled = _connected && !_isSubmitting
            && !string.IsNullOrWhiteSpace(Composer.Text);
        Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(SendButton, active ? "Update request" : "Send");
        ToolTipService.SetToolTip(SendButton, active ? "Update request and stop the previous plan" : "Send (Enter)");
        if (_isSubmitting) SendButton.IsEnabled = false;
        StopButton.IsEnabled = _connected;
        ScopeButton.IsEnabled = !_isSubmitting;
        RemoveScopeButton.IsEnabled = !_isSubmitting;
        ConnectionStatus.Text = _connected ? "Core connected" : _reconnecting ? "Connecting to Sage" : "Core disconnected";
        ConnectionDot.Fill = new SolidColorBrush(_connected ? Color.FromArgb(255, 147, 206, 195) : Color.FromArgb(255, 151, 151, 151));
        ModelStatus.Text = "Sage local inference is in development";
        ToolTipService.SetToolTip(ModelStatus, "No hosted API or external model runtime is enabled");
        SetupNotice.IsOpen = _connected && !_storageLocked;
    }

    private void ReorderTasks()
    {
        var ordered = _tasks.OrderByDescending(task => task.IsPinned).ToList();
        for (var index = 0; index < ordered.Count; index++)
        {
            var currentIndex = _tasks.IndexOf(ordered[index]);
            if (currentIndex != index) _tasks.Move(currentIndex, index);
        }
        RefreshVisibleTasks();
    }

    private static bool IsShiftDown() =>
        InputKeyboardSource.GetKeyStateForCurrentThread(VirtualKey.Shift).HasFlag(CoreVirtualKeyStates.Down);

    private static bool IsFinished(WireTaskStatus status) => status is
        WireTaskStatus.Succeeded or WireTaskStatus.Answered or WireTaskStatus.Partial or WireTaskStatus.Failed or WireTaskStatus.Cancelled or WireTaskStatus.Interrupted;
}

public sealed class TaskRow : INotifyPropertyChanged
{
    private bool _selected;
    private bool _hovering;
    private bool _optionsFocused;
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
    public string AccessibleName => $"{Request}, {StatusLabel}{(_pinned ? ", pinned" : "")}";
    public string PinMenuLabel => _pinned ? "Unpin" : "Pin";
    public Visibility PinVisibility => _pinned ? Visibility.Visible : Visibility.Collapsed;
    public double OptionsOpacity => _hovering || _optionsFocused || _selected ? 1 : 0;
    public string StatusLabel => string.IsNullOrEmpty(Task.ContinuedTaskId) || Task.Status == WireTaskStatus.Cancelled ? FormatStatus(Task.Status) : "Continued";
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
        OnPropertyChanged(nameof(OptionsOpacity));
    }

    public void SetHovering(bool hovering)
    {
        _hovering = hovering;
        UpdateVisuals();
        OnPropertyChanged(nameof(RowBackground));
        OnPropertyChanged(nameof(OptionsOpacity));
    }

    public void SetOptionsFocused(bool focused)
    {
        _optionsFocused = focused;
        OnPropertyChanged(nameof(OptionsOpacity));
    }

    private void UpdateVisuals()
    {
        RowBackground = new SolidColorBrush(_selected ? Color.FromArgb(28, 147, 206, 195) : _hovering ? Color.FromArgb(16, 255, 255, 255) : Colors.Transparent);
    }

    private void NotifyPresentationChanged()
    {
        OnPropertyChanged(nameof(Task));
        OnPropertyChanged(nameof(Request));
        OnPropertyChanged(nameof(AccessibleName));
        OnPropertyChanged(nameof(PinMenuLabel));
        OnPropertyChanged(nameof(PinVisibility));
        OnPropertyChanged(nameof(OptionsOpacity));
        OnPropertyChanged(nameof(StatusLabel));
        OnPropertyChanged(nameof(RowBackground));
        OnPropertyChanged(nameof(IsPinned));
    }

    private static string FormatStatus(WireTaskStatus status) => status switch
    {
        WireTaskStatus.WaitingForApproval => "Needs approval",
        WireTaskStatus.WaitingForUser => "Waiting for you",
        WireTaskStatus.Succeeded => "Actions verified",
        WireTaskStatus.Answered => "Answered",
        WireTaskStatus.Partial => "Partially completed",
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
