using System.ComponentModel;
using System.Runtime.CompilerServices;
using Microsoft.UI;
using Microsoft.UI.Dispatching;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using Sage.Ipc.V2;
using System.Collections.ObjectModel;
using Windows.UI;

namespace Sage.Windows;

public sealed partial class MainWindow
{
    private readonly Dictionary<string, string> _streamedResponses = [];

    private Task RequestSnapshotAsync() => _client.RequestStateAsync(includeCompleted: true);

    private void ScheduleSnapshotRefresh()
    {
        if (!_connected || _closed) return;
        _snapshotRefresh.Request();
        _snapshotTimer.Start();
    }

    private async void SnapshotTimer_Tick(DispatcherQueueTimer sender, object args)
    {
        if (_closed || !_connected) { sender.Stop(); return; }
        if (_snapshotRefresh.HasTimedOut(Environment.TickCount64))
        {
            sender.Stop();
            _snapshotRefresh.Reset();
            await ShowErrorAsync("Sage could not refresh its state. Reconnect to check the latest task result.");
            return;
        }
        if (_snapshotRefresh.TryStart(Environment.TickCount64))
        {
            try { await RequestSnapshotAsync(); }
            catch (Exception error)
            {
                sender.Stop();
                _snapshotRefresh.Reset();
                await ShowErrorAsync(error.Message);
            }
        }
        else if (!_snapshotRefresh.InFlight && !_snapshotRefresh.Dirty) sender.Stop();
    }

    private void ResponseTimer_Tick(DispatcherQueueTimer sender, object args)
    {
        sender.Stop();
        ModelResponseDelta[] responses;
        lock (_responseLock)
        {
            responses = _pendingResponses.Values.ToArray();
            _pendingResponses.Clear();
            _responseDispatchQueued = false;
        }
        foreach (var response in responses)
        {
            _streamedResponses[response.TaskId] = response.Text;
            if (_streamedResponses.Count > 256)
            {
                var oldest = _streamedResponses.Keys.FirstOrDefault(id => id != _selectedTaskId && id != response.TaskId);
                if (oldest is not null) _streamedResponses.Remove(oldest);
            }
            if (response.TaskId == _selectedTaskId)
            {
                RefreshTimeline(SelectedTask);
                if (response.Finished) _ = RequestConversationAsync(_selectedConversationId ?? "");
            }
        }
    }

    private async Task RequestConversationAsync(string conversationId)
    {
        if (string.IsNullOrEmpty(conversationId)) return;
        try { await _client.KnowledgeAsync(new KnowledgeCommand { Operation = "list", ConversationId = conversationId }); }
        catch (Exception error) { await ShowErrorAsync(error.Message); }
    }

    private void RefreshVisibleTasks()
    {
        if (HistorySearch is null) return;
        var query = HistorySearch.Text.Trim();
        var visible = _tasks.Where(row => query.Length == 0 || row.Request.Contains(query, StringComparison.CurrentCultureIgnoreCase)
            || row.Task.Request.Contains(query, StringComparison.CurrentCultureIgnoreCase)).ToList();
        SynchronizeRows(_visibleTasks, visible);
        TaskList.SelectedItem = visible.FirstOrDefault(row => row.TaskId == _selectedTaskId);
    }

    private void SynchronizeRows(ObservableCollection<TaskRow> target, List<TaskRow> desired)
    {
        _synchronizingTasks = true;
        try
        {
            for (var index = target.Count - 1; index >= 0; index--)
                if (!desired.Contains(target[index])) target.RemoveAt(index);
            for (var index = 0; index < desired.Count; index++)
            {
                var old = target.IndexOf(desired[index]);
                if (old < 0) target.Insert(index, desired[index]);
                else if (old != index) target.Move(old, index);
            }
        }
        finally { _synchronizingTasks = false; }
    }

    private static void SynchronizeTimeline(ObservableCollection<TimelineRow> target, List<TimelineRow> desired)
    {
        var keys = desired.Select(row => row.Key).ToHashSet();
        for (var index = target.Count - 1; index >= 0; index--)
            if (!keys.Contains(target[index].Key)) target.RemoveAt(index);
        for (var index = 0; index < desired.Count; index++)
        {
            var row = target.FirstOrDefault(row => row.Key == desired[index].Key);
            if (row is null) target.Insert(index, desired[index]);
            else
            {
                row.Update(desired[index]);
                var old = target.IndexOf(row);
                if (old != index) target.Move(old, index);
            }
        }
    }

    private static string ConversationKey(TaskUpdate task) => string.IsNullOrEmpty(task.ConversationId) ? task.TaskId : task.ConversationId;
    private void HistorySearch_TextChanged(object sender, TextChangedEventArgs e) => RefreshVisibleTasks();
    private void NewChatAccelerator_Invoked(KeyboardAccelerator sender, KeyboardAcceleratorInvokedEventArgs e)
    { SetNewTaskState(focusComposer: true); e.Handled = true; }
    private void SearchAccelerator_Invoked(KeyboardAccelerator sender, KeyboardAcceleratorInvokedEventArgs e)
    { HistorySearch.Focus(FocusState.Programmatic); e.Handled = true; }
    private void ComposerAccelerator_Invoked(KeyboardAccelerator sender, KeyboardAcceleratorInvokedEventArgs e)
    { Composer.Focus(FocusState.Programmatic); e.Handled = true; }
    private void Starter_Click(object sender, RoutedEventArgs e)
    { Composer.Text = (sender as Button)?.Tag as string ?? ""; Composer.Focus(FocusState.Programmatic); }
    private void RemoveScope_Click(object sender, RoutedEventArgs e)
    { _taskFolder = null; UpdateFolderScope(); }
    private void UpdateFolderScope()
    {
        PrepareIntent();
        ScopeLabel.Text = string.IsNullOrEmpty(_taskFolder) ? "Add folder" : Path.GetFileName(_taskFolder.TrimEnd(Path.DirectorySeparatorChar));
        if (string.IsNullOrEmpty(ScopeLabel.Text)) ScopeLabel.Text = _taskFolder;
        RemoveScopeButton.Visibility = string.IsNullOrEmpty(_taskFolder) ? Visibility.Collapsed : Visibility.Visible;
        ToolTipService.SetToolTip(ScopeButton, _taskFolder is null ? "Choose a folder this task can read" : $"This request can read {_taskFolder}");
    }

    private void Timeline_Loaded(object sender, RoutedEventArgs e)
    {
        if (_timelineScrollViewer is not null) return;
        _timelineScrollViewer = FindScrollViewer(Timeline);
        if (_timelineScrollViewer is not null)
            _timelineScrollViewer.ViewChanged += (_, _) =>
                _followLatest = _timelineScrollViewer.ScrollableHeight - _timelineScrollViewer.VerticalOffset < 48;
    }

    private static ScrollViewer? FindScrollViewer(DependencyObject parent)
    {
        if (parent is ScrollViewer viewer) return viewer;
        for (var index = 0; index < VisualTreeHelper.GetChildrenCount(parent); index++)
            if (FindScrollViewer(VisualTreeHelper.GetChild(parent, index)) is { } child) return child;
        return null;
    }
}

/// <summary>A single-flight state refresh lane. Event bursts request one trailing refresh.</summary>
internal sealed class SnapshotRefreshGate
{
    public bool Dirty { get; private set; }
    public bool InFlight { get; private set; }
    private long _sentAt;

    public void Request() => Dirty = true;
    public bool TryStart(long now)
    {
        if (!Dirty || InFlight) return false;
        Dirty = false;
        InFlight = true;
        _sentAt = now;
        return true;
    }
    public bool HasTimedOut(long now) => InFlight && now - _sentAt >= 10_000;
    public void Complete() => InFlight = false;
    public void Reset() { Dirty = false; InFlight = false; }
}

/// <summary>Stable UI identities keep streaming text from replacing the conversation list.</summary>
public sealed class TimelineRow : INotifyPropertyChanged
{
    public string Key { get; }
    public string Title { get; private set; }
    public string Detail { get; private set; }
    public string Glyph { get; }
    public Brush Background { get; }

    public TimelineRow(string key, string title, string detail, bool user = false, bool activity = false)
    {
        Key = key;
        Title = title;
        Detail = detail;
        Glyph = key.Contains("reference_context", StringComparison.Ordinal) ? "\uE8A5" : activity ? "\uE9D9" : user ? "\uE77B" : "\uE945";
        Background = new SolidColorBrush(user ? Color.FromArgb(16, 147, 206, 195) : Colors.Transparent);
    }

    public void Update(TimelineRow next)
    {
        if (Title != next.Title) { Title = next.Title; Changed(nameof(Title)); }
        if (Detail != next.Detail) { Detail = next.Detail; Changed(nameof(Detail)); }
    }

    public event PropertyChangedEventHandler? PropertyChanged;
    private void Changed([CallerMemberName] string? name = null) => PropertyChanged?.Invoke(this, new(name));
}
