namespace Sage.Windows;

// Session-scoped ownership, independent of the pipe reader and UI dispatcher.
internal sealed class AdapterOperations
{
    private sealed record Entry(CancellationTokenSource? Cancellation, bool Cancelled, long ExpiresAt);
    private readonly object _gate = new();
    private readonly Dictionary<string, Entry> _entries = new();
    private bool _closed;

    public bool Start(string id, long expiresAt, Func<CancellationToken, Task> work)
    {
        lock (_gate)
        {
            if (_closed) throw new IOException("Worker session disconnected");
            if (!Guid.TryParse(id, out _)) throw new InvalidDataException("Invalid worker request identity");
            RetireFinished();
            if (_entries.TryGetValue(id, out var prior))
            {
                if (prior.Cancelled && prior.Cancellation is null) return false;
                throw new InvalidDataException("Duplicate worker request identity");
            }
            if (_entries.Count >= 4096 || _entries.Values.Count(item => item.Cancellation is not null) >= 16)
                throw new InvalidOperationException("Worker capacity reached before execution");
            var cancellation = new CancellationTokenSource();
            _entries[id] = new(cancellation, false, expiresAt);
            _ = Task.Run(async () =>
            {
                try { await work(cancellation.Token); }
                finally
                {
                    lock (_gate)
                        if (_entries.TryGetValue(id, out var entry)) _entries[id] = entry with { Cancellation = null };
                    cancellation.Dispose();
                }
            });
            return true;
        }
    }

    public void Cancel(string id, long expiresAt)
    {
        CancellationTokenSource? cancellation;
        lock (_gate)
        {
            if (_closed) throw new IOException("Worker session disconnected");
            if (!Guid.TryParse(id, out _)) throw new InvalidDataException("Invalid worker request identity");
            RetireFinished();
            if (_entries.TryGetValue(id, out var entry))
            {
                _entries[id] = entry with { Cancelled = true };
                cancellation = entry.Cancellation;
            }
            else
            {
                if (_entries.Count >= 4096) throw new InvalidOperationException("Cancellation capacity exhausted");
                _entries[id] = new(null, true, expiresAt);
                cancellation = null;
            }
        }
        TryCancel(cancellation);
    }

    public void Close()
    {
        CancellationTokenSource?[] pending;
        lock (_gate)
        {
            _closed = true;
            pending = _entries.Values.Select(item => item.Cancellation).ToArray();
            _entries.Clear();
        }
        foreach (var cancellation in pending) TryCancel(cancellation);
    }

    private static void TryCancel(CancellationTokenSource? cancellation)
    {
        try { cancellation?.Cancel(); }
        catch (ObjectDisposedException) { /* The operation finished concurrently. */ }
    }

    private void RetireFinished()
    {
        var now = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        foreach (var id in _entries.Where(pair => pair.Value.Cancellation is null && pair.Value.ExpiresAt < now - 60_000).Select(pair => pair.Key).ToArray())
            _entries.Remove(id);
    }
}
