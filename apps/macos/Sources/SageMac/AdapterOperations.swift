import Foundation

/// Per-session worker ownership. The reader never awaits an OS operation. A
/// cancellation received before admission also prevents that request starting.
final class AdapterOperations: @unchecked Sendable {
    enum Failure: Error { case invalidRequest, duplicateRequest, capacity, disconnected }
    private struct Entry {
        var task: Task<Void, Never>?
        var cancelled: Bool
        let expiry: Int64
    }
    private let lock = NSLock()
    private var entries: [String: Entry] = [:]
    private var closed = false
    private let maximumActive: Int

    init(maximumActive: Int = 16) { self.maximumActive = maximumActive }

    /// False means Stop arrived before this request. Callers still return a
    /// final failure receipt so the broker can reconcile the dispatch.
    func start(id: String, expiresAt: Int64, work: @escaping @Sendable () async -> Void) throws -> Bool {
        try lock.withLock {
            guard !closed else { throw Failure.disconnected }
            guard UUID(uuidString: id) != nil else { throw Failure.invalidRequest }
            retireFinished()
            if let prior = entries[id] {
                if prior.cancelled && prior.task == nil { return false }
                throw Failure.duplicateRequest
            }
            guard entries.count < 4096, entries.values.filter({ $0.task != nil }).count < maximumActive else { throw Failure.capacity }
            let task = Task.detached { [weak self] in
                await work()
                self?.finished(id)
            }
            entries[id] = Entry(task: task, cancelled: false, expiry: expiresAt)
            return true
        }
    }

    func cancel(id: String, expiresAt: Int64) throws {
        let task: Task<Void, Never>? = try lock.withLock {
            guard !closed else { throw Failure.disconnected }
            guard UUID(uuidString: id) != nil else { throw Failure.invalidRequest }
            retireFinished()
            if var existing = entries[id] {
                existing.cancelled = true
                entries[id] = existing
                return existing.task
            }
            guard entries.count < 4096 else { throw Failure.capacity }
            entries[id] = Entry(task: nil, cancelled: true, expiry: expiresAt)
            return nil
        }
        task?.cancel()
    }

    func close() {
        let tasks = lock.withLock {
            closed = true
            let tasks = entries.values.compactMap(\.task)
            entries.removeAll()
            return tasks
        }
        for task in tasks { task.cancel() }
    }

    private func finished(_ id: String) {
        lock.withLock { entries[id]?.task = nil }
    }

    private func retireFinished() {
        let now = Int64(Date().timeIntervalSince1970 * 1000)
        // No active operation is evicted, even if the underlying OS call has
        // ignored its deadline. Such work still belongs to this session.
        entries = entries.filter { $0.value.task != nil || $0.value.expiry >= now - 60_000 }
    }
}
