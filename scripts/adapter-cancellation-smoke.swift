// Compile with AdapterOperations.swift. No app launch or OS permission is used.
import Foundation

@main
struct CancellationSmoke {
    static func main() async throws {
        let watchdog = Task {
            try await Task.sleep(for: .seconds(5))
            fatalError("Worker cancellation did not settle within five seconds")
        }
        defer { watchdog.cancel() }
        let operations = AdapterOperations(maximumActive: 1)
        let expiry = Int64(Date().timeIntervalSince1970 * 1000) + 30_000
        let id = UUID().uuidString
        let (started, starting) = AsyncStream<Bool>.makeStream()
        let (finished, finishing) = AsyncStream<Bool>.makeStream()
        let admitted = try operations.start(id: id, expiresAt: expiry) {
            starting.yield(true)
            do { try await Task.sleep(for: .seconds(30)) } catch { }
            finishing.yield(Task.isCancelled)
        }
        precondition(admitted)
        var startIterator = started.makeAsyncIterator()
        _ = await startIterator.next()
        do {
            _ = try operations.start(id: UUID().uuidString, expiresAt: expiry) { preconditionFailure("Capacity bound was bypassed") }
            preconditionFailure("Capacity bound was bypassed")
        } catch AdapterOperations.Failure.capacity { }
        try operations.cancel(id: id, expiresAt: expiry)
        var finishIterator = finished.makeAsyncIterator()
        let cancelled = await finishIterator.next()
        precondition(cancelled == true)
        operations.close()

        let reordered = AdapterOperations()
        let queuedID = UUID().uuidString
        try reordered.cancel(id: queuedID, expiresAt: expiry)
        let startedAfterStop = try reordered.start(id: queuedID, expiresAt: expiry) { preconditionFailure("A cancelled queued operation started") }
        precondition(!startedAfterStop)
        reordered.close()
        do {
            _ = try reordered.start(id: UUID().uuidString, expiresAt: expiry) { preconditionFailure("Closed session started an operation") }
            preconditionFailure("Closed session admitted work")
        } catch AdapterOperations.Failure.disconnected { }

        let session = AdapterOperations()
        let (owned, ownership) = AsyncStream<Bool>.makeStream()
        let (closed, closing) = AsyncStream<Bool>.makeStream()
        for _ in 0..<4 {
            let admitted = try session.start(id: UUID().uuidString, expiresAt: expiry) {
                ownership.yield(true)
                do { try await Task.sleep(for: .seconds(30)) } catch { }
                closing.yield(Task.isCancelled)
            }
            precondition(admitted)
        }
        var ownedIterator = owned.makeAsyncIterator()
        for _ in 0..<4 { _ = await ownedIterator.next() }
        session.close()
        var closedIterator = closed.makeAsyncIterator()
        for _ in 0..<4 {
            let cancelled = await closedIterator.next()
            precondition(cancelled == true)
        }
        print("Passed: active cancellation, Stop before admission, bounded workers, and cancellation of all workers on disconnect.")
    }
}
