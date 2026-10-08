// Compile with OrderedFrameWriter.swift and PendingSubmissions.swift.
import Foundation

final class FrameTranscript: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [UInt64] = []
    func append(_ value: UInt64) { lock.withLock { values.append(value) } }
    func snapshot() -> [UInt64] { lock.withLock { values } }
}

@main
struct WriterSmoke {
    static func main() async throws {
        let writer = OrderedFrameWriter(maximumPending: 2048)
        let transcript = FrameTranscript()
        try await withThrowingTaskGroup(of: Void.self) { group in
            for _ in 0..<1000 {
                group.addTask {
                    try await writer.write(encode: { Data(String($0).utf8) }, transmit: { bytes, complete in
                        let sequence = UInt64(String(decoding: bytes, as: UTF8.self))!
                        transcript.append(sequence)
                        // Deliberately acknowledge out of order. Transmission,
                        // rather than completion scheduling, owns the sequence.
                        DispatchQueue.global().asyncAfter(deadline: .now() + .milliseconds(Int(sequence % 7))) {
                            complete(.success(()))
                        }
                    })
                }
            }
            try await group.waitForAll()
        }
        precondition(transcript.snapshot() == Array(UInt64(1)...1000), "Concurrent frames reordered")

        let bounded = OrderedFrameWriter(maximumPending: 1)
        let (receipts, continuation) = AsyncStream<OrderedFrameWriter.Completion>.makeStream()
        let held = Task {
            try await bounded.write(encode: { Data(String($0).utf8) }, transmit: { _, receipt in
                continuation.yield(receipt)
            })
        }
        var iterator = receipts.makeAsyncIterator()
        let release = await iterator.next()!
        do {
            try await bounded.write(encode: { Data(String($0).utf8) }, transmit: { _, receipt in receipt(.success(())) })
            preconditionFailure("Pending frame bound was bypassed")
        } catch OrderedFrameWriter.Failure.queueFull { }
        release(.success(()))
        try await held.value
        continuation.finish()

        let pending = PendingSubmissions()
        let intent = Data("same task and same scopes".utf8)
        let first = try pending.stage(intent)
        try await withThrowingTaskGroup(of: String.self) { group in
            for _ in 0..<100 { group.addTask { try pending.stage(intent).requestID } }
            for try await id in group { precondition(id == first.requestID, "Retry acquired a new identity") }
        }
        precondition(pending.snapshot().count == 1)
        // The transport can disappear without dropping the pending intent.
        let retried = pending.snapshot()[0]
        precondition(retried.payload == intent && retried.requestID == first.requestID)
        pending.resolve(first.requestID)
        let distinct = try pending.stage(intent)
        precondition(distinct.requestID != first.requestID, "A new, intentional request reused a completed identity")
        for number in 1..<16 { _ = try pending.stage(Data("request-\(number)".utf8)) }
        do { _ = try pending.stage(Data("overflow".utf8)); preconditionFailure("Outbox is unbounded") }
        catch PendingSubmissions.Failure.full { }
        print("Passed: 1,000 ordered concurrent frames, reordered acknowledgements, bounded writes, and stable bounded submission retries.")
    }
}
