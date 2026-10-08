import Foundation

/// The transport sequence is assigned in the same serial critical section that
/// enqueues the complete frame. Awaiting an OS completion never holds a thread.
final class OrderedFrameWriter: @unchecked Sendable {
    enum Failure: Error { case queueFull, sequenceExhausted }
    typealias Completion = @Sendable (Result<Void, Error>) -> Void
    private let queue = DispatchQueue(label: "com.ivanpadeliya.sage.ipc.writer")
    private let maximumPending: Int
    private var sequence: UInt64 = 0
    private var pending = 0

    init(maximumPending: Int = 128) { self.maximumPending = maximumPending }

    func write(
        encode: @escaping @Sendable (UInt64) throws -> Data,
        transmit: @escaping @Sendable (Data, @escaping Completion) -> Void
    ) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            queue.async {
                guard self.pending < self.maximumPending else {
                    continuation.resume(throwing: Failure.queueFull)
                    return
                }
                guard self.sequence < UInt64.max else {
                    continuation.resume(throwing: Failure.sequenceExhausted)
                    return
                }
                self.sequence += 1
                do {
                    let data = try encode(self.sequence)
                    self.pending += 1
                    transmit(data) { result in
                        self.queue.async {
                            self.pending -= 1
                            continuation.resume(with: result)
                        }
                    }
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
}
